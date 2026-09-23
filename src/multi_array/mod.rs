//! Safe wrapper for CoreML `MLMultiArray` tensors.

use core::ffi::c_void;
use core::marker::{PhantomData, PhantomPinned};
use core::ops::{Deref, DerefMut};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::ptr::{self, NonNull};
use std::thread;

use half::f16;
use serde::{Deserialize, Serialize};

use crate::error::{from_swift, CoreMLError};
use crate::ffi;

/// Supported CoreML multi-array element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DataType {
    /// 64-bit IEEE-754 float.
    Float64,
    /// 32-bit IEEE-754 float.
    Float32,
    /// 16-bit IEEE-754 float.
    Float16,
    /// 32-bit signed integer.
    Int32,
    Int8,
    Unknown(i64),
}

impl DataType {
    pub(crate) const fn as_ffi(self) -> i64 {
        match self {
            Self::Float64 => 0x10000 | 64,
            Self::Float32 => 0x10000 | 32,
            Self::Float16 => 0x10000 | 16,
            Self::Int32 => 0x20000 | 32,
            Self::Int8 => 0x20000 | 8,
            Self::Unknown(raw) => raw,
        }
    }

    pub(crate) const fn from_ffi(raw: i64) -> Self {
        match raw {
            65_600 => Self::Float64,
            65_568 => Self::Float32,
            65_552 => Self::Float16,
            131_104 => Self::Int32,
            131_080 => Self::Int8,
            other => Self::Unknown(other),
        }
    }

    #[must_use]
    pub const fn element_size(self) -> Option<usize> {
        match self {
            Self::Float64 => Some(8),
            Self::Float32 | Self::Int32 => Some(4),
            Self::Float16 => Some(2),
            Self::Int8 => Some(1),
            Self::Unknown(_) => None,
        }
    }
}

/// Dynamically typed scalar used by the NSNumber-style helpers.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum MultiArrayScalar {
    /// 64-bit float.
    Float64(f64),
    /// 32-bit float.
    Float32(f32),
    /// 16-bit float.
    Float16(f16),
    /// 32-bit integer.
    Int32(i32),
    Int8(i8),
}

impl From<f64> for MultiArrayScalar {
    fn from(value: f64) -> Self {
        Self::Float64(value)
    }
}

impl From<f32> for MultiArrayScalar {
    fn from(value: f32) -> Self {
        Self::Float32(value)
    }
}

impl From<f16> for MultiArrayScalar {
    fn from(value: f16) -> Self {
        Self::Float16(value)
    }
}

impl From<i32> for MultiArrayScalar {
    fn from(value: i32) -> Self {
        Self::Int32(value)
    }
}

impl From<i8> for MultiArrayScalar {
    fn from(value: i8) -> Self {
        Self::Int8(value)
    }
}

mod sealed {
    pub trait Sealed {}
}

pub trait MultiArrayElement: sealed::Sealed + Copy + 'static {
    const DATA_TYPE: DataType;
}

macro_rules! multi_array_element {
    ($element:ty, $data_type:expr) => {
        impl sealed::Sealed for $element {}

        impl MultiArrayElement for $element {
            const DATA_TYPE: DataType = $data_type;
        }
    };
}

multi_array_element!(f64, DataType::Float64);
multi_array_element!(f32, DataType::Float32);
multi_array_element!(f16, DataType::Float16);
multi_array_element!(i32, DataType::Int32);
multi_array_element!(i8, DataType::Int8);

#[repr(C)]
pub struct MultiArrayRef {
    _opaque: [u8; 0],
    _marker: PhantomData<(*mut c_void, PhantomPinned)>,
}

impl MultiArrayRef {
    pub(crate) unsafe fn from_raw<'a>(ptr: NonNull<c_void>) -> &'a Self {
        unsafe { &*ptr.as_ptr().cast::<Self>() }
    }

    pub(crate) unsafe fn from_raw_mut<'a>(ptr: NonNull<c_void>) -> &'a mut Self {
        unsafe { &mut *ptr.as_ptr().cast::<Self>() }
    }

    pub(crate) fn as_ptr(&self) -> *mut c_void {
        ptr::from_ref(self).cast_mut().cast()
    }

    /// Logical shape of the multi-array.
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        self.copy_i64_vector(
            unsafe { ffi::cm_multi_array_rank(self.as_ptr()) },
            ffi::cm_multi_array_copy_shape,
        )
    }

    /// Raw strides reported by CoreML.
    #[must_use]
    pub fn strides(&self) -> Vec<usize> {
        self.copy_i64_vector(
            unsafe { ffi::cm_multi_array_rank(self.as_ptr()) },
            ffi::cm_multi_array_copy_strides,
        )
    }

    /// Total number of addressable scalar elements.
    #[must_use]
    pub fn len(&self) -> usize {
        unsafe { ffi::cm_multi_array_count(self.as_ptr()) }
    }

    /// Whether the array contains zero elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of scalar elements spanned by the backing storage, accounting for
    /// the array's `strides`.
    ///
    /// For a contiguous array this equals [`Self::len`] (the product of the
    /// shape). For a non-contiguous (strided) array it is the extent required to
    /// address every logical element given the reported strides, computed as
    /// `1 + Σ (dimᵢ - 1) * strideᵢ`. The storage slices handed to
    /// [`Self::with_slice`] can be longer than this; using the naive product of
    /// dimensions would under-size them and misaddress strided elements.
    #[must_use]
    pub fn storage_len(&self) -> usize {
        storage_extent_from(&self.shape(), &self.strides()).unwrap_or_else(|| self.len())
    }

    /// Whether the backing storage is tightly packed (contains no stride
    /// padding), i.e. the storage extent equals the logical element count.
    ///
    /// When this returns `true`, a raw storage slice can be treated as densely
    /// laid out in memory order. When it returns `false`, callers must address
    /// elements through the strides or the strided accessors (e.g. [`Self::get`]).
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.storage_len() == self.len()
    }

    /// Scalar data type.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        DataType::from_ffi(unsafe { ffi::cm_multi_array_data_type(self.as_ptr()) })
    }

    pub fn with_bytes<R>(&self, body: impl FnOnce(&[u8], &[usize]) -> R) -> Result<R, CoreMLError> {
        access_storage(self.as_ptr(), false, |data, len, strides| {
            Ok(body(unsafe { bytes_from_raw(data, len) }, strides))
        })
    }

    pub fn with_bytes_mut<R>(
        &mut self,
        body: impl FnOnce(&mut [u8], &[usize]) -> R,
    ) -> Result<R, CoreMLError> {
        access_storage(self.as_ptr(), true, |data, len, strides| {
            Ok(body(unsafe { bytes_from_raw_mut(data, len) }, strides))
        })
    }

    pub fn with_slice<T: MultiArrayElement, R>(
        &self,
        body: impl FnOnce(&[T], &[usize]) -> R,
    ) -> Result<R, CoreMLError> {
        self.expect_element::<T>()?;
        access_storage(self.as_ptr(), false, |data, len, strides| {
            let bytes = unsafe { bytes_from_raw(data, len) };
            Ok(body(typed_slice(bytes)?, strides))
        })
    }

    pub fn with_slice_mut<T: MultiArrayElement, R>(
        &mut self,
        body: impl FnOnce(&mut [T], &[usize]) -> R,
    ) -> Result<R, CoreMLError> {
        self.expect_element::<T>()?;
        access_storage(self.as_ptr(), true, |data, len, strides| {
            let bytes = unsafe { bytes_from_raw_mut(data, len) };
            Ok(body(typed_slice_mut(bytes)?, strides))
        })
    }

    pub fn get<T: MultiArrayElement>(&self, indices: &[usize]) -> Result<T, CoreMLError> {
        self.expect_element::<T>()?;
        let shape = self.shape();
        access_storage(self.as_ptr(), false, |data, len, strides| {
            let bytes = unsafe { bytes_from_raw(data, len) };
            read_element(bytes, element_offset(indices, &shape, strides)?)
        })
    }

    pub fn set<T: MultiArrayElement>(
        &mut self,
        indices: &[usize],
        value: T,
    ) -> Result<(), CoreMLError> {
        self.expect_element::<T>()?;
        let shape = self.shape();
        access_storage(self.as_ptr(), true, |data, len, strides| {
            let bytes = unsafe { bytes_from_raw_mut(data, len) };
            write_element(bytes, element_offset(indices, &shape, strides)?, value)
        })
    }

    pub fn to_vec<T: MultiArrayElement>(&self) -> Result<Vec<T>, CoreMLError> {
        self.expect_element::<T>()?;
        let shape = self.shape();
        if element_count(&shape)? == 0 {
            return Ok(Vec::new());
        }
        access_storage(self.as_ptr(), false, |data, len, strides| {
            gather(unsafe { bytes_from_raw(data, len) }, &shape, strides)
        })
    }

    pub fn copy_from_slice<T: MultiArrayElement>(
        &mut self,
        source: &[T],
    ) -> Result<(), CoreMLError> {
        self.expect_element::<T>()?;
        let shape = self.shape();
        let count = element_count(&shape)?;
        if source.len() != count {
            return Err(CoreMLError::MultiArrayFailed(format!(
                "source slice length ({}) does not match the multi-array element count ({count})",
                source.len()
            )));
        }
        if count == 0 {
            return Ok(());
        }
        access_storage(self.as_ptr(), true, |data, len, strides| {
            scatter(unsafe { bytes_from_raw_mut(data, len) }, &shape, strides, source)
        })
    }

    pub fn copy_to_owned(&self) -> Result<MultiArray, CoreMLError> {
        let data_type = self.data_type();
        let element_size = data_type.element_size().ok_or_else(|| {
            CoreMLError::TypeMismatch(format!(
                "cannot copy an MLMultiArray with unsupported data type {data_type:?}"
            ))
        })?;
        let shape = self.shape();
        let mut copy = MultiArray::new(&shape, data_type)?;
        if element_count(&shape)? == 0 {
            return Ok(copy);
        }
        let target: &mut MultiArrayRef = &mut copy;
        access_storage(self.as_ptr(), false, |source, source_len, source_strides| {
            let source = unsafe { bytes_from_raw(source, source_len) };
            access_storage(target.as_ptr(), true, |destination, destination_len, destination_strides| {
                copy_strided(
                    source,
                    source_strides,
                    unsafe { bytes_from_raw_mut(destination, destination_len) },
                    destination_strides,
                    &shape,
                    element_size,
                )
            })
        })?;
        Ok(copy)
    }

    /// Transfer the contents into another multi-array, allowing data-type or stride changes.
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes differ, the runtime is too old, or CoreML rejects the
    /// destination array.
    pub fn transfer_to(&self, destination: &mut MultiArrayRef) -> Result<(), CoreMLError> {
        let source_shape = self.shape();
        let destination_shape = destination.shape();
        if source_shape != destination_shape {
            return Err(CoreMLError::InvalidArgument(format!(
                "cannot transfer a {source_shape:?} MLMultiArray into a {destination_shape:?} one"
            )));
        }
        for data_type in [self.data_type(), destination.data_type()] {
            if data_type.element_size().is_none() {
                return Err(CoreMLError::TypeMismatch(format!(
                    "cannot transfer an MLMultiArray with unsupported data type {data_type:?}"
                )));
            }
        }
        let mut error = ptr::null_mut();
        let status = unsafe {
            ffi::cm_multi_array_transfer_to(self.as_ptr(), destination.as_ptr(), &raw mut error)
        };
        if status != ffi::status::OK {
            return Err(from_swift(status, error));
        }
        Ok(())
    }

    /// Retrieve one scalar using C-style linear indexing.
    #[must_use]
    pub fn number_at_linear_index(&self, index: usize) -> Option<MultiArrayScalar> {
        let indices = linear_index_to_indices(index, &self.shape())?;
        self.number_at_indices(&indices)
    }

    /// Set one scalar using C-style linear indexing.
    ///
    /// # Errors
    ///
    /// Returns an error when the index is out of bounds.
    pub fn set_number_at_linear_index(
        &mut self,
        index: usize,
        value: impl Into<MultiArrayScalar>,
    ) -> Result<(), CoreMLError> {
        let indices = linear_index_to_indices(index, &self.shape()).ok_or_else(|| {
            CoreMLError::IndexOutOfRange(
                "linear index is outside the multi-array bounds".to_owned(),
            )
        })?;
        self.set_number_at_indices(&indices, value)
    }

    /// Retrieve one scalar using logical indices.
    #[must_use]
    pub fn number_at_indices(&self, indices: &[usize]) -> Option<MultiArrayScalar> {
        match self.data_type() {
            DataType::Float64 => self.get(indices).ok().map(MultiArrayScalar::Float64),
            DataType::Float32 => self.get(indices).ok().map(MultiArrayScalar::Float32),
            DataType::Float16 => self.get(indices).ok().map(MultiArrayScalar::Float16),
            DataType::Int32 => self.get(indices).ok().map(MultiArrayScalar::Int32),
            DataType::Int8 => self.get(indices).ok().map(MultiArrayScalar::Int8),
            DataType::Unknown(_) => None,
        }
    }

    /// Set one scalar using logical indices.
    ///
    /// # Errors
    ///
    /// Returns an error when the indices are out of bounds.
    pub fn set_number_at_indices(
        &mut self,
        indices: &[usize],
        value: impl Into<MultiArrayScalar>,
    ) -> Result<(), CoreMLError> {
        let value = value.into();
        match self.data_type() {
            DataType::Float64 => self.set(indices, scalar_as_f64(value)),
            DataType::Float32 => self.set(indices, scalar_as_f64(value) as f32),
            DataType::Float16 => self.set(indices, f16::from_f64(scalar_as_f64(value))),
            DataType::Int32 => self.set(indices, scalar_as_i32(value)),
            DataType::Int8 => self.set(indices, saturate_i8(scalar_as_i32(value))),
            data_type @ DataType::Unknown(_) => Err(CoreMLError::TypeMismatch(format!(
                "cannot store a scalar into an MLMultiArray with unsupported data type {data_type:?}"
            ))),
        }
    }

    fn expect_element<T: MultiArrayElement>(&self) -> Result<(), CoreMLError> {
        let data_type = self.data_type();
        if data_type == T::DATA_TYPE {
            Ok(())
        } else {
            Err(CoreMLError::TypeMismatch(format!(
                "MLMultiArray stores {data_type:?} elements, not {:?}",
                T::DATA_TYPE
            )))
        }
    }

    fn copy_i64_vector(
        &self,
        len: usize,
        copy_fn: unsafe extern "C" fn(*mut c_void, *mut i64, usize) -> usize,
    ) -> Vec<usize> {
        let mut buffer = vec![0_i64; len];
        let copied = unsafe { copy_fn(self.as_ptr(), buffer.as_mut_ptr(), buffer.len()) };
        buffer.truncate(copied);
        buffer
            .into_iter()
            .map(|value| usize::try_from(value).ok().unwrap_or_default())
            .collect()
    }
}

impl AsRef<Self> for MultiArrayRef {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl core::fmt::Debug for MultiArrayRef {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MultiArray")
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("data_type", &self.data_type())
            .field("len", &self.len())
            .finish()
    }
}

/// Owned wrapper around an `MLMultiArray`.
pub struct MultiArray {
    ptr: NonNull<c_void>,
}

impl MultiArray {
    /// Create a new `Float32` multi-array.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML cannot allocate the array.
    pub fn new_f32(shape: &[usize]) -> Result<Self, CoreMLError> {
        Self::new(shape, DataType::Float32)
    }

    /// Create a new `Float16` multi-array.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML cannot allocate the array.
    pub fn new_f16(shape: &[usize]) -> Result<Self, CoreMLError> {
        Self::new(shape, DataType::Float16)
    }

    /// Create a new `Int32` multi-array.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML cannot allocate the array.
    pub fn new_i32(shape: &[usize]) -> Result<Self, CoreMLError> {
        Self::new(shape, DataType::Int32)
    }

    /// Create a new `Float64` multi-array.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML cannot allocate the array.
    pub fn new_f64(shape: &[usize]) -> Result<Self, CoreMLError> {
        Self::new(shape, DataType::Float64)
    }

    /// Create a new multi-array with the requested element type.
    ///
    /// # Errors
    ///
    /// Returns an error if CoreML cannot allocate the array.
    pub fn new(shape: &[usize], data_type: DataType) -> Result<Self, CoreMLError> {
        let element_size = data_type.element_size().ok_or_else(|| {
            CoreMLError::InvalidArgument(format!(
                "cannot allocate an MLMultiArray with unsupported data type {data_type:?}"
            ))
        })?;
        element_count(shape)?
            .checked_mul(element_size)
            .ok_or_else(|| {
                CoreMLError::InvalidArgument(format!(
                    "multi-array shape {shape:?} overflows the addressable byte count"
                ))
            })?;
        let shape_i64 = shape
            .iter()
            .map(|&dimension| {
                i64::try_from(dimension).map_err(|_| {
                    CoreMLError::InvalidArgument(format!(
                        "multi-array dimension {dimension} does not fit in Int64"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut error = ptr::null_mut();
        let mut out = ptr::null_mut();
        let status = unsafe {
            ffi::cm_multi_array_new(
                shape_i64.as_ptr(),
                shape_i64.len(),
                data_type.as_ffi(),
                &raw mut out,
                &raw mut error,
            )
        };
        if status != ffi::status::OK {
            if !out.is_null() {
                unsafe { ffi::cm_object_release(out) };
            }
            return Err(from_swift(status, error));
        }
        unsafe { Self::from_retained(out) }.ok_or_else(|| from_swift(status, error))
    }

    /// Concatenate several multi-arrays into a new array.
    ///
    /// # Errors
    ///
    /// Returns an error when the input arrays are shape-incompatible or CoreML rejects the request.
    pub fn concatenate<A: AsRef<MultiArrayRef>>(
        arrays: &[A],
        axis: isize,
        data_type: DataType,
    ) -> Result<Self, CoreMLError> {
        let Some(first) = arrays.first() else {
            return Err(CoreMLError::InvalidArgument(
                "at least one MLMultiArray is required for concatenation".to_owned(),
            ));
        };
        if data_type.element_size().is_none() {
            return Err(CoreMLError::InvalidArgument(format!(
                "cannot concatenate into unsupported data type {data_type:?}"
            )));
        }
        let reference_shape = first.as_ref().shape();
        if reference_shape.is_empty() {
            return Err(CoreMLError::InvalidArgument(
                "cannot concatenate scalar MLMultiArray values".to_owned(),
            ));
        }
        let axis = Self::normalize_axis(axis, reference_shape.len())?;
        for array in arrays {
            let array = array.as_ref();
            let element_type = array.data_type();
            if element_type.element_size().is_none() {
                return Err(CoreMLError::TypeMismatch(format!(
                    "cannot concatenate an MLMultiArray with unsupported data type {element_type:?}"
                )));
            }
            let shape = array.shape();
            if shape.len() != reference_shape.len() {
                return Err(CoreMLError::InvalidArgument(
                    "all MLMultiArray values must have the same rank for concatenation".to_owned(),
                ));
            }
            for dimension in 0..reference_shape.len() {
                if dimension != axis && shape[dimension] != reference_shape[dimension] {
                    return Err(CoreMLError::InvalidArgument(format!(
                        "all MLMultiArray dimensions except axis {axis} must match"
                    )));
                }
            }
        }

        let ptrs: Vec<*mut c_void> = arrays.iter().map(|array| array.as_ref().as_ptr()).collect();
        let mut error = ptr::null_mut();
        let mut out = ptr::null_mut();
        let status = unsafe {
            ffi::cm_multi_array_concat(
                ptrs.as_ptr(),
                ptrs.len(),
                axis as isize,
                data_type.as_ffi(),
                &raw mut out,
                &raw mut error,
            )
        };
        if status != ffi::status::OK {
            if !out.is_null() {
                unsafe { ffi::cm_object_release(out) };
            }
            return Err(from_swift(status, error));
        }
        unsafe { Self::from_retained(out) }.ok_or_else(|| from_swift(status, error))
    }

    pub(crate) unsafe fn from_retained(ptr: *mut c_void) -> Option<Self> {
        NonNull::new(ptr).map(|ptr| Self { ptr })
    }

    fn normalize_axis(axis: isize, rank: usize) -> Result<usize, CoreMLError> {
        if rank == 0 {
            return Err(CoreMLError::InvalidArgument(
                "cannot select an axis for a rank-0 MLMultiArray".to_owned(),
            ));
        }
        Ok(axis.rem_euclid(rank as isize) as usize)
    }
}

impl Deref for MultiArray {
    type Target = MultiArrayRef;

    fn deref(&self) -> &MultiArrayRef {
        unsafe { MultiArrayRef::from_raw(self.ptr) }
    }
}

impl DerefMut for MultiArray {
    fn deref_mut(&mut self) -> &mut MultiArrayRef {
        unsafe { MultiArrayRef::from_raw_mut(self.ptr) }
    }
}

impl AsRef<MultiArrayRef> for MultiArray {
    fn as_ref(&self) -> &MultiArrayRef {
        self
    }
}

impl Drop for MultiArray {
    fn drop(&mut self) {
        unsafe { ffi::cm_object_release(self.ptr.as_ptr()) };
    }
}

impl core::fmt::Debug for MultiArray {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(&**self, f)
    }
}

pub struct MultiArrayView<'a> {
    array: MultiArray,
    _source: PhantomData<&'a ()>,
}

impl MultiArrayView<'_> {
    pub(crate) unsafe fn from_retained(ptr: *mut c_void) -> Option<Self> {
        unsafe { MultiArray::from_retained(ptr) }.map(|array| Self {
            array,
            _source: PhantomData,
        })
    }
}

impl Deref for MultiArrayView<'_> {
    type Target = MultiArrayRef;

    fn deref(&self) -> &MultiArrayRef {
        &self.array
    }
}

impl AsRef<MultiArrayRef> for MultiArrayView<'_> {
    fn as_ref(&self) -> &MultiArrayRef {
        self
    }
}

impl core::fmt::Debug for MultiArrayView<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(&**self, f)
    }
}

struct StorageAccess<F, R> {
    body: Option<F>,
    outcome: Option<thread::Result<Result<R, CoreMLError>>>,
}

unsafe extern "C" fn storage_access_trampoline<F, R>(
    data: *mut c_void,
    len: usize,
    strides: *const isize,
    rank: usize,
    context: *mut c_void,
) where
    F: FnOnce(*mut u8, usize, &[usize]) -> Result<R, CoreMLError>,
{
    let Some(access) = (unsafe { context.cast::<StorageAccess<F, R>>().as_mut() }) else {
        return;
    };
    let Some(body) = access.body.take() else {
        return;
    };
    access.outcome = Some(catch_unwind(AssertUnwindSafe(|| {
        let strides = unsafe { strides_from_raw(strides, rank) }?;
        body(data.cast(), len, &strides)
    })));
}

fn access_storage<F, R>(array: *mut c_void, mutable: bool, body: F) -> Result<R, CoreMLError>
where
    F: FnOnce(*mut u8, usize, &[usize]) -> Result<R, CoreMLError>,
{
    let mut access = StorageAccess {
        body: Some(body),
        outcome: None,
    };
    let status = unsafe {
        ffi::cm_multi_array_access_bytes(
            array,
            mutable,
            storage_access_trampoline::<F, R>,
            (&raw mut access).cast(),
        )
    };
    match access.outcome {
        Some(Ok(result)) => result,
        Some(Err(payload)) => resume_unwind(payload),
        None if status != ffi::status::OK => Err(CoreMLError::MultiArrayFailed(format!(
            "MLMultiArray storage access failed with status {status}"
        ))),
        None => Err(CoreMLError::MultiArrayFailed(
            "MLMultiArray did not provide its storage".to_owned(),
        )),
    }
}

unsafe fn strides_from_raw(strides: *const isize, rank: usize) -> Result<Vec<usize>, CoreMLError> {
    if rank == 0 {
        return Ok(Vec::new());
    }
    if strides.is_null() {
        return Err(CoreMLError::MultiArrayFailed(
            "MLMultiArray storage access reported no strides".to_owned(),
        ));
    }
    unsafe { std::slice::from_raw_parts(strides, rank) }
        .iter()
        .map(|&stride| {
            usize::try_from(stride).map_err(|_| {
                CoreMLError::MultiArrayFailed(format!(
                    "MLMultiArray reported an unsupported negative stride {stride}"
                ))
            })
        })
        .collect()
}

unsafe fn bytes_from_raw<'a>(data: *mut u8, len: usize) -> &'a [u8] {
    if data.is_null() || len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }
    }
}

unsafe fn bytes_from_raw_mut<'a>(data: *mut u8, len: usize) -> &'a mut [u8] {
    if data.is_null() || len == 0 {
        &mut []
    } else {
        unsafe { std::slice::from_raw_parts_mut(data, len) }
    }
}

#[allow(clippy::cast_ptr_alignment)]
fn typed_slice<T: MultiArrayElement>(bytes: &[u8]) -> Result<&[T], CoreMLError> {
    if bytes.is_empty() {
        return Ok(&[]);
    }
    check_alignment::<T>(bytes.as_ptr())?;
    let len = bytes.len() / core::mem::size_of::<T>();
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), len) })
}

#[allow(clippy::cast_ptr_alignment)]
fn typed_slice_mut<T: MultiArrayElement>(bytes: &mut [u8]) -> Result<&mut [T], CoreMLError> {
    if bytes.is_empty() {
        return Ok(&mut []);
    }
    check_alignment::<T>(bytes.as_ptr())?;
    let len = bytes.len() / core::mem::size_of::<T>();
    Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<T>(), len) })
}

fn check_alignment<T>(data: *const u8) -> Result<(), CoreMLError> {
    if data.align_offset(core::mem::align_of::<T>()) == 0 {
        Ok(())
    } else {
        Err(CoreMLError::MultiArrayFailed(format!(
            "MLMultiArray storage is not aligned for {}",
            core::any::type_name::<T>()
        )))
    }
}

fn element_count(shape: &[usize]) -> Result<usize, CoreMLError> {
    shape
        .iter()
        .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
        .ok_or_else(|| {
            CoreMLError::InvalidArgument(format!(
                "multi-array shape {shape:?} overflows the addressable element count"
            ))
        })
}

fn element_offset(
    indices: &[usize],
    shape: &[usize],
    strides: &[usize],
) -> Result<usize, CoreMLError> {
    if indices.len() != shape.len() {
        return Err(CoreMLError::IndexOutOfRange(format!(
            "expected {} indices for a rank-{} MLMultiArray, got {}",
            shape.len(),
            shape.len(),
            indices.len()
        )));
    }
    if indices.iter().zip(shape).any(|(&index, &dimension)| index >= dimension) {
        return Err(CoreMLError::IndexOutOfRange(
            "indices are outside the multi-array shape".to_owned(),
        ));
    }
    storage_offset(indices, strides)
}

fn storage_offset(indices: &[usize], strides: &[usize]) -> Result<usize, CoreMLError> {
    if indices.len() != strides.len() {
        return Err(CoreMLError::MultiArrayFailed(format!(
            "MLMultiArray reported {} strides for rank {}",
            strides.len(),
            indices.len()
        )));
    }
    indices
        .iter()
        .zip(strides)
        .try_fold(0_usize, |offset, (&index, &stride)| {
            index
                .checked_mul(stride)
                .and_then(|term| offset.checked_add(term))
        })
        .ok_or_else(|| {
            CoreMLError::MultiArrayFailed("MLMultiArray element offset overflowed".to_owned())
        })
}

fn element_byte_range<T>(offset: usize, len: usize) -> Result<core::ops::Range<usize>, CoreMLError> {
    let size = core::mem::size_of::<T>();
    offset
        .checked_mul(size)
        .and_then(|start| start.checked_add(size).map(|end| start..end))
        .filter(|range| range.end <= len)
        .ok_or_else(|| {
            CoreMLError::MultiArrayFailed(
                "MLMultiArray element offset lies outside its storage".to_owned(),
            )
        })
}

fn read_element<T: MultiArrayElement>(bytes: &[u8], offset: usize) -> Result<T, CoreMLError> {
    let range = element_byte_range::<T>(offset, bytes.len())?;
    Ok(unsafe { ptr::read_unaligned(bytes[range].as_ptr().cast::<T>()) })
}

fn write_element<T: MultiArrayElement>(
    bytes: &mut [u8],
    offset: usize,
    value: T,
) -> Result<(), CoreMLError> {
    let range = element_byte_range::<T>(offset, bytes.len())?;
    unsafe { ptr::write_unaligned(bytes[range].as_mut_ptr().cast::<T>(), value) };
    Ok(())
}

fn for_each_index(
    shape: &[usize],
    mut visit: impl FnMut(&[usize]) -> Result<(), CoreMLError>,
) -> Result<(), CoreMLError> {
    let count = element_count(shape)?;
    let mut indices = vec![0_usize; shape.len()];
    for _ in 0..count {
        visit(&indices)?;
        for dimension in (0..shape.len()).rev() {
            indices[dimension] += 1;
            if indices[dimension] < shape[dimension] {
                break;
            }
            indices[dimension] = 0;
        }
    }
    Ok(())
}

fn is_c_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let mut expected = 1_usize;
    for (&dimension, &stride) in shape.iter().zip(strides).rev() {
        if dimension > 1 && stride != expected {
            return false;
        }
        match expected.checked_mul(dimension) {
            Some(next) => expected = next,
            None => return false,
        }
    }
    true
}

fn gather<T: MultiArrayElement>(
    bytes: &[u8],
    shape: &[usize],
    strides: &[usize],
) -> Result<Vec<T>, CoreMLError> {
    let count = element_count(shape)?;
    if is_c_contiguous(shape, strides) {
        let byte_len = count
            .checked_mul(core::mem::size_of::<T>())
            .filter(|&byte_len| byte_len <= bytes.len())
            .ok_or_else(|| {
                CoreMLError::MultiArrayFailed(
                    "MLMultiArray storage is shorter than its shape".to_owned(),
                )
            })?;
        let mut values = Vec::<T>::with_capacity(count);
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), values.as_mut_ptr().cast::<u8>(), byte_len);
            values.set_len(count);
        }
        return Ok(values);
    }
    let mut values = Vec::with_capacity(count);
    for_each_index(shape, |indices| {
        values.push(read_element(bytes, storage_offset(indices, strides)?)?);
        Ok(())
    })?;
    Ok(values)
}

fn scatter<T: MultiArrayElement>(
    bytes: &mut [u8],
    shape: &[usize],
    strides: &[usize],
    source: &[T],
) -> Result<(), CoreMLError> {
    let mut values = source.iter();
    for_each_index(shape, |indices| {
        let value = values.next().ok_or_else(|| {
            CoreMLError::MultiArrayFailed("source slice ended before the multi-array".to_owned())
        })?;
        write_element(bytes, storage_offset(indices, strides)?, *value)
    })
}

fn copy_strided(
    source: &[u8],
    source_strides: &[usize],
    destination: &mut [u8],
    destination_strides: &[usize],
    shape: &[usize],
    element_size: usize,
) -> Result<(), CoreMLError> {
    let byte_range = |offset: usize, len: usize| {
        offset
            .checked_mul(element_size)
            .and_then(|start| start.checked_add(element_size).map(|end| start..end))
            .filter(|range| range.end <= len)
            .ok_or_else(|| {
                CoreMLError::MultiArrayFailed(
                    "MLMultiArray element offset lies outside its storage".to_owned(),
                )
            })
    };
    for_each_index(shape, |indices| {
        let from = byte_range(storage_offset(indices, source_strides)?, source.len())?;
        let to = byte_range(storage_offset(indices, destination_strides)?, destination.len())?;
        destination[to].copy_from_slice(&source[from]);
        Ok(())
    })
}

fn linear_index_to_indices(index: usize, shape: &[usize]) -> Option<Vec<usize>> {
    if index >= element_count(shape).ok()? {
        return None;
    }
    let mut remaining = index;
    let mut indices = vec![0_usize; shape.len()];
    for (position, &dimension) in shape.iter().enumerate().rev() {
        indices[position] = remaining % dimension;
        remaining /= dimension;
    }
    Some(indices)
}

fn scalar_as_f64(value: MultiArrayScalar) -> f64 {
    match value {
        MultiArrayScalar::Float64(value) => value,
        MultiArrayScalar::Float32(value) => f64::from(value),
        MultiArrayScalar::Float16(value) => f64::from(value),
        MultiArrayScalar::Int32(value) => f64::from(value),
        MultiArrayScalar::Int8(value) => f64::from(value),
    }
}

fn scalar_as_i32(value: MultiArrayScalar) -> i32 {
    match value {
        MultiArrayScalar::Float64(value) => value as i32,
        MultiArrayScalar::Float32(value) => value as i32,
        MultiArrayScalar::Float16(value) => value.to_f32() as i32,
        MultiArrayScalar::Int32(value) => value,
        MultiArrayScalar::Int8(value) => i32::from(value),
    }
}

fn saturate_i8(value: i32) -> i8 {
    i8::try_from(value).unwrap_or(if value < 0 { i8::MIN } else { i8::MAX })
}

/// Compute the number of scalar elements spanned by the backing storage of an
/// `MLMultiArray` with the given `shape` and `strides`.
///
/// The extent is `1 + Σ (dimᵢ - 1) * strideᵢ`, which is the highest addressable
/// element offset plus one. All arithmetic is overflow-checked; on overflow or
/// a shape/stride length mismatch this returns `None` so the caller can fall
/// back to a conservative length. A rank-0 (scalar) array occupies a single
/// element, and any zero-sized dimension yields an extent of zero.
fn storage_extent_from(shape: &[usize], strides: &[usize]) -> Option<usize> {
    if shape.is_empty() {
        return Some(1);
    }
    if strides.len() != shape.len() {
        return None;
    }
    let mut extent: usize = 1;
    for (&dimension, &stride) in shape.iter().zip(strides) {
        if dimension == 0 {
            return Some(0);
        }
        let span = (dimension - 1).checked_mul(stride)?;
        extent = extent.checked_add(span)?;
    }
    Some(extent)
}

#[cfg(test)]
mod tests {
    use super::{
        copy_strided, element_offset, for_each_index, gather, is_c_contiguous,
        linear_index_to_indices, read_element, saturate_i8, scatter, storage_extent_from,
        typed_slice, DataType,
    };
    use crate::error::CoreMLError;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_ne_bytes()).collect()
    }

    #[test]
    fn data_type_raw_values_round_trip_and_keep_unknown_values() {
        for data_type in [
            DataType::Float64,
            DataType::Float32,
            DataType::Float16,
            DataType::Int32,
            DataType::Int8,
        ] {
            assert_eq!(DataType::from_ffi(data_type.as_ffi()), data_type);
        }
        assert_eq!(DataType::from_ffi(0x20000 | 8), DataType::Int8);
        assert_eq!(DataType::from_ffi(0x20000 | 16), DataType::Unknown(0x20000 | 16));
        assert_eq!(DataType::Unknown(5).as_ffi(), 5);
        assert_eq!(DataType::from_ffi(1 << 40), DataType::Unknown(1 << 40));
    }

    #[test]
    fn data_type_serde_keeps_unknown_raw_values() {
        let unknown: DataType = serde_json::from_str(r#"{"unknown":131088}"#).unwrap();
        assert_eq!(unknown, DataType::Unknown(131_088));
        let int8: DataType = serde_json::from_str(r#""int8""#).unwrap();
        assert_eq!(int8, DataType::Int8);
    }

    #[test]
    fn strided_gather_skips_padding() {
        let storage = f32_bytes(&[1.0, 2.0, 3.0, -1.0, -1.0, 4.0, 5.0, 6.0]);
        let values: Vec<f32> = gather(&storage, &[2, 3], &[5, 1]).unwrap();
        assert_eq!(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn strided_scatter_leaves_padding_untouched() {
        let mut storage = f32_bytes(&[-1.0; 8]);
        scatter(&mut storage, &[2, 3], &[5, 1], &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let values: Vec<f32> = gather(&storage, &[8], &[1]).unwrap();
        assert_eq!(values, [1.0, 2.0, 3.0, -1.0, -1.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn strided_copy_repacks_into_a_contiguous_layout() {
        let source = f32_bytes(&[1.0, -1.0, 2.0, -1.0, 3.0, -1.0, 4.0, -1.0]);
        let mut destination = vec![0_u8; 4 * 4];
        copy_strided(&source, &[4, 2], &mut destination, &[2, 1], &[2, 2], 4).unwrap();
        let values: Vec<f32> = gather(&destination, &[4], &[1]).unwrap();
        assert_eq!(values, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn offsets_outside_the_storage_are_errors_not_reads() {
        let storage = f32_bytes(&[1.0, 2.0]);
        assert!(gather::<f32>(&storage, &[2, 2], &[2, 1]).is_err());
        assert!(read_element::<f32>(&storage, 2).is_err());
        assert!(read_element::<f32>(&storage, usize::MAX).is_err());
        let mut short = vec![0_u8; 4];
        assert!(scatter(&mut short, &[2], &[1], &[1.0_f32, 2.0]).is_err());
        let mut destination = vec![0_u8; 8];
        assert!(copy_strided(&storage, &[3], &mut destination, &[1], &[2], 4).is_err());
    }

    #[test]
    fn element_offsets_validate_indices_and_overflow() {
        assert_eq!(element_offset(&[1, 2], &[2, 3], &[5, 1]).unwrap(), 7);
        assert!(matches!(
            element_offset(&[2, 0], &[2, 3], &[3, 1]),
            Err(CoreMLError::IndexOutOfRange(_))
        ));
        assert!(matches!(
            element_offset(&[0], &[2, 3], &[3, 1]),
            Err(CoreMLError::IndexOutOfRange(_))
        ));
        assert!(element_offset(&[1, 1], &[2, 2], &[usize::MAX, 1]).is_err());
        assert!(element_offset(&[1, 1], &[2, 2], &[1]).is_err());
    }

    #[test]
    fn contiguity_ignores_unit_dimensions() {
        assert!(is_c_contiguous(&[2, 3], &[3, 1]));
        assert!(is_c_contiguous(&[1, 3], &[99, 1]));
        assert!(!is_c_contiguous(&[2, 3], &[5, 1]));
        assert!(!is_c_contiguous(&[2, 3], &[1, 2]));
        assert!(!is_c_contiguous(&[2], &[1, 1]));
    }

    #[test]
    fn index_iteration_is_c_order() {
        let mut visited = Vec::new();
        for_each_index(&[2, 2], |indices| {
            visited.push(indices.to_vec());
            Ok(())
        })
        .unwrap();
        assert_eq!(visited, [[0, 0], [0, 1], [1, 0], [1, 1]]);
        let mut count = 0;
        for_each_index(&[3, 0], |_| {
            count += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(count, 0);
        assert_eq!(linear_index_to_indices(5, &[2, 3]), Some(vec![1, 2]));
        assert_eq!(linear_index_to_indices(6, &[2, 3]), None);
        assert_eq!(linear_index_to_indices(0, &[0, 3]), None);
    }

    #[test]
    fn typed_slices_require_alignment() {
        let values = [1.0_f32, 2.0, 3.0];
        let storage = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), 12) };
        assert_eq!(typed_slice::<f32>(storage).unwrap(), &[1.0, 2.0, 3.0]);
        let misaligned = &storage[1..];
        assert!(typed_slice::<f32>(misaligned).is_err());
        assert!(typed_slice::<f32>(&[]).unwrap().is_empty());
    }

    #[test]
    fn int8_conversion_saturates() {
        assert_eq!(saturate_i8(300), i8::MAX);
        assert_eq!(saturate_i8(-300), i8::MIN);
        assert_eq!(saturate_i8(-5), -5);
    }

    #[test]
    fn contiguous_extent_matches_product_of_dims() {
        // Row-major contiguous [2, 3] -> strides [3, 1], extent 6.
        assert_eq!(storage_extent_from(&[2, 3], &[3, 1]), Some(6));
        // Row-major contiguous [2, 3, 4] -> strides [12, 4, 1], extent 24.
        assert_eq!(storage_extent_from(&[2, 3, 4], &[12, 4, 1]), Some(24));
    }

    #[test]
    fn strided_extent_accounts_for_padding() {
        // Non-contiguous [2, 3] with a padded leading stride: each row occupies
        // 5 elements of storage even though only 3 are logical. Naive product
        // would report 6, but the real backing extent is 1 + 1*5 + 2*1 = 8.
        assert_eq!(storage_extent_from(&[2, 3], &[5, 1]), Some(8));
        // Inner-padded layout: stride 2 between adjacent columns.
        // extent = 1 + (2-1)*6 + (3-1)*2 = 1 + 6 + 4 = 11.
        assert_eq!(storage_extent_from(&[2, 3], &[6, 2]), Some(11));
    }

    #[test]
    fn scalar_and_zero_dim_edge_cases() {
        assert_eq!(storage_extent_from(&[], &[]), Some(1));
        assert_eq!(storage_extent_from(&[0, 4], &[4, 1]), Some(0));
    }

    #[test]
    fn mismatched_lengths_and_overflow_return_none() {
        assert_eq!(storage_extent_from(&[2, 3], &[3]), None);
        assert_eq!(storage_extent_from(&[2, 2], &[usize::MAX, 1]), None);
    }
}
