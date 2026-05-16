//! Safe wrapper for CoreML `MLMultiArray` tensors.

use core::ffi::c_void;
use std::ptr;

use half::f16;
use serde::{Deserialize, Serialize};

use crate::error::{from_status_message, from_swift, CoreMLError};
use crate::ffi;

/// Supported CoreML multi-array element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    /// 64-bit IEEE-754 float.
    Float64,
    /// 32-bit IEEE-754 float.
    Float32,
    /// 16-bit IEEE-754 float.
    Float16,
    /// 32-bit signed integer.
    Int32,
}

impl DataType {
    pub(crate) const fn as_ffi(self) -> i32 {
        match self {
            Self::Float64 => 0x10000 | 64,
            Self::Float32 => 0x10000 | 32,
            Self::Float16 => 0x10000 | 16,
            Self::Int32 => 0x20000 | 32,
        }
    }

    fn from_ffi(raw: i32) -> Option<Self> {
        match raw {
            65_600 => Some(Self::Float64),
            65_568 => Some(Self::Float32),
            65_552 => Some(Self::Float16),
            131_104 => Some(Self::Int32),
            _ => None,
        }
    }
}

/// Owned wrapper around an `MLMultiArray`.
pub struct MultiArray {
    pub(crate) ptr: *mut c_void,
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
        let shape_i64: Vec<i64> = shape.iter().map(|&dimension| dimension as i64).collect();
        let mut error = ptr::null_mut();
        let mut out = ptr::null_mut();
        let status = unsafe {
            ffi::cm_multi_array_new(
                shape_i64.as_ptr(),
                shape_i64.len(),
                data_type.as_ffi(),
                &mut out,
                &mut error,
            )
        };
        if status != ffi::status::OK || out.is_null() {
            return Err(from_swift(status, error));
        }
        Ok(Self { ptr: out })
    }

    /// Logical shape of the multi-array.
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        self.copy_i64_vector(unsafe { ffi::cm_multi_array_rank(self.ptr) }, ffi::cm_multi_array_copy_shape)
    }

    /// Raw strides reported by CoreML.
    #[must_use]
    pub fn strides(&self) -> Vec<usize> {
        self.copy_i64_vector(
            unsafe { ffi::cm_multi_array_rank(self.ptr) },
            ffi::cm_multi_array_copy_strides,
        )
    }

    /// Total number of addressable scalar elements.
    #[must_use]
    pub fn len(&self) -> usize {
        unsafe { ffi::cm_multi_array_count(self.ptr) }
    }

    /// Whether the array contains zero elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Scalar data type.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        let raw = unsafe { ffi::cm_multi_array_data_type(self.ptr) };
        DataType::from_ffi(raw).unwrap_or(DataType::Float32)
    }

    /// Immutable raw `Float32` storage view.
    #[must_use]
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        (self.data_type() == DataType::Float32).then(|| unsafe {
            std::slice::from_raw_parts(self.data_ptr().cast::<f32>(), self.len())
        })
    }

    /// Mutable raw `Float32` storage view.
    #[must_use]
    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        (self.data_type() == DataType::Float32).then(|| unsafe {
            std::slice::from_raw_parts_mut(self.data_ptr().cast::<f32>(), self.len())
        })
    }

    /// Immutable raw `Float16` storage view.
    #[must_use]
    pub fn as_f16_slice(&self) -> Option<&[f16]> {
        (self.data_type() == DataType::Float16).then(|| unsafe {
            std::slice::from_raw_parts(self.data_ptr().cast::<f16>(), self.len())
        })
    }

    /// Mutable raw `Float16` storage view.
    #[must_use]
    pub fn as_f16_slice_mut(&mut self) -> Option<&mut [f16]> {
        (self.data_type() == DataType::Float16).then(|| unsafe {
            std::slice::from_raw_parts_mut(self.data_ptr().cast::<f16>(), self.len())
        })
    }

    /// Immutable raw `Int32` storage view.
    #[must_use]
    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        (self.data_type() == DataType::Int32).then(|| unsafe {
            std::slice::from_raw_parts(self.data_ptr().cast::<i32>(), self.len())
        })
    }

    /// Mutable raw `Int32` storage view.
    #[must_use]
    pub fn as_i32_slice_mut(&mut self) -> Option<&mut [i32]> {
        (self.data_type() == DataType::Int32).then(|| unsafe {
            std::slice::from_raw_parts_mut(self.data_ptr().cast::<i32>(), self.len())
        })
    }

    /// Immutable raw `Float64` storage view.
    #[must_use]
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        (self.data_type() == DataType::Float64).then(|| unsafe {
            std::slice::from_raw_parts(self.data_ptr().cast::<f64>(), self.len())
        })
    }

    /// Mutable raw `Float64` storage view.
    #[must_use]
    pub fn as_f64_slice_mut(&mut self) -> Option<&mut [f64]> {
        (self.data_type() == DataType::Float64).then(|| unsafe {
            std::slice::from_raw_parts_mut(self.data_ptr().cast::<f64>(), self.len())
        })
    }

    /// Copy from a `Float32` slice into the underlying storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the type or element count do not match.
    pub fn copy_from_f32_slice(&mut self, src: &[f32]) -> Result<(), CoreMLError> {
        let Some(dst) = self.as_f32_slice_mut() else {
            return Err(CoreMLError::TypeMismatch(
                "MLMultiArray is not backed by Float32 storage".to_owned(),
            ));
        };
        Self::copy_checked(dst, src, "Float32")
    }

    /// Copy from a `Float16` slice into the underlying storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the type or element count do not match.
    pub fn copy_from_f16_slice(&mut self, src: &[f16]) -> Result<(), CoreMLError> {
        let Some(dst) = self.as_f16_slice_mut() else {
            return Err(CoreMLError::TypeMismatch(
                "MLMultiArray is not backed by Float16 storage".to_owned(),
            ));
        };
        Self::copy_checked(dst, src, "Float16")
    }

    /// Copy from an `Int32` slice into the underlying storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the type or element count do not match.
    pub fn copy_from_i32_slice(&mut self, src: &[i32]) -> Result<(), CoreMLError> {
        let Some(dst) = self.as_i32_slice_mut() else {
            return Err(CoreMLError::TypeMismatch(
                "MLMultiArray is not backed by Int32 storage".to_owned(),
            ));
        };
        Self::copy_checked(dst, src, "Int32")
    }

    /// Copy from a `Float64` slice into the underlying storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the type or element count do not match.
    pub fn copy_from_f64_slice(&mut self, src: &[f64]) -> Result<(), CoreMLError> {
        let Some(dst) = self.as_f64_slice_mut() else {
            return Err(CoreMLError::TypeMismatch(
                "MLMultiArray is not backed by Float64 storage".to_owned(),
            ));
        };
        Self::copy_checked(dst, src, "Float64")
    }

    /// Read one `Float32` element using logical indices.
    #[must_use]
    pub fn get_f32(&self, indices: &[usize]) -> Option<f32> {
        let offset = self.scalar_offset(indices)?;
        self.as_f32_slice().and_then(|slice| slice.get(offset).copied())
    }

    /// Write one `Float32` element using logical indices.
    ///
    /// # Errors
    ///
    /// Returns an error if the indices are invalid or the storage is not `Float32`.
    pub fn set_f32(&mut self, indices: &[usize], value: f32) -> Result<(), CoreMLError> {
        let offset = self.scalar_offset(indices).ok_or_else(|| {
            CoreMLError::IndexOutOfRange("indices are outside the multi-array shape".to_owned())
        })?;
        let slice = self.as_f32_slice_mut().ok_or_else(|| {
            CoreMLError::TypeMismatch("MLMultiArray is not backed by Float32 storage".to_owned())
        })?;
        slice[offset] = value;
        Ok(())
    }

    /// Read one `Float16` element using logical indices.
    #[must_use]
    pub fn get_f16(&self, indices: &[usize]) -> Option<f16> {
        let offset = self.scalar_offset(indices)?;
        self.as_f16_slice().and_then(|slice| slice.get(offset).copied())
    }

    /// Write one `Float16` element using logical indices.
    ///
    /// # Errors
    ///
    /// Returns an error if the indices are invalid or the storage is not `Float16`.
    pub fn set_f16(&mut self, indices: &[usize], value: f16) -> Result<(), CoreMLError> {
        let offset = self.scalar_offset(indices).ok_or_else(|| {
            CoreMLError::IndexOutOfRange("indices are outside the multi-array shape".to_owned())
        })?;
        let slice = self.as_f16_slice_mut().ok_or_else(|| {
            CoreMLError::TypeMismatch("MLMultiArray is not backed by Float16 storage".to_owned())
        })?;
        slice[offset] = value;
        Ok(())
    }

    /// Read one `Int32` element using logical indices.
    #[must_use]
    pub fn get_i32(&self, indices: &[usize]) -> Option<i32> {
        let offset = self.scalar_offset(indices)?;
        self.as_i32_slice().and_then(|slice| slice.get(offset).copied())
    }

    /// Write one `Int32` element using logical indices.
    ///
    /// # Errors
    ///
    /// Returns an error if the indices are invalid or the storage is not `Int32`.
    pub fn set_i32(&mut self, indices: &[usize], value: i32) -> Result<(), CoreMLError> {
        let offset = self.scalar_offset(indices).ok_or_else(|| {
            CoreMLError::IndexOutOfRange("indices are outside the multi-array shape".to_owned())
        })?;
        let slice = self.as_i32_slice_mut().ok_or_else(|| {
            CoreMLError::TypeMismatch("MLMultiArray is not backed by Int32 storage".to_owned())
        })?;
        slice[offset] = value;
        Ok(())
    }

    /// Read one `Float64` element using logical indices.
    #[must_use]
    pub fn get_f64(&self, indices: &[usize]) -> Option<f64> {
        let offset = self.scalar_offset(indices)?;
        self.as_f64_slice().and_then(|slice| slice.get(offset).copied())
    }

    /// Write one `Float64` element using logical indices.
    ///
    /// # Errors
    ///
    /// Returns an error if the indices are invalid or the storage is not `Float64`.
    pub fn set_f64(&mut self, indices: &[usize], value: f64) -> Result<(), CoreMLError> {
        let offset = self.scalar_offset(indices).ok_or_else(|| {
            CoreMLError::IndexOutOfRange("indices are outside the multi-array shape".to_owned())
        })?;
        let slice = self.as_f64_slice_mut().ok_or_else(|| {
            CoreMLError::TypeMismatch("MLMultiArray is not backed by Float64 storage".to_owned())
        })?;
        slice[offset] = value;
        Ok(())
    }

    pub(crate) const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub(crate) fn from_raw(ptr: *mut c_void) -> Option<Self> {
        (!ptr.is_null()).then_some(Self { ptr })
    }

    fn copy_i64_vector(
        &self,
        len: usize,
        copy_fn: unsafe extern "C" fn(*mut c_void, *mut i64, usize) -> usize,
    ) -> Vec<usize> {
        let mut buffer = vec![0_i64; len];
        let copied = unsafe { copy_fn(self.ptr, buffer.as_mut_ptr(), buffer.len()) };
        buffer.truncate(copied);
        buffer
            .into_iter()
            .map(|value| usize::try_from(value).ok().unwrap_or_default())
            .collect()
    }

    fn data_ptr(&self) -> *mut c_void {
        unsafe { ffi::cm_multi_array_data_pointer(self.ptr) }
    }

    fn scalar_offset(&self, indices: &[usize]) -> Option<usize> {
        let shape = self.shape();
        if indices.len() != shape.len() {
            return None;
        }
        let strides = self.strides();
        let mut offset = 0_usize;
        for ((&index, &dimension), &stride) in indices.iter().zip(&shape).zip(&strides) {
            if index >= dimension {
                return None;
            }
            offset = offset.saturating_add(index.saturating_mul(stride));
        }
        Some(offset)
    }

    fn copy_checked<T: Copy>(dst: &mut [T], src: &[T], type_name: &str) -> Result<(), CoreMLError> {
        if dst.len() != src.len() {
            return Err(from_status_message(
                ffi::status::MULTI_ARRAY_FAILED,
                format!(
                    "source slice length ({}) does not match {type_name} multi-array length ({})",
                    src.len(),
                    dst.len()
                ),
            ));
        }
        dst.copy_from_slice(src);
        Ok(())
    }
}

impl Drop for MultiArray {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::cm_object_release(self.ptr) };
        }
    }
}

impl core::fmt::Debug for MultiArray {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MultiArray")
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("data_type", &self.data_type())
            .field("len", &self.len())
            .finish()
    }
}
