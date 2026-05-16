//! CoreML framework error-domain constants and codes.

/// Objective-C `MLModelErrorDomain` constant.
pub const ML_MODEL_ERROR_DOMAIN: &str = "com.apple.CoreML";

/// Framework-defined `MLModelError` codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum MLModelError {
    /// Unexpected framework-level failure.
    Generic = 0,
    /// Wrong feature type supplied to a model input.
    FeatureType = 1,
    /// File or transport I/O failure.
    Io = 3,
    /// Custom-layer lookup or execution failure.
    CustomLayer = 4,
    /// Custom-model lookup or execution failure.
    CustomModel = 5,
    /// Updated-model save/write failure.
    Update = 6,
    /// Unsupported parameter lookup/update failure.
    Parameters = 7,
    /// Model decryption key fetch failure.
    ModelDecryptionKeyFetch = 8,
    /// Model decryption failure.
    ModelDecryption = 9,
    /// Model-collection deployment failure.
    ModelCollection = 10,
    /// Prediction cancelled before completion.
    PredictionCancelled = 11,
}

impl MLModelError {
    /// Convert a numeric NSError code into a known `MLModelError` value.
    #[must_use]
    pub const fn from_code(code: i32) -> Option<Self> {
        match code {
            0 => Some(Self::Generic),
            1 => Some(Self::FeatureType),
            3 => Some(Self::Io),
            4 => Some(Self::CustomLayer),
            5 => Some(Self::CustomModel),
            6 => Some(Self::Update),
            7 => Some(Self::Parameters),
            8 => Some(Self::ModelDecryptionKeyFetch),
            9 => Some(Self::ModelDecryption),
            10 => Some(Self::ModelCollection),
            11 => Some(Self::PredictionCancelled),
            _ => None,
        }
    }

    /// Numeric NSError code published by CoreML.
    #[must_use]
    pub const fn code(self) -> i32 {
        self as i32
    }
}
