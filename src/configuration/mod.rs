//! `MLModelConfiguration` builder types.

/// CoreML compute-unit selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeUnits {
    /// Force inference onto CPU only.
    CpuOnly,
    /// Allow CPU and GPU execution.
    CpuAndGpu,
    /// Allow CPU and Apple Neural Engine execution.
    CpuAndNeuralEngine,
    /// Let CoreML use the best available compute units.
    #[default]
    All,
}

impl ComputeUnits {
    pub(crate) const fn as_ffi(self) -> i32 {
        match self {
            Self::CpuOnly => 0,
            Self::CpuAndGpu => 1,
            Self::All => 2,
            Self::CpuAndNeuralEngine => 3,
        }
    }
}

/// Safe Rust builder for `MLModelConfiguration`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelConfiguration {
    pub(crate) compute_units: ComputeUnits,
    pub(crate) allow_low_precision_accumulation_on_gpu: bool,
    pub(crate) display_name: Option<String>,
}

impl ModelConfiguration {
    /// Create a configuration with CoreML defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the compute units CoreML may use.
    #[must_use]
    pub fn with_compute_units(mut self, units: ComputeUnits) -> Self {
        self.compute_units = units;
        self
    }

    /// Allow lower-precision accumulation on GPU when supported.
    #[must_use]
    pub fn with_allow_low_precision_accumulation_on_gpu(mut self, allow: bool) -> Self {
        self.allow_low_precision_accumulation_on_gpu = allow;
        self
    }

    /// Set the human-readable display name for the loaded model instance.
    #[must_use]
    pub fn with_display_name(mut self, name: &str) -> Self {
        self.display_name = Some(name.to_owned());
        self
    }

    /// Current compute-unit selection.
    #[must_use]
    pub const fn compute_units(&self) -> ComputeUnits {
        self.compute_units
    }

    /// Whether low-precision GPU accumulation is enabled.
    #[must_use]
    pub const fn allow_low_precision_accumulation_on_gpu(&self) -> bool {
        self.allow_low_precision_accumulation_on_gpu
    }

    /// Optional display name supplied at load time.
    #[must_use]
    pub fn display_name(&self) -> Option<&str> {
        self.display_name.as_deref()
    }
}
