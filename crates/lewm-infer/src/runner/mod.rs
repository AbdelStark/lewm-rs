//! CPU inference runners for exported `LeWM` graphs.

mod loader;
mod traits;

#[cfg(feature = "tract-nnef")]
mod tract_nnef_runner;
#[cfg(feature = "tract-onnx")]
mod tract_onnx_runner;

pub use crate::runner::loader::{detect_checkpoint_format, load};
pub use crate::runner::traits::{
    IMAGE_ELEMENT_COUNT, InferenceRunner, PREDICTOR_BATCH, RunnerError, RunnerFormat,
    RunnerMetadata,
};

#[cfg(feature = "tract-nnef")]
pub use crate::runner::tract_nnef_runner::TractNnefRunner;
#[cfg(feature = "tract-onnx")]
pub use crate::runner::tract_onnx_runner::TractOnnxRunner;
