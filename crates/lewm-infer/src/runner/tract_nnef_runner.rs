//! NNEF runner backed by `tract-nnef`.

use std::path::Path;

use tract_nnef::prelude::*;

use crate::runner::traits::{
    IMAGE_ELEMENT_COUNT, InferenceRunner, RunnerError, RunnerFormat, RunnerMetadata,
    available_intra_op_threads, require_graph, validate_predict_shapes,
};

type Runnable = TypedRunnableModel<TypedModel>;

/// Inference runner for `encoder.nnef` and `predictor.nnef`.
#[derive(Debug)]
pub struct TractNnefRunner {
    encoder: Runnable,
    predictor: Runnable,
    metadata: RunnerMetadata,
}

impl TractNnefRunner {
    /// Load and optimize an NNEF graph pair from a checkpoint directory.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] when graph files are missing or Tract loading,
    /// optimization, or runnable planning fails.
    pub fn new(checkpoint_dir: &Path) -> Result<Self, RunnerError> {
        let encoder_path = require_graph(checkpoint_dir.join("encoder.nnef"))?;
        let predictor_path = require_graph(checkpoint_dir.join("predictor.nnef"))?;
        let intra_op_threads = available_intra_op_threads();
        let encoder = load_nnef_graph(&encoder_path, intra_op_threads)?;
        let predictor = load_nnef_graph(&predictor_path, intra_op_threads)?;
        Ok(Self {
            encoder,
            predictor,
            metadata: RunnerMetadata {
                format: RunnerFormat::Nnef,
                encoder_path,
                predictor_path,
                optimized: true,
                intra_op_threads,
            },
        })
    }
}

impl InferenceRunner for TractNnefRunner {
    fn encode(&mut self, pixels: &[f32; IMAGE_ELEMENT_COUNT]) -> Result<Vec<f32>, RunnerError> {
        let input = Tensor::from_shape(&[1, 3, 224, 224], pixels).map_err(|error| {
            RunnerError::Backend {
                context: "building NNEF encoder input tensor".to_owned(),
                source: error.to_string(),
            }
        })?;
        run_one_output(
            self.metadata.format,
            &mut self.encoder,
            tvec!(input.into_tvalue()),
        )
    }

    fn predict(
        &mut self,
        history: &[f32],
        actions: &[f32],
        h: usize,
        a: usize,
    ) -> Result<Vec<f32>, RunnerError> {
        let latent_dim = validate_predict_shapes(history, actions, h, a)?;
        let history = Tensor::from_shape(&[1, h, latent_dim], history).map_err(|error| {
            RunnerError::Backend {
                context: "building NNEF predictor history tensor".to_owned(),
                source: error.to_string(),
            }
        })?;
        let actions =
            Tensor::from_shape(&[1, h, a], actions).map_err(|error| RunnerError::Backend {
                context: "building NNEF predictor action tensor".to_owned(),
                source: error.to_string(),
            })?;
        run_one_output(
            self.metadata.format,
            &mut self.predictor,
            tvec!(history.into_tvalue(), actions.into_tvalue()),
        )
    }

    fn metadata(&self) -> RunnerMetadata {
        self.metadata.clone()
    }
}

fn load_nnef_graph(path: &Path, intra_op_threads: usize) -> Result<Runnable, RunnerError> {
    let options = PlanOptions {
        executor: Some(multithread::Executor::multithread(intra_op_threads)),
        ..PlanOptions::default()
    };
    let model = tract_nnef::nnef()
        .model_for_path(path)
        .map_err(|error| nnef_backend_error(path, &error))?;
    let model = model
        .into_optimized()
        .map_err(|error| nnef_backend_error(path, &error))?;
    model
        .into_runnable_with_options(&options)
        .map_err(|error| nnef_backend_error(path, &error))
}

fn run_one_output(
    format: RunnerFormat,
    model: &mut Runnable,
    inputs: TVec<TValue>,
) -> Result<Vec<f32>, RunnerError> {
    let outputs = model.run(inputs).map_err(|error| RunnerError::Backend {
        context: format!("running {format} graph"),
        source: error.to_string(),
    })?;
    let first = outputs.first().ok_or_else(|| RunnerError::Backend {
        context: format!("extracting {format} graph output"),
        source: "runner returned no outputs".to_owned(),
    })?;
    let view = first
        .to_array_view::<f32>()
        .map_err(|error| RunnerError::Backend {
            context: format!("extracting {format} F32 output"),
            source: error.to_string(),
        })?;
    Ok(view.iter().copied().collect())
}

fn nnef_backend_error(path: &Path, error: &TractError) -> RunnerError {
    RunnerError::Backend {
        context: format!("loading optimized NNEF graph {}", path.display()),
        source: error.to_string(),
    }
}
