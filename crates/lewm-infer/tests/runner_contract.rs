//! Integration tests for the public runner loading and execution contract.
#![allow(unexpected_cfgs)]

#[cfg(feature = "tract-onnx")]
mod onnx_contract {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use lewm_infer::plan::{CpuCem, cem_rng};
    use lewm_infer::runner::{
        IMAGE_ELEMENT_COUNT, InferenceRunner, RunnerFormat, detect_checkpoint_format, load,
    };
    use prost::Message;
    use tract_onnx::pb::tensor_proto::DataType;
    use tract_onnx::pb::tensor_shape_proto::dimension::Value as DimensionValue;
    use tract_onnx::pb::type_proto::{Tensor, Value as TypeValue};
    use tract_onnx::pb::{
        GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorShapeProto, TypeProto,
        ValueInfoProto, Version,
    };

    const H: usize = 2;
    const A: usize = 3;
    const H_I64: i64 = 2;
    const A_I64: i64 = 3;
    const LATENT_DIM_I64: i64 = 4;

    #[test]
    // Tarpaulin's ptrace backend can report false enum assertion failures after
    // Tract graph loading; the normal CI test matrix still runs this contract.
    #[cfg_attr(tarpaulin, ignore)]
    fn tract_onnx_runner_load_and_encode() -> Result<(), Box<dyn std::error::Error>> {
        let root = onnx_fixture_dir("lewm-onnx-encode")?;
        let mut runner = load(&root)?;
        assert_eq!(detect_checkpoint_format(&root), Some(RunnerFormat::Onnx));
        assert_eq!(runner.metadata().format, RunnerFormat::Onnx);
        assert!(runner.metadata().optimized);
        assert!(runner.metadata().intra_op_threads >= 1);

        let mut pixels = vec![0.0_f32; IMAGE_ELEMENT_COUNT].into_boxed_slice();
        pixels[42] = 7.5;
        let pixels: Box<[f32; IMAGE_ELEMENT_COUNT]> = pixels
            .try_into()
            .map_err(|_| "pixel fixture has wrong element count")?;
        let output = runner.encode(pixels.as_ref())?;
        assert_eq!(output.len(), IMAGE_ELEMENT_COUNT);
        assert!((output[42] - 7.5).abs() <= f32::EPSILON);

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn tract_onnx_runner_predict() -> Result<(), Box<dyn std::error::Error>> {
        let root = onnx_fixture_dir("lewm-onnx-predict")?;
        let mut runner = load(&root)?;
        let history = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let actions = [0.0_f32; H * A];

        let output = runner.predict(&history, &actions, H, A)?;
        assert_eq!(output.len(), history.len());
        assert!(
            output
                .iter()
                .zip(history.iter())
                .all(|(actual, expected)| (*actual - *expected).abs() <= f32::EPSILON)
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn tract_onnx_runner_cpu_cem_plan() -> Result<(), Box<dyn std::error::Error>> {
        let root = onnx_fixture_dir("lewm-onnx-cpu-cem")?;
        let mut runner = load(&root)?;
        let planner = CpuCem {
            n_iter: 1,
            n_cand: 2,
            n_elite: 1,
            horizon_plan: 1,
            sigma_init: 1.0,
            sigma_min: 0.05,
        };
        let history = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let goal = [5.0_f32, 6.0, 7.0, 8.0];
        let mut rng = cem_rng(0)?;

        let result = planner.plan(&mut *runner, &history, &goal, &mut rng, A)?;

        assert_eq!(result.best_actions.len(), A);
        assert!((result.best_cost - 0.0).abs() <= f32::EPSILON);
        assert_eq!(result.trace.len(), 1);

        fs::remove_dir_all(root)?;
        Ok(())
    }

    fn onnx_fixture_dir(prefix: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let root = unique_temp_dir(prefix)?;
        write_model(
            &root.join("encoder.onnx"),
            &identity_model(
                "encoder",
                &[("pixels", &[1_i64, 3, 224, 224][..])],
                "pixels",
                "encoded",
                &[1_i64, 3, 224, 224],
            ),
        )?;
        write_model(
            &root.join("predictor.onnx"),
            &identity_model(
                "predictor",
                &[
                    ("history", &[1_i64, H_I64, LATENT_DIM_I64][..]),
                    ("actions", &[1_i64, H_I64, A_I64][..]),
                ],
                "history",
                "predicted",
                &[1_i64, H_I64, LATENT_DIM_I64],
            ),
        )?;
        Ok(root)
    }

    fn write_model(path: &Path, model: &ModelProto) -> std::io::Result<()> {
        fs::write(path, model.encode_to_vec())
    }

    fn identity_model(
        graph_name: &str,
        inputs: &[(&str, &[i64])],
        identity_input: &str,
        output: &str,
        output_shape: &[i64],
    ) -> ModelProto {
        ModelProto {
            ir_version: i64::from(Version::IrVersion as i32),
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 18,
            }],
            producer_name: "lewm-infer-tests".to_owned(),
            producer_version: env!("CARGO_PKG_VERSION").to_owned(),
            domain: "lewm.rs.test".to_owned(),
            model_version: 1,
            doc_string: String::new(),
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    input: vec![identity_input.to_owned()],
                    output: vec![output.to_owned()],
                    name: format!("{graph_name}_identity"),
                    op_type: "Identity".to_owned(),
                    domain: String::new(),
                    attribute: Vec::new(),
                    doc_string: String::new(),
                }],
                name: graph_name.to_owned(),
                initializer: Vec::new(),
                sparse_initializer: Vec::new(),
                doc_string: String::new(),
                input: inputs
                    .iter()
                    .map(|(name, shape)| value_info(name, shape))
                    .collect(),
                output: vec![value_info(output, output_shape)],
                value_info: Vec::new(),
                quantization_annotation: Vec::new(),
            }),
            metadata_props: Vec::new(),
            training_info: Vec::new(),
            functions: Vec::new(),
        }
    }

    fn value_info(name: &str, shape: &[i64]) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_owned(),
            r#type: Some(TypeProto {
                denotation: String::new(),
                value: Some(TypeValue::TensorType(Tensor {
                    elem_type: DataType::Float as i32,
                    shape: Some(TensorShapeProto {
                        dim: shape
                            .iter()
                            .map(|dimension| tract_onnx::pb::tensor_shape_proto::Dimension {
                                denotation: String::new(),
                                value: Some(DimensionValue::DimValue(*dimension)),
                            })
                            .collect(),
                    }),
                })),
            }),
            doc_string: String::new(),
        }
    }

    fn unique_temp_dir(prefix: &str) -> std::io::Result<PathBuf> {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        path.push(format!("{prefix}-{nanos}"));
        fs::create_dir_all(&path)?;
        Ok(path)
    }

    fn assert_runner_object_safe(_: &mut dyn InferenceRunner) {}

    #[test]
    fn inference_runner_trait_is_object_safe() -> Result<(), Box<dyn std::error::Error>> {
        let root = onnx_fixture_dir("lewm-onnx-object-safe")?;
        let mut runner = load(&root)?;
        assert_runner_object_safe(runner.as_mut());
        fs::remove_dir_all(root)?;
        Ok(())
    }
}

#[cfg(feature = "tract-nnef")]
mod nnef_contract {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use lewm_infer::plan::{CpuCem, cem_rng};
    use lewm_infer::runner::{IMAGE_ELEMENT_COUNT, RunnerFormat, detect_checkpoint_format, load};

    const H: usize = 2;
    const A: usize = 3;

    const ENCODER_NNEF: &str = r"
version 1.0;

graph encoder( input ) -> ( output )
{
    input = external(shape = [1, 3, 224, 224]);
    output = relu(input);
}
";

    const PREDICTOR_NNEF: &str = r"
version 1.0;

graph predictor( history, actions ) -> ( output )
{
    history = external(shape = [1, 2, 4]);
    actions = external(shape = [1, 2, 3]);
    output = relu(history);
}
";

    #[test]
    // Tarpaulin's ptrace backend can report false enum assertion failures after
    // Tract graph loading; the normal CI test matrix still runs this contract.
    #[cfg_attr(tarpaulin, ignore)]
    fn tract_nnef_runner_load_and_encode() -> Result<(), Box<dyn std::error::Error>> {
        let root = nnef_fixture_dir("lewm-nnef-encode")?;
        let mut runner = load(&root)?;
        assert_eq!(detect_checkpoint_format(&root), Some(RunnerFormat::Nnef));
        assert_eq!(runner.metadata().format, RunnerFormat::Nnef);
        assert!(runner.metadata().optimized);
        assert!(runner.metadata().intra_op_threads >= 1);

        let mut pixels = vec![0.0_f32; IMAGE_ELEMENT_COUNT].into_boxed_slice();
        pixels[42] = 7.5;
        let pixels: Box<[f32; IMAGE_ELEMENT_COUNT]> = pixels
            .try_into()
            .map_err(|_| "pixel fixture has wrong element count")?;
        let output = runner.encode(pixels.as_ref())?;
        assert_eq!(output.len(), IMAGE_ELEMENT_COUNT);
        assert!((output[42] - 7.5).abs() <= f32::EPSILON);

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn tract_nnef_runner_predict() -> Result<(), Box<dyn std::error::Error>> {
        let root = nnef_fixture_dir("lewm-nnef-predict")?;
        let mut runner = load(&root)?;
        let history = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let actions = [0.0_f32; H * A];

        let output = runner.predict(&history, &actions, H, A)?;
        assert_eq!(output.len(), history.len());
        assert!(
            output
                .iter()
                .zip(history.iter())
                .all(|(actual, expected)| (*actual - *expected).abs() <= f32::EPSILON)
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn tract_nnef_runner_cpu_cem_plan() -> Result<(), Box<dyn std::error::Error>> {
        let root = nnef_fixture_dir("lewm-nnef-cpu-cem")?;
        let mut runner = load(&root)?;
        let planner = CpuCem {
            n_iter: 1,
            n_cand: 2,
            n_elite: 1,
            horizon_plan: 1,
            sigma_init: 1.0,
            sigma_min: 0.05,
        };
        let history = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let goal = [5.0_f32, 6.0, 7.0, 8.0];
        let mut rng = cem_rng(0)?;

        let result = planner.plan(&mut *runner, &history, &goal, &mut rng, A)?;

        assert_eq!(result.best_actions.len(), A);
        assert!((result.best_cost - 0.0).abs() <= f32::EPSILON);
        assert_eq!(result.trace.len(), 1);

        fs::remove_dir_all(root)?;
        Ok(())
    }

    fn nnef_fixture_dir(prefix: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let root = unique_temp_dir(prefix)?;
        write_nnef_archive(&root.join("encoder.nnef"), ENCODER_NNEF)?;
        write_nnef_archive(&root.join("predictor.nnef"), PREDICTOR_NNEF)?;
        Ok(root)
    }

    fn write_nnef_archive(path: &std::path::Path, graph: &str) -> std::io::Result<()> {
        let file = fs::File::create(path)?;
        let mut archive = tar::Builder::new(file);
        let mut header = tar::Header::new_gnu();
        header.set_size(graph.len().try_into().map_err(std::io::Error::other)?);
        header.set_mode(0o644);
        header.set_cksum();
        archive.append_data(&mut header, "graph.nnef", graph.as_bytes())?;
        archive.finish()
    }

    fn unique_temp_dir(prefix: &str) -> std::io::Result<PathBuf> {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        path.push(format!("{prefix}-{nanos}"));
        fs::create_dir_all(&path)?;
        Ok(path)
    }
}
