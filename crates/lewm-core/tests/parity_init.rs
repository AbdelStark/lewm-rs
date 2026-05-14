//! RFC 0008 initialization and parameter-shape parity checks.

#![cfg(feature = "parity-fixtures")]

#[allow(dead_code)]
mod support;

use std::collections::BTreeMap;

use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::{Bool, Int, Tensor};
use burn_ndarray::{NdArray, NdArrayDevice};
use lewm_core::{Jepa, JepaConfig, LewmCoreError};

type CpuBackend = NdArray<f32>;

#[test]
fn parity_init_parameter_shape_audit() -> Result<(), Box<dyn std::error::Error>> {
    let device = NdArrayDevice::default();
    let model = Jepa::<CpuBackend>::init(JepaConfig::default(), &device)?;
    let meta = support::load_reference_model_meta()?;
    let shapes = collect_parameter_shapes(&model);

    assert_eq!(model.num_params(), count_float_params(&shapes));
    assert_eq!(
        meta["source_model"]["state_dict_value_count"], 18_042_672,
        "reference metadata value count changed; refresh the parity shape audit"
    );
    assert_required_shape(
        &shapes,
        "encoder.embeddings.patch_embed.proj.weight",
        &[192, 3, 14, 14],
    )?;
    assert_required_shape(&shapes, "encoder.embeddings.patch_embed.proj.bias", &[192])?;
    assert_required_shape(&shapes, "encoder.embeddings.cls_token", &[1, 1, 192])?;
    assert_required_shape(&shapes, "encoder.embeddings.pos_embed", &[1, 257, 192])?;
    assert_required_shape(&shapes, "encoder.blocks.0.attn.qkv.weight", &[192, 576])?;
    assert_required_shape(&shapes, "encoder.blocks.11.mlp.fc2.weight", &[768, 192])?;
    assert_required_shape(&shapes, "action_encoder.smoother.weight", &[10, 10, 1])?;
    assert_required_shape(&shapes, "action_encoder.fc2.weight", &[768, 192])?;
    assert_required_shape(&shapes, "predictor.pos_embed", &[1, 3, 192])?;
    assert_required_shape(&shapes, "predictor.blocks.0.attn.qkv.weight", &[192, 3072])?;
    assert_required_shape(
        &shapes,
        "predictor.blocks.5.adaln.linear.weight",
        &[192, 1152],
    )?;
    assert_required_shape(&shapes, "projector.fc1.weight", &[192, 2048])?;
    assert_required_shape(&shapes, "projector.fc2.weight", &[2048, 192])?;
    assert_required_shape(&shapes, "pred_proj.fc1.weight", &[192, 2048])?;
    assert_required_shape(&shapes, "pred_proj.fc2.weight", &[2048, 192])?;

    Ok(())
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ParameterShape {
    dtype: &'static str,
    shape: Vec<usize>,
}

#[derive(Default)]
struct ShapeVisitor {
    stack: Vec<String>,
    shapes: BTreeMap<String, ParameterShape>,
}

impl ModuleVisitor<CpuBackend> for ShapeVisitor {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.stack.push(name.to_owned());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.stack.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<CpuBackend, D>>) {
        self.push("float", param.dims());
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<CpuBackend, D, Int>>) {
        self.push("int", param.dims());
    }

    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<CpuBackend, D, Bool>>) {
        self.push("bool", param.dims());
    }
}

impl ShapeVisitor {
    fn push<const D: usize>(&mut self, dtype: &'static str, shape: [usize; D]) {
        self.shapes.insert(
            self.stack.join("."),
            ParameterShape {
                dtype,
                shape: shape.to_vec(),
            },
        );
    }
}

fn collect_parameter_shapes(model: &Jepa<CpuBackend>) -> BTreeMap<String, ParameterShape> {
    let mut visitor = ShapeVisitor::default();
    model.visit(&mut visitor);
    visitor.shapes
}

fn count_float_params(shapes: &BTreeMap<String, ParameterShape>) -> usize {
    shapes
        .values()
        .filter(|shape| shape.dtype == "float")
        .map(|shape| shape.shape.iter().product::<usize>())
        .sum()
}

fn assert_required_shape(
    shapes: &BTreeMap<String, ParameterShape>,
    name: &str,
    expected: &[usize],
) -> Result<(), LewmCoreError> {
    let found = shapes
        .get(name)
        .ok_or_else(|| LewmCoreError::ParamNotFound {
            name: name.to_owned(),
        })?;
    assert_eq!(found.dtype, "float", "{name} dtype");
    assert_eq!(found.shape, expected, "{name} shape");
    Ok(())
}
