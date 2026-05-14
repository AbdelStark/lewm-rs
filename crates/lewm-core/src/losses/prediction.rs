//! Prediction MSE loss from RFC 0003.

use burn::tensor::{Tensor, backend::Backend};

/// Compute `L_pred = mean((pred - target)^2)` over batch, time, and feature axes.
///
/// `pred` and `target` are `(B, T_p, D)` tensors. The target arm is
/// intentionally not detached, so autodiff gradients flow through both the
/// predicted and target branches.
pub fn prediction_loss<B: Backend>(pred: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 1> {
    debug_assert_eq!(
        pred.dims(),
        target.dims(),
        "prediction loss input shapes must match"
    );

    let diff = pred - target;
    diff.clone().mul(diff).mean()
}

#[cfg(test)]
mod tests {
    use burn::tensor::backend::{AutodiffBackend, Backend};
    use burn::tensor::{Tensor, TensorData};

    use super::*;

    type CpuBackend = burn_ndarray::NdArray<f32>;
    type CpuAutodiffBackend = burn_autodiff::Autodiff<CpuBackend>;

    #[test]
    fn pred_loss_shapes_and_value() {
        let device = burn_ndarray::NdArrayDevice::default();
        let pred = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]),
            &device,
        );
        let target = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(vec![1.0, 1.0, 5.0, 4.0], [1, 2, 2]),
            &device,
        );

        let loss = prediction_loss(pred, target);

        assert_eq!(loss.dims(), [1]);
        assert_close(scalar(&loss), 1.25);
    }

    #[test]
    fn pred_loss_allows_zero_loss() {
        let device = burn_ndarray::NdArrayDevice::default();
        let pred = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(vec![0.5, -0.25, 2.0, 4.0], [1, 2, 2]),
            &device,
        );
        let target = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(vec![0.5, -0.25, 2.0, 4.0], [1, 2, 2]),
            &device,
        );

        let loss = prediction_loss(pred, target);

        assert_close(scalar(&loss), 0.0);
    }

    #[test]
    #[should_panic(expected = "prediction loss input shapes must match")]
    fn pred_loss_shape_mismatch_debug_asserts() {
        let device = burn_ndarray::NdArrayDevice::default();
        let pred = Tensor::<CpuBackend, 3>::zeros([1, 2, 2], &device);
        let target = Tensor::<CpuBackend, 3>::zeros([1, 1, 2], &device);

        let _ = prediction_loss(pred, target);
    }

    #[test]
    fn pred_loss_gradient_flows_through_both_arms() {
        let device = burn_ndarray::NdArrayDevice::default();
        let pred = Tensor::<CpuAutodiffBackend, 3>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]),
            &device,
        )
        .require_grad();
        let target = Tensor::<CpuAutodiffBackend, 3>::from_data(
            TensorData::new(vec![1.0, 1.0, 5.0, 4.0], [1, 2, 2]),
            &device,
        )
        .require_grad();

        let loss = prediction_loss(pred.clone(), target.clone());
        let grads = loss.backward();
        let pred_grad = pred
            .grad(&grads)
            .expect("prediction branch gradient should exist");
        let target_grad = target
            .grad(&grads)
            .expect("target branch gradient should exist");
        let pred_values = tensor_values_inner::<CpuAutodiffBackend, 3>(&pred_grad);
        let target_values = tensor_values_inner::<CpuAutodiffBackend, 3>(&target_grad);

        assert_close(pred_values[0], 0.0);
        assert_close(pred_values[1], 0.5);
        assert_close(pred_values[2], -1.0);
        assert_close(pred_values[3], 0.0);
        for (pred_grad, target_grad) in pred_values.iter().zip(target_values) {
            assert_close(*pred_grad + target_grad, 0.0);
        }
    }

    fn scalar<B: Backend>(tensor: &Tensor<B, 1>) -> f32 {
        tensor_values(tensor)[0]
    }

    fn tensor_values<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Vec<f32> {
        tensor
            .to_data()
            .to_vec::<f32>()
            .expect("test tensor should contain f32 values")
    }

    fn tensor_values_inner<B: AutodiffBackend, const D: usize>(
        tensor: &Tensor<B::InnerBackend, D>,
    ) -> Vec<f32> {
        tensor
            .to_data()
            .to_vec::<f32>()
            .expect("test tensor should contain f32 values")
    }

    #[track_caller]
    fn assert_close(actual: f32, expected: f32) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= 1.0e-6,
            "expected {expected}, got {actual}, diff {diff}"
        );
    }
}
