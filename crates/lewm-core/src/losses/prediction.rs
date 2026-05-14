//! Prediction MSE loss from RFC 0003.

use crate::LewmCoreError;

/// Compute `L_pred = mean((pred - target)^2)` over batch, time, and feature axes.
///
/// `pred` and `target` are row-major `(B, T_p, D)` buffers. The target arm is
/// intentionally not detached by this kernel; the eventual Burn tensor wrapper
/// should call the same math with regular autodiff tensors so gradient flows
/// through both the predicted and target branches.
///
/// # Errors
///
/// Returns [`LewmCoreError::InvalidTensorOp`] when any dimension is zero, the
/// element count overflows, or either buffer contains non-finite values. Returns
/// [`LewmCoreError::InvalidShape`] when either buffer length differs from
/// `batch * steps * dim`.
pub fn prediction_loss(
    pred: &[f32],
    target: &[f32],
    batch: usize,
    steps: usize,
    dim: usize,
) -> Result<f32, LewmCoreError> {
    let expected_len = validate_prediction_shape(pred, target, batch, steps, dim)?;

    let squared_error = pred
        .iter()
        .zip(target)
        .map(|(pred_value, target_value)| {
            let diff = f64::from(*pred_value) - f64::from(*target_value);
            diff * diff
        })
        .sum::<f64>();

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    Ok((squared_error / expected_len as f64) as f32)
}

fn validate_prediction_shape(
    pred: &[f32],
    target: &[f32],
    batch: usize,
    steps: usize,
    dim: usize,
) -> Result<usize, LewmCoreError> {
    if batch == 0 || steps == 0 || dim == 0 {
        return Err(LewmCoreError::InvalidTensorOp {
            reason: "prediction loss dimensions must be non-zero".to_owned(),
        });
    }

    let expected_len = batch
        .checked_mul(steps)
        .and_then(|count| count.checked_mul(dim))
        .ok_or_else(|| LewmCoreError::InvalidTensorOp {
            reason: "prediction loss element count overflowed usize".to_owned(),
        })?;

    if pred.len() != expected_len {
        return Err(LewmCoreError::InvalidShape {
            expected: vec![batch, steps, dim],
            found: vec![pred.len()],
        });
    }

    if target.len() != expected_len {
        return Err(LewmCoreError::InvalidShape {
            expected: vec![batch, steps, dim],
            found: vec![target.len()],
        });
    }

    if pred.iter().chain(target).any(|value| !value.is_finite()) {
        return Err(LewmCoreError::InvalidTensorOp {
            reason: "prediction loss inputs must be finite".to_owned(),
        });
    }

    Ok(expected_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pred_loss_shapes_and_value() -> Result<(), LewmCoreError> {
        let pred = [1.0, 2.0, 3.0, 4.0];
        let target = [1.0, 1.0, 5.0, 4.0];

        let loss = prediction_loss(&pred, &target, 1, 2, 2)?;

        assert_close(loss, 1.25);
        Ok(())
    }

    #[test]
    fn pred_loss_allows_zero_loss() -> Result<(), LewmCoreError> {
        let pred = [0.5, -0.25, 2.0, 4.0];
        let target = [0.5, -0.25, 2.0, 4.0];

        let loss = prediction_loss(&pred, &target, 1, 2, 2)?;

        assert_close(loss, 0.0);
        Ok(())
    }

    #[test]
    fn pred_loss_rejects_shape_mismatch() {
        let err = prediction_loss(&[1.0, 2.0], &[1.0], 1, 1, 2)
            .expect_err("target length should be rejected");

        assert!(matches!(err, LewmCoreError::InvalidShape { .. }));
    }

    #[test]
    fn pred_loss_rejects_non_finite_values() {
        let err = prediction_loss(&[1.0, f32::NAN], &[1.0, 2.0], 1, 1, 2)
            .expect_err("non-finite input should be rejected");

        assert!(err.to_string().contains("finite"));
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
