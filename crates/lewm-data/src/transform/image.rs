//! Image preprocessing for raw HWC RGB frames.

use image::imageops::FilterType;

use crate::DataError;

const RGB_CHANNELS: usize = 3;
const DEFAULT_TARGET_SIZE: u32 = 224;
const DEFAULT_MEAN: [f32; RGB_CHANNELS] = [0.5, 0.5, 0.5];
const DEFAULT_STD: [f32; RGB_CHANNELS] = [0.5, 0.5, 0.5];
const U8_SCALE: f32 = 255.0;

/// Interpolation mode used when input frames are not already at target size.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum InterpKind {
    /// Bilinear resize via `image::imageops::FilterType::Triangle`.
    Bilinear,
    /// Bicubic resize via `image::imageops::FilterType::CatmullRom`.
    Bicubic,
}

impl InterpKind {
    fn filter_type(self) -> FilterType {
        match self {
            Self::Bilinear => FilterType::Triangle,
            Self::Bicubic => FilterType::CatmullRom,
        }
    }
}

/// Converts flat HWC `u8` RGB images into normalized CHW `f32` buffers.
#[derive(Debug, Clone, PartialEq)]
pub struct ImagePreprocessor {
    /// Target square image size.
    pub target_size: u32,
    /// Per-channel normalization mean.
    pub mean: [f32; RGB_CHANNELS],
    /// Per-channel normalization standard deviation.
    pub std: [f32; RGB_CHANNELS],
    /// Interpolation mode for non-identity resize.
    pub interp: InterpKind,
}

impl Default for ImagePreprocessor {
    fn default() -> Self {
        Self {
            target_size: DEFAULT_TARGET_SIZE,
            mean: DEFAULT_MEAN,
            std: DEFAULT_STD,
            interp: InterpKind::Bilinear,
        }
    }
}

impl ImagePreprocessor {
    /// Convert a flat HWC `u8` RGB buffer to resized, normalized CHW `f32`.
    ///
    /// The identity-size path does not call into the resize kernel, preserving
    /// the strict `u8 -> f32 / 255` contract for already-224x224 training data.
    ///
    /// # Errors
    ///
    /// Returns an error when the preprocessor config is invalid, source
    /// dimensions overflow `usize`, or `src` is not exactly `src_h * src_w * 3`.
    pub fn apply(&self, src: &[u8], src_h: u32, src_w: u32) -> Result<Vec<f32>, DataError> {
        self.validate()?;
        let expected_len = hwc_len(src_h, src_w)?;
        if src.len() != expected_len {
            return Err(DataError::InvalidTransform(format!(
                "expected HWC RGB buffer length {expected_len}, found {}",
                src.len()
            )));
        }

        let resized;
        let pixels = if src_h == self.target_size && src_w == self.target_size {
            src
        } else {
            let image = image::RgbImage::from_raw(src_w, src_h, src.to_vec()).ok_or_else(|| {
                DataError::InvalidTransform(format!(
                    "could not create RGB image from shape ({src_h}, {src_w}, 3)"
                ))
            })?;
            resized = image::imageops::resize(
                &image,
                self.target_size,
                self.target_size,
                self.interp.filter_type(),
            )
            .into_raw();
            &resized
        };

        self.normalize_hwc_to_chw(pixels)
    }

    fn validate(&self) -> Result<(), DataError> {
        if self.target_size == 0 {
            return Err(DataError::InvalidTransform(
                "target_size must be greater than zero".to_string(),
            ));
        }
        for channel in 0..RGB_CHANNELS {
            if !self.mean[channel].is_finite() {
                return Err(DataError::InvalidTransform(format!(
                    "mean[{channel}] must be finite"
                )));
            }
            if !self.std[channel].is_finite() || self.std[channel] == 0.0 {
                return Err(DataError::InvalidTransform(format!(
                    "std[{channel}] must be finite and non-zero"
                )));
            }
        }
        Ok(())
    }

    fn normalize_hwc_to_chw(&self, src: &[u8]) -> Result<Vec<f32>, DataError> {
        let target = usize::try_from(self.target_size).map_err(|_| {
            DataError::InvalidTransform(format!(
                "target_size {} does not fit usize",
                self.target_size
            ))
        })?;
        let plane = target
            .checked_mul(target)
            .ok_or_else(|| DataError::InvalidTransform("target image area overflow".to_string()))?;
        let output_len = plane
            .checked_mul(RGB_CHANNELS)
            .ok_or_else(|| DataError::InvalidTransform("output image size overflow".to_string()))?;
        if src.len() != output_len {
            return Err(DataError::InvalidTransform(format!(
                "resized image has {} bytes, expected {output_len}",
                src.len()
            )));
        }

        let mut output = vec![0.0; output_len];
        for pixel_index in 0..plane {
            let src_offset = pixel_index * RGB_CHANNELS;
            for channel in 0..RGB_CHANNELS {
                let unit_value = f32::from(src[src_offset + channel]) / U8_SCALE;
                output[(channel * plane) + pixel_index] =
                    (unit_value - self.mean[channel]) / self.std[channel];
            }
        }
        Ok(output)
    }
}

fn hwc_len(src_h: u32, src_w: u32) -> Result<usize, DataError> {
    let height = usize::try_from(src_h).map_err(|_| {
        DataError::InvalidTransform(format!("source height {src_h} does not fit usize"))
    })?;
    let width = usize::try_from(src_w).map_err(|_| {
        DataError::InvalidTransform(format!("source width {src_w} does not fit usize"))
    })?;
    height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(RGB_CHANNELS))
        .ok_or_else(|| DataError::InvalidTransform("source image size overflow".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn assert_close(left: f32, right: f32) {
        assert!(
            (left - right).abs() <= 1e-6,
            "left={left}, right={right}, diff={}",
            (left - right).abs()
        );
    }

    #[test]
    fn image_preprocess_identity_at_224() -> Result<(), Box<dyn std::error::Error>> {
        let preprocessor = ImagePreprocessor::default();
        let mut src = vec![0_u8; 224 * 224 * RGB_CHANNELS];
        src[0] = 0;
        src[1] = 128;
        src[2] = 255;
        src[3] = 64;
        src[4] = 96;
        src[5] = 192;

        let output = preprocessor.apply(&src, 224, 224)?;
        let plane = 224 * 224;

        assert_eq!(output.len(), RGB_CHANNELS * plane);
        assert_close(output[0], -1.0);
        assert_close(output[plane], (f32::from(128_u8) / U8_SCALE - 0.5) / 0.5);
        assert_close(output[2 * plane], 1.0);
        assert_close(output[1], (f32::from(64_u8) / U8_SCALE - 0.5) / 0.5);
        Ok(())
    }

    #[test]
    fn image_preprocess_normalize() -> Result<(), Box<dyn std::error::Error>> {
        let preprocessor = ImagePreprocessor {
            target_size: 1,
            ..ImagePreprocessor::default()
        };

        let output = preprocessor.apply(&[0, 128, 255], 1, 1)?;

        assert_close(output[0], -1.0);
        assert_close(output[1], (f32::from(128_u8) / U8_SCALE - 0.5) / 0.5);
        assert_close(output[2], 1.0);
        Ok(())
    }

    #[test]
    fn image_preprocess_resize_192_to_224() -> Result<(), Box<dyn std::error::Error>> {
        let preprocessor = ImagePreprocessor::default();
        let mut src = Vec::with_capacity(192 * 192 * RGB_CHANNELS);
        for _ in 0..(192 * 192) {
            src.extend_from_slice(&[64, 128, 255]);
        }

        let output = preprocessor.apply(&src, 192, 192)?;
        let plane = 224 * 224;

        assert_eq!(output.len(), RGB_CHANNELS * plane);
        assert_close(output[0], (f32::from(64_u8) / U8_SCALE - 0.5) / 0.5);
        assert_close(output[plane], (f32::from(128_u8) / U8_SCALE - 0.5) / 0.5);
        assert_close(output[2 * plane], 1.0);
        Ok(())
    }

    proptest! {
        #[test]
        fn image_preprocess_deterministic(
            (src_h, src_w, src) in (1_u32..16, 1_u32..16).prop_flat_map(|(src_h, src_w)| {
                let height = usize::try_from(src_h).unwrap_or(0);
                let width = usize::try_from(src_w).unwrap_or(0);
                let len = height.saturating_mul(width).saturating_mul(RGB_CHANNELS);
                (Just(src_h), Just(src_w), prop::collection::vec(any::<u8>(), len))
            })
        ) {
            let preprocessor = ImagePreprocessor {
                target_size: 8,
                ..ImagePreprocessor::default()
            };

            let first = preprocessor.apply(&src, src_h, src_w)
                .map_err(|err| TestCaseError::fail(err.to_string()))?;
            let second = preprocessor.apply(&src, src_h, src_w)
                .map_err(|err| TestCaseError::fail(err.to_string()))?;

            prop_assert_eq!(first, second);
        }
    }
}
