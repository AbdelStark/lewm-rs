//! Image preprocessing for inference-time JPEG and PNG inputs.

use std::fmt;
use std::path::{Path, PathBuf};

use image::imageops::FilterType;
use image::{ImageReader, RgbImage};

use crate::runner::IMAGE_ELEMENT_COUNT;

/// RGB channel count used by the exported encoder.
pub const RGB_CHANNELS: usize = 3;
/// Inference image side length in pixels.
pub const TARGET_IMAGE_SIZE: u32 = 224;
/// Per-channel `ImageNet` / HF `ViT` mean used by the training data path.
pub const IMAGENET_MEAN: [f32; RGB_CHANNELS] = [0.5, 0.5, 0.5];
/// Per-channel `ImageNet` / HF `ViT` standard deviation used by the training data path.
pub const IMAGENET_STD: [f32; RGB_CHANNELS] = [0.5, 0.5, 0.5];

const U8_SCALE: f32 = 255.0;

/// Decode, resize, and normalize an inference image from disk.
///
/// The output is CHW `f32` with shape `(3, 224, 224)`, matching
/// `lewm-data`'s default preprocessing contract.
///
/// # Errors
///
/// Returns [`PreprocessError`] when the file cannot be read, decoded, resized,
/// or converted into the fixed encoder input shape.
pub fn preprocess_path(
    path: impl AsRef<Path>,
) -> Result<Box<[f32; IMAGE_ELEMENT_COUNT]>, PreprocessError> {
    let path = path.as_ref();
    let reader = ImageReader::open(path).map_err(|source| PreprocessError::Read {
        path: path.to_path_buf(),
        source: source.to_string(),
    })?;
    let reader = reader
        .with_guessed_format()
        .map_err(|source| PreprocessError::Read {
            path: path.to_path_buf(),
            source: source.to_string(),
        })?;
    let image = reader.decode().map_err(|source| PreprocessError::Decode {
        path: path.to_path_buf(),
        source: source.to_string(),
    })?;
    preprocess_rgb8(&image.to_rgb8())
}

/// Resize and normalize an RGB image that has already been decoded.
///
/// # Errors
///
/// Returns [`PreprocessError`] when the image cannot be converted into the
/// fixed encoder input shape.
pub fn preprocess_rgb8(
    image: &RgbImage,
) -> Result<Box<[f32; IMAGE_ELEMENT_COUNT]>, PreprocessError> {
    if image.width() == 0 || image.height() == 0 {
        return Err(PreprocessError::InvalidImage {
            reason: "image width and height must be non-zero".to_owned(),
        });
    }

    let resized;
    let pixels = if image.width() == TARGET_IMAGE_SIZE && image.height() == TARGET_IMAGE_SIZE {
        image.as_raw()
    } else {
        resized = image::imageops::resize(
            image,
            TARGET_IMAGE_SIZE,
            TARGET_IMAGE_SIZE,
            FilterType::Triangle,
        )
        .into_raw();
        &resized
    };

    normalize_hwc_to_chw(pixels)
}

fn normalize_hwc_to_chw(pixels: &[u8]) -> Result<Box<[f32; IMAGE_ELEMENT_COUNT]>, PreprocessError> {
    if pixels.len() != IMAGE_ELEMENT_COUNT {
        return Err(PreprocessError::InvalidImage {
            reason: format!(
                "resized RGB image has {} bytes, expected {IMAGE_ELEMENT_COUNT}",
                pixels.len()
            ),
        });
    }

    let plane = IMAGE_ELEMENT_COUNT / RGB_CHANNELS;
    let mut output = vec![0.0_f32; IMAGE_ELEMENT_COUNT].into_boxed_slice();
    for pixel_index in 0..plane {
        let src_offset = pixel_index * RGB_CHANNELS;
        for channel in 0..RGB_CHANNELS {
            let unit = f32::from(pixels[src_offset + channel]) / U8_SCALE;
            output[(channel * plane) + pixel_index] =
                (unit - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
        }
    }

    output
        .try_into()
        .map_err(|_| PreprocessError::InvalidImage {
            reason: "normalized image did not match encoder input shape".to_owned(),
        })
}

/// Image preprocessing error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreprocessError {
    /// The image file could not be read or its format could not be inferred.
    Read {
        /// File path that failed.
        path: PathBuf,
        /// Source error string.
        source: String,
    },
    /// The image file could not be decoded.
    Decode {
        /// File path that failed.
        path: PathBuf,
        /// Source error string.
        source: String,
    },
    /// Decoded image dimensions or channel data were invalid.
    InvalidImage {
        /// Failure reason.
        reason: String,
    },
}

impl fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, source } => {
                write!(f, "failed to read image {}: {source}", path.display())
            },
            Self::Decode { path, source } => {
                write!(f, "failed to decode image {}: {source}", path.display())
            },
            Self::InvalidImage { reason } => write!(f, "invalid image input: {reason}"),
        }
    }
}

impl std::error::Error for PreprocessError {}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use image::{Rgb, RgbImage};

    use super::*;

    fn assert_close(left: f32, right: f32) {
        assert!(
            (left - right).abs() <= 1e-6,
            "left={left}, right={right}, diff={}",
            (left - right).abs()
        );
    }

    #[test]
    fn preprocess_rgb8_normalizes_chw() -> Result<(), Box<dyn std::error::Error>> {
        let image = RgbImage::from_pixel(1, 1, Rgb([0, 128, 255]));

        let output = preprocess_rgb8(&image)?;

        assert_eq!(output.len(), IMAGE_ELEMENT_COUNT);
        assert_close(output[0], -1.0);
        assert_close(
            output[224 * 224],
            (f32::from(128_u8) / U8_SCALE - 0.5) / 0.5,
        );
        assert_close(output[2 * 224 * 224], 1.0);
        Ok(())
    }

    #[test]
    fn preprocess_path_reads_png_and_jpeg() -> Result<(), Box<dyn std::error::Error>> {
        let root = unique_temp_dir("lewm-infer-preprocess")?;
        let png_path = root.join("input.png");
        let jpeg_path = root.join("input.jpg");
        let image = RgbImage::from_pixel(3, 2, Rgb([255, 255, 255]));
        image.save(&png_path)?;
        image.save(&jpeg_path)?;

        let png = preprocess_path(&png_path)?;
        let jpeg = preprocess_path(&jpeg_path)?;

        assert_eq!(png.len(), IMAGE_ELEMENT_COUNT);
        assert_eq!(jpeg.len(), IMAGE_ELEMENT_COUNT);
        assert_close(png[0], 1.0);
        assert_close(jpeg[0], 1.0);
        fs::remove_dir_all(root)?;
        Ok(())
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
