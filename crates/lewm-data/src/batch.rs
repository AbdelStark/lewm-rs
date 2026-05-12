//! Batch collation for raw dataset samples.

use std::{fmt, marker::PhantomData};

use crate::{ActionNormalizer, DataError, ImagePreprocessor, Sample, SampleMeta};

const RGB_CHANNELS: usize = 3;

/// Pixel dtype requested for a collated batch.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum BatchDtype {
    /// Keep pixels as F32.
    F32,
    /// Cast pixels to BF16 after F32 preprocessing.
    Bf16,
}

/// Minimal backend boundary used by the data plane before model tensors exist.
pub trait BatchBackend: fmt::Debug + Send + Sync + 'static {
    /// Device handle type for the backend.
    type Device: Clone + fmt::Debug + Eq + PartialEq + Send + Sync + 'static;

    /// Human-readable backend name for diagnostics.
    fn name(device: &Self::Device) -> &'static str;

    /// Return whether this backend/device accepts a batch pixel dtype.
    fn supports_dtype(device: &Self::Device, dtype: BatchDtype) -> bool;
}

/// Host-side backend for tests and CPU data-pipeline smoke runs.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct HostBackend;

/// Host-side device.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum HostDevice {
    /// CPU host memory.
    #[default]
    Cpu,
}

impl BatchBackend for HostBackend {
    fn name(_device: &Self::Device) -> &'static str {
        "host"
    }

    fn supports_dtype(_device: &Self::Device, _dtype: BatchDtype) -> bool {
        true
    }

    type Device = HostDevice;
}

/// A shaped, row-major batch tensor payload.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchTensor<B: BatchBackend, const D: usize> {
    shape: [usize; D],
    dtype: BatchDtype,
    device: B::Device,
    values: Vec<f32>,
    backend: PhantomData<B>,
}

impl<B: BatchBackend, const D: usize> BatchTensor<B, D> {
    fn new(
        shape: [usize; D],
        dtype: BatchDtype,
        device: B::Device,
        values: Vec<f32>,
    ) -> Result<Self, DataError> {
        let expected_len = checked_shape_len(shape, "batch tensor size")?;
        if values.len() != expected_len {
            return Err(DataError::InvalidTransform(format!(
                "batch tensor shape {shape:?} expects {expected_len} values, found {}",
                values.len()
            )));
        }

        Ok(Self {
            shape,
            dtype,
            device,
            values,
            backend: PhantomData,
        })
    }

    /// Return the tensor shape.
    #[must_use]
    pub fn shape(&self) -> [usize; D] {
        self.shape
    }

    /// Return the tensor dtype.
    #[must_use]
    pub fn dtype(&self) -> BatchDtype {
        self.dtype
    }

    /// Return the backend device handle.
    #[must_use]
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Return the row-major values.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Consume the tensor into its row-major values.
    #[must_use]
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }

    /// Return the number of scalar values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Return whether the tensor has no scalar values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Tensor batch consumed by training and evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch<B: BatchBackend> {
    /// Pixel tensor with shape `(batch, time, 3, height, width)`.
    pub pixels: BatchTensor<B, 5>,
    /// Normalized action tensor with shape `(batch, time, action_dim)`.
    pub actions: BatchTensor<B, 3>,
    /// Per-sample metadata for traceability.
    pub meta: Vec<SampleMeta>,
}

/// Collate raw samples into device-tagged tensors using bulk host buffers.
///
/// Pixels are resized/normalized frame-by-frame and packed directly into one
/// `(B, T, 3, H, W)` host buffer before constructing the batch payload. Actions
/// are normalized into one `(B, T, A)` host buffer and always remain F32.
///
/// # Errors
///
/// Returns an error when the sample list is empty, sample shapes disagree,
/// frame/action buffers do not match their declared shapes, transform inputs
/// are invalid, or BF16 is requested on a backend device that does not support it.
pub fn collate<B: BatchBackend>(
    samples: &[Sample],
    image_preproc: &ImagePreprocessor,
    action_norm: &ActionNormalizer,
    device: &B::Device,
    dtype: BatchDtype,
) -> Result<Batch<B>, DataError> {
    if samples.is_empty() {
        return Err(DataError::EmptyDataset(
            "cannot collate an empty sample list".to_string(),
        ));
    }
    if !B::supports_dtype(device, dtype) {
        return Err(DataError::InvalidConfig(format!(
            "backend {} on {:?} does not support requested pixel dtype {:?}",
            B::name(device),
            device,
            dtype
        )));
    }

    let shape = validate_samples(samples, action_norm.action_dim())?;
    let target_size = usize::try_from(image_preproc.target_size).map_err(|_| {
        DataError::InvalidTransform(format!(
            "target_size {} does not fit usize",
            image_preproc.target_size
        ))
    })?;
    let target_frame_len = checked_hwc_len(target_size, target_size, RGB_CHANNELS)?;
    let source_frame_len = checked_hwc_len(shape.frame_h, shape.frame_w, RGB_CHANNELS)?;

    let pixel_capacity = checked_mul(
        samples.len(),
        checked_mul(shape.time, target_frame_len, "batch pixel size")?,
        "batch pixel size",
    )?;
    let action_capacity = checked_mul(
        samples.len(),
        checked_mul(shape.time, shape.action_dim, "batch action size")?,
        "batch action size",
    )?;
    let mut pixels = Vec::with_capacity(pixel_capacity);
    let mut actions = Vec::with_capacity(action_capacity);
    let mut meta = Vec::with_capacity(samples.len());

    for sample in samples {
        append_sample_pixels(sample, image_preproc, source_frame_len, &mut pixels)?;
        actions.extend(action_norm.apply(&sample.actions)?);
        meta.push(sample.meta);
    }

    debug_assert_eq!(pixels.len(), pixel_capacity);
    debug_assert_eq!(actions.len(), action_capacity);

    if dtype == BatchDtype::Bf16 {
        cast_f32_buffer_to_bf16(&mut pixels);
    }

    let pixels = BatchTensor::new(
        [
            samples.len(),
            shape.time,
            RGB_CHANNELS,
            target_size,
            target_size,
        ],
        dtype,
        device.clone(),
        pixels,
    )?;
    let actions = BatchTensor::new(
        [samples.len(), shape.time, shape.action_dim],
        BatchDtype::F32,
        device.clone(),
        actions,
    )?;

    Ok(Batch {
        pixels,
        actions,
        meta,
    })
}

#[derive(Debug, Clone, Copy)]
struct BatchShape {
    time: usize,
    frame_h: usize,
    frame_w: usize,
    action_dim: usize,
}

fn validate_samples(samples: &[Sample], normalizer_dim: usize) -> Result<BatchShape, DataError> {
    let first = samples.first().ok_or_else(|| {
        DataError::EmptyDataset("cannot validate an empty sample list".to_string())
    })?;
    let (time, frame_h, frame_w, channels) = first.frame_shape;
    let (action_time, action_dim) = first.action_shape;
    let shape = BatchShape {
        time,
        frame_h,
        frame_w,
        action_dim,
    };
    validate_shape_contract(0, first, shape, channels, action_time, normalizer_dim)?;

    for (index, sample) in samples.iter().enumerate().skip(1) {
        if sample.frame_shape != first.frame_shape {
            return Err(inconsistent_shapes(format!(
                "sample {index} frame_shape {:?} differs from reference {:?}",
                sample.frame_shape, first.frame_shape
            )));
        }
        if sample.action_shape != first.action_shape {
            return Err(inconsistent_shapes(format!(
                "sample {index} action_shape {:?} differs from reference {:?}",
                sample.action_shape, first.action_shape
            )));
        }
        validate_shape_contract(
            index,
            sample,
            shape,
            sample.frame_shape.3,
            sample.action_shape.0,
            normalizer_dim,
        )?;
    }

    Ok(shape)
}

fn validate_shape_contract(
    index: usize,
    sample: &Sample,
    shape: BatchShape,
    channels: usize,
    action_time: usize,
    normalizer_dim: usize,
) -> Result<(), DataError> {
    if shape.time != action_time {
        return Err(inconsistent_shapes(format!(
            "sample {index} has frame time {} but action time {action_time}",
            shape.time
        )));
    }
    if channels != RGB_CHANNELS {
        return Err(inconsistent_shapes(format!(
            "sample {index} has {channels} image channels, expected {RGB_CHANNELS}"
        )));
    }
    if shape.action_dim != normalizer_dim {
        return Err(inconsistent_shapes(format!(
            "sample {index} action_dim {} does not match normalizer dim {normalizer_dim}",
            shape.action_dim
        )));
    }

    let expected_frame_len = checked_mul(
        shape.time,
        checked_hwc_len(shape.frame_h, shape.frame_w, channels)?,
        "frame buffer size",
    )?;
    if sample.frames_t.len() != expected_frame_len {
        return Err(inconsistent_shapes(format!(
            "sample {index} has {} frame bytes, expected {expected_frame_len}",
            sample.frames_t.len()
        )));
    }

    let expected_action_len = checked_mul(shape.time, shape.action_dim, "action buffer size")?;
    if sample.actions.len() != expected_action_len {
        return Err(inconsistent_shapes(format!(
            "sample {index} has {} action values, expected {expected_action_len}",
            sample.actions.len()
        )));
    }

    Ok(())
}

fn append_sample_pixels(
    sample: &Sample,
    image_preproc: &ImagePreprocessor,
    source_frame_len: usize,
    pixels: &mut Vec<f32>,
) -> Result<(), DataError> {
    let src_h = u32::try_from(sample.frame_shape.1).map_err(|_| {
        DataError::InvalidTransform(format!(
            "source height {} does not fit u32",
            sample.frame_shape.1
        ))
    })?;
    let src_w = u32::try_from(sample.frame_shape.2).map_err(|_| {
        DataError::InvalidTransform(format!(
            "source width {} does not fit u32",
            sample.frame_shape.2
        ))
    })?;

    for frame_index in 0..sample.frame_shape.0 {
        let start = checked_mul(frame_index, source_frame_len, "frame offset")?;
        let end = start
            .checked_add(source_frame_len)
            .ok_or_else(|| DataError::InvalidTransform("frame offset overflow".to_string()))?;
        let frame = sample.frames_t.get(start..end).ok_or_else(|| {
            inconsistent_shapes(format!(
                "sample frame {frame_index} slice {start}..{end} is outside frame buffer length {}",
                sample.frames_t.len()
            ))
        })?;
        pixels.extend(image_preproc.apply(frame, src_h, src_w)?);
    }

    Ok(())
}

fn cast_f32_buffer_to_bf16(values: &mut [f32]) {
    for value in values {
        *value = round_f32_to_bf16(*value);
    }
}

fn round_f32_to_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounding_bias = 0x0000_7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(rounding_bias) & 0xffff_0000)
}

fn checked_shape_len<const D: usize>(shape: [usize; D], context: &str) -> Result<usize, DataError> {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| checked_mul(acc, *dim, context))
}

fn checked_hwc_len(height: usize, width: usize, channels: usize) -> Result<usize, DataError> {
    checked_mul(
        checked_mul(height, width, "image area")?,
        channels,
        "image buffer size",
    )
}

fn checked_mul(left: usize, right: usize, context: &str) -> Result<usize, DataError> {
    left.checked_mul(right)
        .ok_or_else(|| DataError::InvalidTransform(format!("{context} overflow")))
}

fn inconsistent_shapes(detail: String) -> DataError {
    DataError::InconsistentShapes { detail }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct F32OnlyBackend;

    impl BatchBackend for F32OnlyBackend {
        fn name(_device: &Self::Device) -> &'static str {
            "f32-only"
        }

        fn supports_dtype(_device: &Self::Device, dtype: BatchDtype) -> bool {
            dtype == BatchDtype::F32
        }

        type Device = HostDevice;
    }

    #[test]
    fn collate_shape_contract() -> Result<(), Box<dyn std::error::Error>> {
        let device = HostDevice::Cpu;
        let samples = vec![
            sample(0, [1.0, -1.0, 3.0, 3.0]),
            sample(10, [5.0, 7.0, 9.0, 11.0]),
        ];
        let image_preproc = ImagePreprocessor {
            target_size: 2,
            ..ImagePreprocessor::default()
        };
        let action_norm = ActionNormalizer::new(vec![1.0, -1.0], vec![2.0, 4.0])?;

        let batch = collate::<HostBackend>(
            &samples,
            &image_preproc,
            &action_norm,
            &device,
            BatchDtype::F32,
        )?;

        assert_eq!(batch.pixels.shape(), [2, 2, 3, 2, 2]);
        assert_eq!(batch.actions.shape(), [2, 2, 2]);
        assert_eq!(batch.pixels.dtype(), BatchDtype::F32);
        assert_eq!(batch.actions.dtype(), BatchDtype::F32);
        assert_eq!(batch.pixels.device(), &device);
        assert_eq!(batch.meta, vec![samples[0].meta, samples[1].meta]);
        assert_eq!(
            batch.actions.values(),
            &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 3.0]
        );
        Ok(())
    }

    #[test]
    fn collate_casts_pixels_to_bf16() -> Result<(), Box<dyn std::error::Error>> {
        let device = HostDevice::Cpu;
        let samples = vec![single_frame_sample([1, 2, 3], [0.0])];
        let image_preproc = ImagePreprocessor {
            target_size: 1,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            ..ImagePreprocessor::default()
        };
        let action_norm = ActionNormalizer::new(vec![0.0], vec![1.0])?;

        let batch = collate::<HostBackend>(
            &samples,
            &image_preproc,
            &action_norm,
            &device,
            BatchDtype::Bf16,
        )?;

        assert_eq!(batch.pixels.shape(), [1, 1, 3, 1, 1]);
        assert_eq!(batch.pixels.dtype(), BatchDtype::Bf16);
        assert_eq!(
            batch.pixels.values(),
            &[
                round_f32_to_bf16(f32::from(1_u8) / 255.0),
                round_f32_to_bf16(f32::from(2_u8) / 255.0),
                round_f32_to_bf16(f32::from(3_u8) / 255.0),
            ]
        );
        assert_eq!(batch.actions.dtype(), BatchDtype::F32);
        Ok(())
    }

    #[test]
    fn collate_rejects_inconsistent_shapes() -> Result<(), Box<dyn std::error::Error>> {
        let device = HostDevice::Cpu;
        let mut second = sample(10, [5.0, 7.0, 9.0, 11.0]);
        second.frame_shape = (1, 2, 2, 3);
        let samples = vec![sample(0, [1.0, -1.0, 3.0, 3.0]), second];
        let action_norm = ActionNormalizer::new(vec![0.0, 0.0], vec![1.0, 1.0])?;

        let err = collate::<HostBackend>(
            &samples,
            &ImagePreprocessor {
                target_size: 2,
                ..ImagePreprocessor::default()
            },
            &action_norm,
            &device,
            BatchDtype::F32,
        )
        .err()
        .ok_or("collate should reject inconsistent frame shapes")?;

        assert!(matches!(err, DataError::InconsistentShapes { .. }));
        Ok(())
    }

    #[test]
    fn collate_rejects_bf16_on_unsupported_backend() -> Result<(), Box<dyn std::error::Error>> {
        let device = HostDevice::Cpu;
        let samples = vec![sample(0, [1.0, -1.0, 3.0, 3.0])];
        let action_norm = ActionNormalizer::new(vec![0.0, 0.0], vec![1.0, 1.0])?;

        let err = collate::<F32OnlyBackend>(
            &samples,
            &ImagePreprocessor {
                target_size: 2,
                ..ImagePreprocessor::default()
            },
            &action_norm,
            &device,
            BatchDtype::Bf16,
        )
        .err()
        .ok_or("backend should reject BF16 pixel batches")?;

        assert!(matches!(err, DataError::InvalidConfig(_)));
        Ok(())
    }

    fn sample(offset: u8, actions: [f32; 4]) -> Sample {
        let mut frames_t = Vec::with_capacity(2 * 2 * 2 * RGB_CHANNELS);
        for frame in 0..2_u8 {
            for pixel in 0..4_u8 {
                frames_t.push(offset.saturating_add(frame).saturating_add(pixel));
                frames_t.push(offset.saturating_add(10).saturating_add(pixel));
                frames_t.push(offset.saturating_add(20).saturating_add(pixel));
            }
        }

        Sample {
            frames_t,
            frame_shape: (2, 2, 2, RGB_CHANNELS),
            actions: actions.to_vec(),
            action_shape: (2, 2),
            meta: SampleMeta {
                episode_id: u32::from(offset),
                start_frame: 0,
                shard: 0,
            },
        }
    }

    fn single_frame_sample(pixel: [u8; RGB_CHANNELS], actions: [f32; 1]) -> Sample {
        Sample {
            frames_t: pixel.to_vec(),
            frame_shape: (1, 1, 1, RGB_CHANNELS),
            actions: actions.to_vec(),
            action_shape: (1, 1),
            meta: SampleMeta {
                episode_id: 0,
                start_frame: 0,
                shard: 0,
            },
        }
    }
}
