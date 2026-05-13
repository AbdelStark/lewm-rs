//! RFC 0013 deterministic RNG sub-stream tree and serialization helpers.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::LewmCoreError;

/// Byte length of a serialized RFC 0013 `ChaCha20Rng` state.
pub const RNG_STATE_BYTES: usize = 48;

/// RFC 0013 data-shuffle RNG sub-stream name.
pub const DATA_SHUFFLE_STREAM: &str = "rng:data_shuffle";

/// RFC 0013 model-initialization RNG sub-stream name.
pub const MODEL_INIT_STREAM: &str = "rng:model_init";

/// RFC 0013 `SIGReg` sketch RNG sub-stream name.
pub const SIGREG_SKETCH_STREAM: &str = "rng:sigreg_sketch";

/// RFC 0013 dropout RNG sub-stream name.
pub const DROPOUT_STREAM: &str = "rng:dropout";

/// RFC 0013 CEM planner RNG sub-stream name.
pub const CEM_STREAM: &str = "rng:cem";

/// RFC 0013 miscellaneous RNG sub-stream name.
pub const MISC_STREAM: &str = "rng:misc";

/// Registered RFC 0013 RNG sub-stream names.
pub const RFC_0013_STREAMS: &[&str] = &[
    DATA_SHUFFLE_STREAM,
    MODEL_INIT_STREAM,
    SIGREG_SKETCH_STREAM,
    DROPOUT_STREAM,
    CEM_STREAM,
    MISC_STREAM,
];

/// Serializable `ChaCha20Rng` state.
#[derive(Debug, Clone, Copy, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct RngState {
    /// Original 32-byte sub-stream seed.
    pub seed: [u8; 32],
    /// Current `ChaCha` word offset.
    pub word_pos: u128,
}

impl RngState {
    /// Convert the state into the RFC 0013 fixed-width byte representation.
    #[must_use]
    pub fn to_bytes(self) -> [u8; RNG_STATE_BYTES] {
        let mut bytes = [0_u8; RNG_STATE_BYTES];
        bytes[..32].copy_from_slice(&self.seed);
        bytes[32..].copy_from_slice(&self.word_pos.to_le_bytes());
        bytes
    }

    /// Parse the RFC 0013 fixed-width byte representation.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::RngState`] when the input is not exactly 48 bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LewmCoreError> {
        if bytes.len() != RNG_STATE_BYTES {
            return Err(LewmCoreError::RngState {
                reason: format!(
                    "expected {RNG_STATE_BYTES} bytes, received {} bytes",
                    bytes.len()
                ),
            });
        }

        let mut seed = [0_u8; 32];
        seed.copy_from_slice(&bytes[..32]);
        let mut word_pos = [0_u8; 16];
        word_pos.copy_from_slice(&bytes[32..]);
        Ok(Self {
            seed,
            word_pos: u128::from_le_bytes(word_pos),
        })
    }
}

/// Return whether `name` is in the RFC 0013 RNG sub-stream tree.
#[must_use]
pub fn is_registered_substream(name: &str) -> bool {
    RFC_0013_STREAMS.contains(&name)
}

/// Derive a deterministic 32-byte seed for an RFC 0013 RNG sub-stream.
#[must_use]
pub fn substream_seed(global: u64, name: &str) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&global.to_le_bytes());
    hasher.update(b"::");
    hasher.update(name.as_bytes());
    *hasher.finalize().as_bytes()
}

/// Create a `ChaCha20Rng` for a named RFC 0013 sub-stream.
///
/// # Errors
///
/// Returns [`LewmCoreError::RngSubstream`] when `name` is not one of the
/// registered RFC 0013 stream names.
pub fn substream_rng(global: u64, name: &str) -> Result<ChaCha20Rng, LewmCoreError> {
    if !is_registered_substream(name) {
        return Err(LewmCoreError::RngSubstream {
            name: name.to_owned(),
        });
    }

    Ok(ChaCha20Rng::from_seed(substream_seed(global, name)))
}

/// Capture the seed and word position needed to restore `rng`.
#[must_use]
pub fn rng_state(rng: &ChaCha20Rng) -> RngState {
    RngState {
        seed: rng.get_seed(),
        word_pos: rng.get_word_pos(),
    }
}

/// Serialize an RNG state as `(seed: [u8; 32], word_pos: u128)`.
#[must_use]
pub fn serialize_rng(rng: &ChaCha20Rng) -> Vec<u8> {
    rng_state(rng).to_bytes().to_vec()
}

/// Restore a `ChaCha20Rng` from [`serialize_rng`] bytes.
///
/// # Errors
///
/// Returns [`LewmCoreError::RngState`] when `bytes` is not exactly 48 bytes.
pub fn deserialize_rng(bytes: &[u8]) -> Result<ChaCha20Rng, LewmCoreError> {
    let state = RngState::from_bytes(bytes)?;
    let mut rng = ChaCha20Rng::from_seed(state.seed);
    rng.set_word_pos(state.word_pos);
    Ok(rng)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use rand::RngCore;

    use super::*;

    #[test]
    fn substream_seed_distinct() {
        let seeds = RFC_0013_STREAMS
            .iter()
            .map(|stream| substream_seed(0, stream))
            .collect::<BTreeSet<_>>();

        assert_eq!(seeds.len(), RFC_0013_STREAMS.len());
        assert_eq!(
            substream_seed(0, MODEL_INIT_STREAM),
            substream_seed(0, MODEL_INIT_STREAM)
        );
        assert_ne!(
            substream_seed(0, MODEL_INIT_STREAM),
            substream_seed(1, MODEL_INIT_STREAM)
        );
    }

    #[test]
    fn substream_rng_rejects_unknown_streams() {
        assert!(substream_rng(0, "rng:unknown").is_err());
    }

    #[test]
    fn rng_serialize_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        let mut original = substream_rng(42, DROPOUT_STREAM)?;
        let _first_draw = original.next_u64();
        let _second_draw = original.next_u32();

        let bytes = serialize_rng(&original);
        assert_eq!(bytes.len(), RNG_STATE_BYTES);

        let state = RngState::from_bytes(&bytes)?;
        assert_eq!(state.seed, original.get_seed());
        assert_eq!(state.word_pos, original.get_word_pos());

        let mut restored = deserialize_rng(&bytes)?;
        assert_eq!(restored.get_seed(), original.get_seed());
        assert_eq!(restored.get_word_pos(), original.get_word_pos());

        for _ in 0..16 {
            assert_eq!(restored.next_u64(), original.next_u64());
        }
        Ok(())
    }

    #[test]
    fn rng_deserialize_rejects_wrong_length() {
        let err =
            deserialize_rng(&[0_u8; RNG_STATE_BYTES - 1]).expect_err("invalid length should fail");
        assert!(matches!(err, LewmCoreError::RngState { .. }));
    }
}
