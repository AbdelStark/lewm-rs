use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

const FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/parity_fixture.npz"
);
const FIXTURE_META_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/parity_fixture.meta.json"
);

pub(crate) struct ParityFixture {
    pub(crate) pixels: NpyF32,
    pub(crate) actions: NpyF32,
    pub(crate) seed: i32,
    pub(crate) git_short_sha: String,
    pub(crate) fixture_hash: String,
}

pub(crate) struct NpyF32 {
    pub(crate) shape: Vec<usize>,
    pub(crate) values: Vec<f32>,
}

#[derive(Debug)]
pub(crate) struct FixtureError(String);

impl fmt::Display for FixtureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for FixtureError {}

pub(crate) fn load_fixture() -> Result<ParityFixture, FixtureError> {
    let fixture_bytes = fs::read(FIXTURE_PATH).map_err(|err| {
        FixtureError(format!(
            "failed to read parity fixture at {FIXTURE_PATH}: {err}"
        ))
    })?;
    let entries = read_npz(&fixture_bytes)?;
    let pixels = read_f32_array(&entries, "pixels.npy")?;
    let actions = read_f32_array(&entries, "actions.npy")?;
    let seed = read_i32_scalar(&entries, "seed.npy")?;
    let git_short_sha = read_bytes_scalar(&entries, "git_short_sha.npy")?;
    let fixture_hash = blake3::hash(&fixture_bytes).to_hex().to_string();

    Ok(ParityFixture {
        pixels,
        actions,
        seed,
        git_short_sha,
        fixture_hash,
    })
}

pub(crate) fn load_fixture_meta() -> Result<serde_json::Value, FixtureError> {
    let raw = fs::read_to_string(Path::new(FIXTURE_META_PATH)).map_err(|err| {
        FixtureError(format!(
            "failed to read parity fixture metadata at {FIXTURE_META_PATH}: {err}"
        ))
    })?;
    serde_json::from_str(&raw)
        .map_err(|err| FixtureError(format!("invalid parity fixture metadata JSON: {err}")))
}

fn read_npz(bytes: &[u8]) -> Result<HashMap<String, Vec<u8>>, FixtureError> {
    let mut entries = HashMap::new();
    let mut offset = 0usize;

    while offset + 4 <= bytes.len() {
        let signature = read_u32_le(bytes, offset)?;
        if signature == 0x0201_4b50 {
            break;
        }
        if signature != 0x0403_4b50 {
            return Err(FixtureError(format!(
                "unexpected zip local header signature 0x{signature:08x}"
            )));
        }
        if offset + 30 > bytes.len() {
            return Err(FixtureError("truncated zip local header".to_owned()));
        }

        let flags = read_u16_le(bytes, offset + 6)?;
        let compression = read_u16_le(bytes, offset + 8)?;
        let compressed_size = read_u32_le(bytes, offset + 18)?;
        let uncompressed_size = read_u32_le(bytes, offset + 22)?;
        let file_name_len = read_u16_le(bytes, offset + 26)? as usize;
        let extra_len = read_u16_le(bytes, offset + 28)? as usize;
        if flags & 0x08 != 0 {
            return Err(FixtureError(
                "zip data descriptors are not supported for parity fixtures".to_owned(),
            ));
        }
        if compression != 0 {
            return Err(FixtureError(format!(
                "unsupported compressed zip entry method {compression}"
            )));
        }

        let name_start = offset + 30;
        let name_end = name_start
            .checked_add(file_name_len)
            .ok_or_else(|| FixtureError("zip file name offset overflowed".to_owned()))?;
        let extra_end = name_end
            .checked_add(extra_len)
            .ok_or_else(|| FixtureError("zip extra offset overflowed".to_owned()))?;
        if extra_end > bytes.len() {
            return Err(FixtureError("truncated zip extra data".to_owned()));
        }
        let (compressed_size, uncompressed_size) = zip_entry_sizes(
            compressed_size,
            uncompressed_size,
            &bytes[name_end..extra_end],
        )?;
        let data_start = extra_end;
        let data_end = data_start
            .checked_add(compressed_size)
            .ok_or_else(|| FixtureError("zip data size overflowed".to_owned()))?;
        if data_end > bytes.len() {
            return Err(FixtureError("truncated zip entry data".to_owned()));
        }

        let name = std::str::from_utf8(&bytes[name_start..name_end])
            .map_err(|err| FixtureError(format!("invalid zip entry name: {err}")))?;
        let data = bytes[data_start..data_end].to_vec();
        if data.len() != uncompressed_size {
            return Err(FixtureError(format!(
                "zip entry {name} size mismatch: expected {uncompressed_size}, got {}",
                data.len()
            )));
        }
        entries.insert(name.to_owned(), data);
        offset = data_end;
    }

    Ok(entries)
}

fn zip_entry_sizes(
    compressed_size: u32,
    uncompressed_size: u32,
    extra: &[u8],
) -> Result<(usize, usize), FixtureError> {
    if compressed_size != u32::MAX && uncompressed_size != u32::MAX {
        return Ok((compressed_size as usize, uncompressed_size as usize));
    }

    let mut offset = 0usize;
    while offset + 4 <= extra.len() {
        let header_id = read_u16_le(extra, offset)?;
        let data_size = read_u16_le(extra, offset + 2)? as usize;
        let data_start = offset + 4;
        let data_end = data_start
            .checked_add(data_size)
            .ok_or_else(|| FixtureError("zip64 extra size overflowed".to_owned()))?;
        if data_end > extra.len() {
            return Err(FixtureError("truncated zip64 extra data".to_owned()));
        }
        if header_id == 0x0001 {
            if data_size < 16 {
                return Err(FixtureError(
                    "zip64 size extra field is too short".to_owned(),
                ));
            }
            let uncompressed = read_u64_le(extra, data_start)?;
            let compressed = read_u64_le(extra, data_start + 8)?;
            let compressed = usize::try_from(compressed)
                .map_err(|err| FixtureError(format!("zip64 compressed size overflowed: {err}")))?;
            let uncompressed = usize::try_from(uncompressed).map_err(|err| {
                FixtureError(format!("zip64 uncompressed size overflowed: {err}"))
            })?;
            return Ok((compressed, uncompressed));
        }
        offset = data_end;
    }

    Err(FixtureError(
        "zip entry used zip64 sizes without a zip64 extra field".to_owned(),
    ))
}

fn read_f32_array(entries: &HashMap<String, Vec<u8>>, name: &str) -> Result<NpyF32, FixtureError> {
    let array = entries
        .get(name)
        .ok_or_else(|| FixtureError(format!("missing npz entry {name}")))?;
    let parsed = parse_npy(array)?;
    if parsed.descr != "<f4" {
        return Err(FixtureError(format!(
            "{name} has dtype {}, expected <f4",
            parsed.descr
        )));
    }
    let expected_bytes = element_count(&parsed.shape)?
        .checked_mul(4)
        .ok_or_else(|| FixtureError(format!("{name} byte count overflowed")))?;
    if parsed.data.len() != expected_bytes {
        return Err(FixtureError(format!(
            "{name} data length mismatch: expected {expected_bytes}, got {}",
            parsed.data.len()
        )));
    }

    let values = parsed
        .data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(NpyF32 {
        shape: parsed.shape,
        values,
    })
}

fn read_i32_scalar(entries: &HashMap<String, Vec<u8>>, name: &str) -> Result<i32, FixtureError> {
    let array = entries
        .get(name)
        .ok_or_else(|| FixtureError(format!("missing npz entry {name}")))?;
    let parsed = parse_npy(array)?;
    if parsed.descr != "<i4" || !parsed.shape.is_empty() || parsed.data.len() != 4 {
        return Err(FixtureError(format!(
            "{name} must be a scalar little-endian i32"
        )));
    }
    Ok(i32::from_le_bytes([
        parsed.data[0],
        parsed.data[1],
        parsed.data[2],
        parsed.data[3],
    ]))
}

fn read_bytes_scalar(
    entries: &HashMap<String, Vec<u8>>,
    name: &str,
) -> Result<String, FixtureError> {
    let array = entries
        .get(name)
        .ok_or_else(|| FixtureError(format!("missing npz entry {name}")))?;
    let parsed = parse_npy(array)?;
    if !parsed.descr.starts_with("|S") || !parsed.shape.is_empty() {
        return Err(FixtureError(format!(
            "{name} must be a scalar fixed-width byte string"
        )));
    }
    let end = parsed
        .data
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(parsed.data.len());
    String::from_utf8(parsed.data[..end].to_vec())
        .map_err(|err| FixtureError(format!("{name} is not valid UTF-8: {err}")))
}

struct ParsedNpy<'a> {
    descr: String,
    shape: Vec<usize>,
    data: &'a [u8],
}

fn parse_npy(bytes: &[u8]) -> Result<ParsedNpy<'_>, FixtureError> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(FixtureError("invalid npy magic".to_owned()));
    }
    let major = bytes[6];
    let header_len;
    let data_offset;
    match major {
        1 => {
            header_len = read_u16_le(bytes, 8)? as usize;
            data_offset = 10usize
                .checked_add(header_len)
                .ok_or_else(|| FixtureError("npy header offset overflowed".to_owned()))?;
        },
        2 | 3 => {
            header_len = read_u32_le(bytes, 8)? as usize;
            data_offset = 12usize
                .checked_add(header_len)
                .ok_or_else(|| FixtureError("npy header offset overflowed".to_owned()))?;
        },
        _ => return Err(FixtureError(format!("unsupported npy version {major}"))),
    }
    if data_offset > bytes.len() {
        return Err(FixtureError("truncated npy header".to_owned()));
    }

    let header_start = if major == 1 { 10 } else { 12 };
    let header = std::str::from_utf8(&bytes[header_start..data_offset])
        .map_err(|err| FixtureError(format!("invalid npy header: {err}")))?;
    if !header.contains("'fortran_order': False") {
        return Err(FixtureError(
            "parity fixture arrays must be C-order".to_owned(),
        ));
    }
    let descr = parse_header_string(header, "descr")?;
    let shape = parse_shape(header)?;

    Ok(ParsedNpy {
        descr,
        shape,
        data: &bytes[data_offset..],
    })
}

fn parse_header_string(header: &str, key: &str) -> Result<String, FixtureError> {
    let marker = format!("'{key}': '");
    let start = header
        .find(&marker)
        .ok_or_else(|| FixtureError(format!("npy header missing {key}")))?
        + marker.len();
    let end = header[start..]
        .find('\'')
        .ok_or_else(|| FixtureError(format!("npy header has unterminated {key}")))?
        + start;
    Ok(header[start..end].to_owned())
}

fn parse_shape(header: &str) -> Result<Vec<usize>, FixtureError> {
    let marker = "'shape': (";
    let start = header
        .find(marker)
        .ok_or_else(|| FixtureError("npy header missing shape".to_owned()))?
        + marker.len();
    let end = header[start..]
        .find(')')
        .ok_or_else(|| FixtureError("npy header has unterminated shape".to_owned()))?
        + start;
    let raw = header[start..end].trim();
    if raw.is_empty() {
        return Ok(Vec::new());
    }

    raw.split(',')
        .filter_map(|part| {
            let part = part.trim();
            if part.is_empty() { None } else { Some(part) }
        })
        .map(|part| {
            part.parse::<usize>()
                .map_err(|err| FixtureError(format!("invalid npy shape dimension {part}: {err}")))
        })
        .collect()
}

fn element_count(shape: &[usize]) -> Result<usize, FixtureError> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| FixtureError("npy element count overflowed".to_owned()))
    })
}

fn read_u16_le(bytes: &[u8], offset: usize) -> Result<u16, FixtureError> {
    if offset + 2 > bytes.len() {
        return Err(FixtureError("truncated u16".to_owned()));
    }
    Ok(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32, FixtureError> {
    if offset + 4 > bytes.len() {
        return Err(FixtureError("truncated u32".to_owned()));
    }
    Ok(u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}

fn read_u64_le(bytes: &[u8], offset: usize) -> Result<u64, FixtureError> {
    if offset + 8 > bytes.len() {
        return Err(FixtureError("truncated u64".to_owned()));
    }
    Ok(u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ]))
}
