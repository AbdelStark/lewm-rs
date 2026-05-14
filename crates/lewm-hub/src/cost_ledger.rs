//! Cost-ledger parsing, appending, and integrity checks.

use std::{fmt, fs, path::Path, time::Duration};

/// Project hard cap for Hugging Face Jobs spend, in cents.
pub const HARD_CAP_USD_CENTS: u64 = 20_000;

const LEDGER_HEADER: &str = r"# `lewm-rs` cost ledger

> Updated automatically by `lewm-hub::cost_ledger::append_entry` at every job termination.
> Manual entries are forbidden; use `cost_ledger::backfill --from <job_url>` to import.

| Date (UTC)          | Phase | Job ID            | Hardware     | Wall   | Cost (USD) | Cumulative (USD) | Notes |
|---------------------|-------|-------------------|--------------|--------|-----------:|----------------:|-------|
";

/// Monetary amount represented as whole USD cents.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub struct UsdAmount {
    cents: u64,
}

impl UsdAmount {
    /// Construct an amount from USD cents.
    pub const fn from_cents(cents: u64) -> Self {
        Self { cents }
    }

    /// Return the amount in USD cents.
    pub const fn cents(self) -> u64 {
        self.cents
    }

    /// Parse a non-negative amount with at most two decimal places.
    ///
    /// # Errors
    ///
    /// Returns an error when the amount is negative, malformed, or more precise
    /// than cents.
    pub fn parse(value: &str) -> Result<Self, CostLedgerError> {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.starts_with('-') || trimmed.starts_with('+') {
            return Err(CostLedgerError::InvalidAmount(value.to_owned()));
        }

        let (dollars_part, cents_part) = match trimmed.split_once('.') {
            Some((dollars, cents)) => (dollars, Some(cents)),
            None => (trimmed, None),
        };

        if dollars_part.is_empty()
            || !dollars_part
                .chars()
                .all(|character| character.is_ascii_digit())
        {
            return Err(CostLedgerError::InvalidAmount(value.to_owned()));
        }

        let dollars = dollars_part
            .parse::<u64>()
            .map_err(|_| CostLedgerError::InvalidAmount(value.to_owned()))?;
        let dollars_cents = dollars
            .checked_mul(100)
            .ok_or(CostLedgerError::AmountOverflow)?;
        let cents = match cents_part {
            None => 0,
            Some(part)
                if !part.is_empty()
                    && part.len() <= 2
                    && part.chars().all(|character| character.is_ascii_digit()) =>
            {
                let mut padded = part.to_owned();
                while padded.len() < 2 {
                    padded.push('0');
                }
                padded
                    .parse::<u64>()
                    .map_err(|_| CostLedgerError::InvalidAmount(value.to_owned()))?
            },
            Some(_) => return Err(CostLedgerError::InvalidAmount(value.to_owned())),
        };

        Ok(Self::from_cents(
            dollars_cents
                .checked_add(cents)
                .ok_or(CostLedgerError::AmountOverflow)?,
        ))
    }

    fn checked_add(self, other: Self) -> Result<Self, CostLedgerError> {
        Ok(Self::from_cents(
            self.cents
                .checked_add(other.cents)
                .ok_or(CostLedgerError::AmountOverflow)?,
        ))
    }
}

impl fmt::Display for UsdAmount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{:02}", self.cents / 100, self.cents % 100)
    }
}

/// A ledger entry before the cumulative column is recomputed.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CostEntry {
    /// UTC timestamp in `YYYY-MM-DD HH:MM:SS` form.
    pub date_utc: String,
    /// Project phase, for example `P3`.
    pub phase: String,
    /// Hugging Face Jobs id.
    pub job_id: String,
    /// Hugging Face hardware flavor.
    pub hardware: String,
    /// Wall-clock duration, for example `0:28:13`.
    pub wall: String,
    /// Conservatively rounded cost in USD.
    pub cost_usd: UsdAmount,
    /// Free-form note without Markdown table delimiters.
    pub notes: String,
}

/// A ledger row with its stored cumulative amount.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CostLedgerRow {
    /// Entry fields for the row.
    pub entry: CostEntry,
    /// Stored cumulative amount for this row.
    pub cumulative_usd: UsdAmount,
}

/// Cost-ledger failures.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum CostLedgerError {
    /// Filesystem operation failed.
    #[error("cost-ledger I/O failed: {0}")]
    Io(String),

    /// Markdown ledger content did not match the expected table format.
    #[error("invalid cost-ledger content: {0}")]
    InvalidLedger(String),

    /// A money value could not be parsed as USD cents.
    #[error("invalid USD amount: {0}")]
    InvalidAmount(String),

    /// A field cannot be rendered safely inside a Markdown table cell.
    #[error("invalid Markdown table field {field}: {value}")]
    InvalidField {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: String,
    },

    /// Cumulative spend exceeded the configured cap.
    #[error("cost-ledger cap exceeded: cumulative {cumulative} USD > cap {cap} USD")]
    CapExceeded {
        /// Cumulative amount.
        cumulative: UsdAmount,
        /// Configured cap.
        cap: UsdAmount,
    },

    /// Stored cumulative column does not match the recomputed cumulative sum.
    #[error(
        "cost-ledger cumulative mismatch at row {row}: stored {stored} USD, expected {expected} USD"
    )]
    CumulativeMismatch {
        /// One-based ledger row number.
        row: usize,
        /// Stored cumulative amount.
        stored: UsdAmount,
        /// Recomputed cumulative amount.
        expected: UsdAmount,
    },

    /// Integer overflow while computing monetary amounts.
    #[error("cost-ledger amount overflow")]
    AmountOverflow,
}

impl From<std::io::Error> for CostLedgerError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

/// Append an entry, rewrite cumulative values from scratch, and enforce the hard cap.
///
/// Missing ledger files are created with the RFC 0010 header.
///
/// # Errors
///
/// Returns an error when the existing ledger is malformed, an entry cannot be
/// rendered safely, filesystem access fails, cumulative values are inconsistent,
/// or the hard cap would be exceeded.
pub fn append_entry(
    entry: CostEntry,
    ledger_path: &Path,
) -> Result<Vec<CostLedgerRow>, CostLedgerError> {
    let mut entries = read_ledger(ledger_path)?
        .into_iter()
        .map(|row| row.entry)
        .collect::<Vec<_>>();
    entries.push(entry);

    let rows = rows_from_entries(&entries, UsdAmount::from_cents(HARD_CAP_USD_CENTS))?;
    write_ledger(ledger_path, &rows)?;
    Ok(rows)
}

/// Read a ledger and verify its cumulative column against the hard cap.
///
/// # Errors
///
/// Returns an error when the ledger cannot be read or parsed, when cumulative
/// values are inconsistent, or when the hard cap is exceeded.
pub fn read_ledger(ledger_path: &Path) -> Result<Vec<CostLedgerRow>, CostLedgerError> {
    if !ledger_path.exists() {
        return Ok(Vec::new());
    }

    let raw = fs::read_to_string(ledger_path)?;
    let mut rows = Vec::new();
    for line in raw.lines() {
        if let Some(row) = parse_table_row(line)? {
            rows.push(row);
        }
    }
    verify_rows(&rows, UsdAmount::from_cents(HARD_CAP_USD_CENTS))?;
    Ok(rows)
}

/// Verify that a ledger is internally consistent and under a supplied cap.
///
/// # Errors
///
/// Returns an error when parsing fails, cumulative values are inconsistent, or
/// any cumulative row exceeds `cap`.
pub fn verify_ledger(
    ledger_path: &Path,
    cap: UsdAmount,
) -> Result<Vec<CostLedgerRow>, CostLedgerError> {
    if !ledger_path.exists() {
        return Ok(Vec::new());
    }

    let raw = fs::read_to_string(ledger_path)?;
    let mut rows = Vec::new();
    for line in raw.lines() {
        if let Some(row) = parse_table_row(line)? {
            rows.push(row);
        }
    }
    verify_rows(&rows, cap)?;
    Ok(rows)
}

/// Return the conservatively rounded billable minutes for a wall-clock duration.
pub fn rounded_billable_minutes(wall: Duration) -> u64 {
    let seconds = wall
        .as_secs()
        .saturating_add(u64::from(wall.subsec_nanos() > 0));
    seconds.div_ceil(60)
}

/// Compute cost from a wall-clock duration and a hardware price in cents/hour.
///
/// The wall time is rounded up to the nearest minute, then the final cost is
/// rounded up to the nearest cent so the ledger never understates spend.
///
/// # Errors
///
/// Returns an error on integer overflow.
pub fn cost_for_wall_time(
    wall: Duration,
    price_cents_per_hour: u64,
) -> Result<UsdAmount, CostLedgerError> {
    let minutes = rounded_billable_minutes(wall);
    let numerator = price_cents_per_hour
        .checked_mul(minutes)
        .ok_or(CostLedgerError::AmountOverflow)?;
    Ok(UsdAmount::from_cents(numerator.div_ceil(60)))
}

fn rows_from_entries(
    entries: &[CostEntry],
    cap: UsdAmount,
) -> Result<Vec<CostLedgerRow>, CostLedgerError> {
    let mut cumulative = UsdAmount::from_cents(0);
    let mut rows = Vec::with_capacity(entries.len());

    for entry in entries {
        validate_entry(entry)?;
        cumulative = cumulative.checked_add(entry.cost_usd)?;
        if cumulative > cap {
            return Err(CostLedgerError::CapExceeded { cumulative, cap });
        }
        rows.push(CostLedgerRow {
            entry: entry.clone(),
            cumulative_usd: cumulative,
        });
    }

    Ok(rows)
}

fn verify_rows(rows: &[CostLedgerRow], cap: UsdAmount) -> Result<(), CostLedgerError> {
    let mut expected = UsdAmount::from_cents(0);
    for (index, row) in rows.iter().enumerate() {
        validate_entry(&row.entry)?;
        expected = expected.checked_add(row.entry.cost_usd)?;
        if row.cumulative_usd != expected {
            return Err(CostLedgerError::CumulativeMismatch {
                row: index + 1,
                stored: row.cumulative_usd,
                expected,
            });
        }
        if row.cumulative_usd > cap {
            return Err(CostLedgerError::CapExceeded {
                cumulative: row.cumulative_usd,
                cap,
            });
        }
    }
    Ok(())
}

fn write_ledger(ledger_path: &Path, rows: &[CostLedgerRow]) -> Result<(), CostLedgerError> {
    if let Some(parent) = ledger_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut rendered = LEDGER_HEADER.to_owned();
    for row in rows {
        rendered.push_str(&format_row(row)?);
    }
    fs::write(ledger_path, rendered)?;
    Ok(())
}

fn format_row(row: &CostLedgerRow) -> Result<String, CostLedgerError> {
    Ok(format!(
        "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
        markdown_cell("date_utc", &row.entry.date_utc, false)?,
        markdown_cell("phase", &row.entry.phase, false)?,
        markdown_cell("job_id", &row.entry.job_id, false)?,
        markdown_cell("hardware", &row.entry.hardware, false)?,
        markdown_cell("wall", &row.entry.wall, false)?,
        row.entry.cost_usd,
        row.cumulative_usd,
        markdown_cell("notes", &row.entry.notes, true)?,
    ))
}

fn validate_entry(entry: &CostEntry) -> Result<(), CostLedgerError> {
    markdown_cell("date_utc", &entry.date_utc, false)?;
    markdown_cell("phase", &entry.phase, false)?;
    markdown_cell("job_id", &entry.job_id, false)?;
    markdown_cell("hardware", &entry.hardware, false)?;
    markdown_cell("wall", &entry.wall, false)?;
    markdown_cell("notes", &entry.notes, true)?;
    Ok(())
}

fn markdown_cell(
    field: &'static str,
    value: &str,
    allow_empty: bool,
) -> Result<String, CostLedgerError> {
    let trimmed = value.trim();
    if (!allow_empty && trimmed.is_empty())
        || trimmed.contains('|')
        || trimmed.contains('\n')
        || trimmed.contains('\r')
    {
        return Err(CostLedgerError::InvalidField {
            field,
            value: value.to_owned(),
        });
    }
    Ok(trimmed.to_owned())
}

fn parse_table_row(line: &str) -> Result<Option<CostLedgerRow>, CostLedgerError> {
    let trimmed = line.trim();
    if !trimmed.starts_with('|') || !trimmed.ends_with('|') {
        return Ok(None);
    }

    let cells = trimmed
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .collect::<Vec<_>>();

    match cells.as_slice() {
        [
            "Date (UTC)",
            "Phase",
            "Job ID",
            "Hardware",
            "Wall",
            "Cost (USD)",
            "Cumulative (USD)",
            "Notes",
        ] => Ok(None),
        [separator, _, _, _, _, _, _, _] if separator.chars().all(|character| character == '-') => {
            Ok(None)
        },
        [
            date_utc,
            phase,
            job_id,
            hardware,
            wall,
            cost_usd,
            cumulative_usd,
            notes,
        ] => {
            let entry = CostEntry {
                date_utc: (*date_utc).to_owned(),
                phase: (*phase).to_owned(),
                job_id: (*job_id).to_owned(),
                hardware: (*hardware).to_owned(),
                wall: (*wall).to_owned(),
                cost_usd: UsdAmount::parse(cost_usd)?,
                notes: (*notes).to_owned(),
            };
            let cumulative_usd = UsdAmount::parse(cumulative_usd)?;
            Ok(Some(CostLedgerRow {
                entry,
                cumulative_usd,
            }))
        },
        _ => Err(CostLedgerError::InvalidLedger(format!(
            "expected 8 table cells, got {} in line: {line}",
            cells.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn cost_ledger_cumulative_correct() -> Result<(), Box<dyn std::error::Error>> {
        let path = unique_ledger_path("cumulative");

        append_entry(sample_entry("hfjob-1", 38), &path)?;
        let rows = append_entry(sample_entry("hfjob-2", 112), &path)?;
        let raw = fs::read_to_string(&path)?;

        assert_eq!(rows.len(), 2);
        assert!(raw.contains("| 2026-05-12 15:03:00 | P3 | hfjob-2 | a10g-large | 0:44:10 | 1.12 | 1.50 | smoke pusht |"));
        assert_eq!(verify_ledger(&path, UsdAmount::from_cents(20_000))?, rows);

        remove_file(&path)?;
        Ok(())
    }

    #[test]
    fn cost_ledger_under_200_usd() -> Result<(), Box<dyn std::error::Error>> {
        let path = unique_ledger_path("under-cap");

        append_entry(sample_entry("hfjob-1", 19_999), &path)?;
        let Err(error) = append_entry(sample_entry("hfjob-2", 2), &path) else {
            return Err("cap should be enforced".into());
        };

        assert_eq!(
            error,
            CostLedgerError::CapExceeded {
                cumulative: UsdAmount::from_cents(20_001),
                cap: UsdAmount::from_cents(20_000),
            }
        );

        remove_file(&path)?;
        Ok(())
    }

    #[test]
    fn cost_calculation_rounds_up_to_nearest_minute() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(rounded_billable_minutes(Duration::from_secs(60)), 1);
        assert_eq!(rounded_billable_minutes(Duration::from_secs(61)), 2);
        assert_eq!(
            cost_for_wall_time(Duration::from_secs(60), 150)?,
            UsdAmount::from_cents(3)
        );
        assert_eq!(
            cost_for_wall_time(Duration::from_secs(61), 150)?,
            UsdAmount::from_cents(5)
        );
        Ok(())
    }

    fn sample_entry(job_id: &str, cost_cents: u64) -> CostEntry {
        CostEntry {
            date_utc: "2026-05-12 15:03:00".to_owned(),
            phase: "P3".to_owned(),
            job_id: job_id.to_owned(),
            hardware: "a10g-large".to_owned(),
            wall: "0:44:10".to_owned(),
            cost_usd: UsdAmount::from_cents(cost_cents),
            notes: "smoke pusht".to_owned(),
        }
    }

    fn unique_ledger_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        std::env::temp_dir().join(format!(
            "lewm-hub-cost-ledger-{name}-{}-{nanos}.md",
            std::process::id()
        ))
    }

    fn remove_file(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }
}
