//! `PushT` evaluation report and artifact writers.

use std::fmt::Write as _;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{BooleanArray, Float32Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::{EvalError, PushtEvalReport, TrajectoryStep};

/// Render a deterministic Markdown summary for a `PushT` eval run.
pub fn render_pusht_report(report: &PushtEvalReport) -> String {
    let mut markdown = String::new();
    markdown.push_str("# PushT Evaluation Report\n\n");
    markdown.push_str("| Metric | Value |\n");
    markdown.push_str("|---|---:|\n");
    let _ = writeln!(
        &mut markdown,
        "| Planning success rate | {:.4} |",
        report.success_rate
    );
    let _ = writeln!(&mut markdown, "| Episodes | {} |", report.per_episode.len());
    let _ = writeln!(&mut markdown, "| Total steps | {} |", report.total_steps);
    let _ = writeln!(
        &mut markdown,
        "| Max steps per episode | {} |",
        report.max_steps_per_episode
    );
    let _ = writeln!(&mut markdown, "| Seed | {} |", report.seed);
    let _ = writeln!(
        &mut markdown,
        "| Wall time seconds | {:.3} |",
        report.wall_time_s
    );
    markdown.push_str("\n## Episodes\n\n");
    markdown.push_str("| Episode | Success | Steps | Final cost |\n");
    markdown.push_str("|---:|:---:|---:|---:|\n");
    for episode in &report.per_episode {
        let _ = writeln!(
            &mut markdown,
            "| {} | {} | {} | {:.6} |",
            episode.episode_id,
            if episode.success { "yes" } else { "no" },
            episode.steps_taken,
            episode.final_cost
        );
    }
    markdown
}

/// Write `results.json`, `report.md`, and `trajectories.parquet`.
///
/// # Errors
///
/// Returns an error when the output directory or any artifact cannot be written.
pub fn write_pusht_artifacts(
    report: &PushtEvalReport,
    output_dir: impl AsRef<Path>,
) -> Result<(), EvalError> {
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir).map_err(|source| EvalError::io(output_dir, source))?;

    let results_path = output_dir.join("results.json");
    let results_json = serde_json::to_vec_pretty(report)
        .map_err(|source| EvalError::json("serializing PushT results", source))?;
    std::fs::write(&results_path, results_json)
        .map_err(|source| EvalError::io(&results_path, source))?;

    let report_path = output_dir.join("report.md");
    std::fs::write(&report_path, render_pusht_report(report))
        .map_err(|source| EvalError::io(&report_path, source))?;

    let parquet_path = output_dir.join("trajectories.parquet");
    write_trajectories_parquet(&report.trajectories, &parquet_path)
}

fn write_trajectories_parquet(
    trajectories: &[TrajectoryStep],
    path: &Path,
) -> Result<(), EvalError> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("episode_id", DataType::UInt32, false),
        Field::new("step_index", DataType::UInt32, false),
        Field::new("action_json", DataType::Utf8, false),
        Field::new("cost", DataType::Float32, false),
        Field::new("reward", DataType::Float32, true),
        Field::new("done", DataType::Boolean, false),
        Field::new("success", DataType::Boolean, false),
    ]));

    let episode_ids = trajectories
        .iter()
        .map(|step| step.episode_id)
        .collect::<Vec<_>>();
    let step_indexes = trajectories
        .iter()
        .map(|step| step.step_index)
        .collect::<Vec<_>>();
    let action_json = trajectories
        .iter()
        .map(|step| {
            serde_json::to_string(&step.action)
                .map_err(|source| EvalError::json("serializing trajectory action", source))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let costs = trajectories
        .iter()
        .map(|step| step.cost)
        .collect::<Vec<_>>();
    let rewards = trajectories
        .iter()
        .map(|step| step.reward)
        .collect::<Vec<_>>();
    let done = trajectories
        .iter()
        .map(|step| step.done)
        .collect::<Vec<_>>();
    let success = trajectories
        .iter()
        .map(|step| step.success)
        .collect::<Vec<_>>();

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(UInt32Array::from(episode_ids)),
            Arc::new(UInt32Array::from(step_indexes)),
            Arc::new(StringArray::from(action_json)),
            Arc::new(Float32Array::from(costs)),
            Arc::new(Float32Array::from(rewards)),
            Arc::new(BooleanArray::from(done)),
            Arc::new(BooleanArray::from(success)),
        ],
    )?;

    let file = File::create(path).map_err(|source| EvalError::io(path, source))?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::{EpisodeOutcome, TrajectorySummary};

    use super::*;

    fn sample_report() -> PushtEvalReport {
        PushtEvalReport {
            success_rate: 0.5,
            per_episode: vec![
                EpisodeOutcome {
                    episode_id: 7,
                    success: true,
                    steps_taken: 5,
                    final_cost: 0.25,
                    trajectory_summary: TrajectorySummary {
                        first_state: vec![0.0],
                        final_state: vec![1.0],
                    },
                },
                EpisodeOutcome {
                    episode_id: 9,
                    success: false,
                    steps_taken: 10,
                    final_cost: 1.5,
                    trajectory_summary: TrajectorySummary {
                        first_state: vec![0.0],
                        final_state: vec![0.0],
                    },
                },
            ],
            wall_time_s: 12.3456,
            total_steps: 15,
            seed: 42,
            max_steps_per_episode: 10,
            trajectories: vec![TrajectoryStep {
                episode_id: 7,
                step_index: 0,
                action: vec![0.1, -0.2],
                cost: 0.25,
                reward: Some(1.0),
                done: true,
                success: true,
            }],
        }
    }

    #[test]
    fn report_md_format_stable() {
        let markdown = render_pusht_report(&sample_report());

        insta::assert_snapshot!(markdown, @r"
# PushT Evaluation Report

| Metric | Value |
|---|---:|
| Planning success rate | 0.5000 |
| Episodes | 2 |
| Total steps | 15 |
| Max steps per episode | 10 |
| Seed | 42 |
| Wall time seconds | 12.346 |

## Episodes

| Episode | Success | Steps | Final cost |
|---:|:---:|---:|---:|
| 7 | yes | 5 | 0.250000 |
| 9 | no | 10 | 1.500000 |
");
    }

    #[test]
    fn write_artifacts_creates_json_markdown_and_parquet() -> Result<(), Box<dyn std::error::Error>>
    {
        let suffix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let output_dir = std::env::temp_dir().join(format!("lewm-pusht-report-{suffix}"));

        write_pusht_artifacts(&sample_report(), &output_dir)?;

        assert!(output_dir.join("results.json").is_file());
        assert!(output_dir.join("report.md").is_file());
        assert!(output_dir.join("trajectories.parquet").is_file());

        std::fs::remove_dir_all(output_dir)?;
        Ok(())
    }
}
