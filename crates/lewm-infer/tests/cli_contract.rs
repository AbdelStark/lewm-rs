//! CLI contract tests for `lewm-infer`.
#![cfg(feature = "tract-nnef")]

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use image::{Rgb, RgbImage};
use serde_json::Value;

const ENCODER_NNEF: &str = r"
version 1.0;

graph encoder( input ) -> ( output )
{
    input = external(shape = [1, 3, 224, 224]);
    output = relu(input);
}
";

const PREDICTOR_NNEF: &str = r"
version 1.0;

graph predictor( history, actions ) -> ( output )
{
    history = external(shape = [1, 2, 150528]);
    actions = external(shape = [1, 2, 3]);
    output = relu(history);
}
";

#[test]
fn plan_cli_emits_cost_actions_and_latency_json() -> Result<(), Box<dyn std::error::Error>> {
    let root = unique_temp_dir("lewm-infer-cli-plan")?;
    write_nnef_archive(&root.join("encoder.nnef"), ENCODER_NNEF)?;
    write_nnef_archive(&root.join("predictor.nnef"), PREDICTOR_NNEF)?;
    let start = root.join("start.png");
    let goal = root.join("goal.png");
    let image = RgbImage::from_pixel(2, 3, Rgb([255, 255, 255]));
    image.save(&start)?;
    image.save(&goal)?;

    let output = Command::new(env!("CARGO_BIN_EXE_lewm-infer"))
        .arg("--checkpoint-dir")
        .arg(&root)
        .arg("--action-dim")
        .arg("3")
        .arg("plan")
        .arg("--start")
        .arg(&start)
        .arg("--goal")
        .arg(&goal)
        .arg("--horizon")
        .arg("1")
        .arg("--n-cand")
        .arg("2")
        .arg("--n-iter")
        .arg("1")
        .output()?;

    if !output.status.success() {
        return Err(format!(
            "lewm-infer plan failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    let payload: Value = serde_json::from_slice(&output.stdout)?;
    assert!(payload.get("cost").and_then(Value::as_f64).is_some());
    assert!(payload.get("latency_ms").and_then(Value::as_f64).is_some());
    assert_eq!(
        payload
            .get("best_actions")
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(1)
    );
    assert_eq!(
        payload
            .get("best_actions")
            .and_then(Value::as_array)
            .and_then(|rows| rows.first())
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(3)
    );

    fs::remove_dir_all(root)?;
    Ok(())
}

fn write_nnef_archive(path: &std::path::Path, graph: &str) -> std::io::Result<()> {
    let file = fs::File::create(path)?;
    let mut archive = tar::Builder::new(file);
    let mut header = tar::Header::new_gnu();
    header.set_size(graph.len().try_into().map_err(std::io::Error::other)?);
    header.set_mode(0o644);
    header.set_cksum();
    archive.append_data(&mut header, "graph.nnef", graph.as_bytes())?;
    archive.finish()
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
