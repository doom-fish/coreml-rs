#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn artifact_dir(suite: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("test-artifacts")
        .join(suite);
    fs::create_dir_all(&dir).expect("create test artifact directory");
    dir
}

pub fn asset_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("assets")
        .join(name)
}

pub fn compile_model(source: &Path, suite: &str) -> PathBuf {
    let output_dir = artifact_dir(suite);
    let compiled = output_dir.join(format!(
        "{}.mlmodelc",
        source
            .file_stem()
            .expect("model file stem")
            .to_string_lossy()
    ));
    if compiled.exists() {
        fs::remove_dir_all(&compiled).expect("remove stale compiled model");
    }
    let output = Command::new("xcrun")
        .args([
            "coremlcompiler",
            "compile",
            source.to_str().expect("utf-8 source path"),
            output_dir.to_str().expect("utf-8 output path"),
        ])
        .output()
        .expect("run coremlcompiler");
    assert!(
        output.status.success(),
        "failed to compile {}
stdout:
{}
stderr:
{}",
        source.display(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        compiled.exists(),
        "compiled model missing: {}",
        compiled.display()
    );
    compiled
}

pub fn unique_model_source(suite: &str, asset_stem: &str) -> (PathBuf, String) {
    let stem = format!("{asset_stem}_{suite}_{}", std::process::id()).replace('-', "_");
    let source = artifact_dir(suite).join(format!("{stem}.mlmodel"));
    fs::copy(asset_path(&format!("{asset_stem}.mlmodel")), &source)
        .expect("copy model source fixture");
    (source, stem)
}

pub fn temporary_bundles(stem: &str) -> usize {
    let exact = format!("{stem}.mlmodelc");
    let prefix = format!("{stem}_");
    let output = Command::new("getconf")
        .arg("DARWIN_USER_TEMP_DIR")
        .output()
        .expect("run getconf");
    let temp_dir = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    fs::read_dir(temp_dir)
        .expect("read the per-user temporary directory")
        .filter_map(Result::ok)
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name == exact || (name.starts_with(&prefix) && name.ends_with(".mlmodelc"))
        })
        .count()
}
