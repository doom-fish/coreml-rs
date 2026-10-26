use std::env;
use std::process::Command;

#[derive(Clone, Copy, Debug)]
struct SdkVersion {
    major: u32,
    minor: u32,
}

fn detect_sdk_version() -> Option<SdkVersion> {
    let output = Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-version"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let version = String::from_utf8_lossy(&output.stdout);
    let mut parts = version.trim().split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    Some(SdkVersion { major, minor })
}

const fn sdk_at_least(
    version: Option<SdkVersion>,
    required_major: u32,
    required_minor: u32,
) -> bool {
    match version {
        Some(version) if version.major > required_major => true,
        Some(version) if version.major == required_major => version.minor >= required_minor,
        _ => false,
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=swift-bridge");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=SDKROOT");

    if env::var("DOCS_RS").is_ok() {
        return;
    }

    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=CoreVideo");
    println!("cargo:rustc-link-lib=framework=Metal");

    let swift_dir = "swift-bridge";
    let out_dir = env::var("OUT_DIR").unwrap();
    let swift_build_dir = format!("{out_dir}/swift-build");

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let swift_triple = match target_arch.as_str() {
        "x86_64" => "x86_64-apple-macosx",
        "aarch64" => "arm64-apple-macosx",
        other => panic!("coreml: unsupported target arch '{other}'"),
    };

    let sdk_version = detect_sdk_version();
    let mut swift_args: Vec<String> = vec![
        "build".into(),
        "-c".into(),
        "release".into(),
        "--triple".into(),
        swift_triple.into(),
        "--package-path".into(),
        swift_dir.into(),
        "--scratch-path".into(),
        swift_build_dir.clone(),
    ];

    if sdk_at_least(sdk_version, 14, 0) {
        swift_args.push("-Xswiftc".into());
        swift_args.push("-DCOREML_HAS_MACOS14_SDK".into());
    }
    if sdk_at_least(sdk_version, 14, 4) {
        swift_args.push("-Xswiftc".into());
        swift_args.push("-DCOREML_HAS_MACOS14_4_SDK".into());
    }
    if sdk_at_least(sdk_version, 15, 0) {
        swift_args.push("-Xswiftc".into());
        swift_args.push("-DCOREML_HAS_MACOS15_SDK".into());
    }

    let mut swift_build = Command::new("swift");
    swift_build.args(&swift_args).args(["--jobs", "4"]);
    let output = swift_build.output().expect("Failed to build Swift bridge");

    assert!(
        output.status.success(),
        "Swift build failed ({}):\n{}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let output = swift_build
        .arg("--show-bin-path")
        .output()
        .expect("failed to query the Swift product path");
    assert!(
        output.status.success(),
        "Swift product path query failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let swift_bin_dir = String::from_utf8(output.stdout).expect("Swift product path must be UTF-8");
    let swift_bin_dir = swift_bin_dir.trim();
    assert!(!swift_bin_dir.is_empty(), "Swift product path is empty");

    println!("cargo:rustc-link-search=native={swift_bin_dir}");
    println!("cargo:rustc-link-lib=static=CoreMLBridge");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");

    if let Ok(output) = Command::new("xcode-select").arg("-p").output() {
        if output.status.success() {
            let xcode_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let swift_lib_path =
                format!("{xcode_path}/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{swift_lib_path}");
        }
    }
}
