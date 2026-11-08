//! Build the pinned, patched vendored core, never an upstream stock SDK.
mod build_support;
use anyhow::{bail, Context, Result};
use build_support::{artifact_path, digest, verify_source, Provenance, TargetArtifacts};
use sha2::Digest;
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
};

fn source_root(manifest: &Path, output: &Path, provenance: &Provenance) -> Result<PathBuf> {
    if let Some(override_path) = env::var_os("MOONSHINE_DIR") {
        let root = fs::canonicalize(override_path)
            .context("Explicit MOONSHINE_DIR is invalid; no fallback attempted")?;
        let root = if root.join("core/moonshine-c-api.h").is_file() {
            root
        } else if root.join("moonshine-c-api.h").is_file() {
            root.parent().context("invalid source root")?.to_path_buf()
        } else {
            bail!("Explicit MOONSHINE_DIR has no source header; no fallback attempted")
        };
        verify_source(&root, provenance)?;
        println!("cargo:rerun-if-changed={}", root.join("core").display());
        return Ok(root);
    }
    let archive = manifest.join("core-source.tar.gz");
    if digest(&archive)? != provenance.source_sha256 {
        bail!("Vendored Moonshine source checksum mismatch");
    }
    let directory = output.join(format!("source-{}", &provenance.source_sha256[..16]));
    let root = directory.join("moonshine");
    if !directory.exists() {
        fs::create_dir(&directory)?;
        let decoder = flate2::read::GzDecoder::new(fs::File::open(archive)?);
        tar::Archive::new(decoder).unpack(&directory)?;
    }
    verify_source(&root, provenance)?;
    Ok(root)
}

fn configure_core(root: &Path, output: &Path, ort: &Path, static_framework: bool) -> PathBuf {
    let mut config = cmake::Config::new(root.join("core"));
    let source_id = format!(
        "{:x}",
        sha2::Sha256::digest(root.to_string_lossy().as_bytes())
    );
    config
        .out_dir(output.join(format!("native-{}", &source_id[..16])))
        .profile("Release")
        .define("ONNXRUNTIME_LIB_PATH", ort)
        .define(
            "MOONSHINE_BUILD_SHARED",
            if static_framework { "OFF" } else { "ON" },
        )
        .define(
            "MOONSHINE_BUILD_SWIFT",
            if static_framework { "ON" } else { "OFF" },
        )
        .define("CMAKE_CXX_STANDARD", "20")
        .build_target("moonshine");
    config.build().join("build")
}

fn shared_runtime(
    build: &Path,
    output: &Path,
    dependencies: &[PathBuf],
    target_os: &str,
) -> Result<PathBuf> {
    let runtime = output.join("runtime");
    fs::create_dir_all(&runtime)?;
    let names: &[&str] = match target_os {
        "linux" => &["libmoonshine.so"],
        "macos" => &["libmoonshine.dylib"],
        "windows" => &["moonshine.dll", "moonshine.lib"],
        _ => bail!("Unsupported shared Moonshine platform"),
    };
    for name in names {
        let candidate = [build.join(name), build.join("Release").join(name)]
            .into_iter()
            .find(|p| p.is_file())
            .with_context(|| format!("Missing built runtime {name}"))?;
        fs::copy(candidate, runtime.join(name))?;
    }
    for dependency in dependencies {
        if dependency.extension().is_some_and(|ext| ext == "lib") {
            continue;
        }
        fs::copy(
            dependency,
            runtime.join(dependency.file_name().context("invalid dependency name")?),
        )?;
    }
    Ok(runtime)
}

fn link_dependency(output: &Path, dependencies: &[PathBuf], target_os: &str) -> Result<PathBuf> {
    // Windows' upstream post-build copy expects the import library and DLL
    // together, while the verified cache stores each under its own hash.
    if target_os != "windows" {
        return Ok(dependencies[0].clone());
    }
    let sdk = output.join("ort-sdk");
    fs::create_dir_all(&sdk)?;
    for dependency in dependencies {
        fs::copy(
            dependency,
            sdk.join(dependency.file_name().context("dependency filename")?),
        )?;
    }
    Ok(sdk.join(dependencies[0].file_name().context("ORT filename")?))
}

fn build_and_link(
    manifest: &Path,
    output: &Path,
    source: &Path,
    provenance: &Provenance,
) -> Result<()> {
    let target = env::var("TARGET")?;
    let targets: BTreeMap<String, TargetArtifacts> =
        serde_json::from_slice(&fs::read(manifest.join("ort-artifacts.json"))?)?;
    let target_spec = targets.get(&target).with_context(|| {
        format!("No pinned Moonshine runtime dependencies for {target}; no fallback attempted")
    })?;
    let dependencies: Vec<_> = target_spec
        .files
        .iter()
        .map(|item| {
            artifact_path(
                &build_support::cache_root()?,
                item,
                build_support::offline(),
            )
        })
        .collect::<Result<_>>()?;
    let target_os = env::var("CARGO_CFG_TARGET_OS")?;
    let ort = link_dependency(output, &dependencies, &target_os)?;
    let static_framework = target_os == "macos" && target_spec.mode == "static";
    let build = configure_core(&source, &output, &ort, static_framework);
    if static_framework {
        println!("cargo:rustc-link-search=framework={}", build.display());
        println!("cargo:rustc-link-lib=framework=moonshine");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=Foundation");
    } else {
        let runtime = shared_runtime(&build, &output, &dependencies, &target_os)?;
        println!("cargo:rustc-link-search=native={}", runtime.display());
        println!("cargo:rustc-link-lib=dylib=moonshine");
    }
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=dylib=c++");
    }
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=pthread");
    }
    println!(
        "cargo:warning=Using vendored Moonshine {} (final-audio fix)",
        provenance.version
    );
    Ok(())
}

fn run() -> Result<()> {
    if env::var_os("MOONSHINE_VERSION").is_some() {
        bail!("MOONSHINE_VERSION cannot override the pinned vendored runtime; unset it");
    }
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").context("manifest")?);
    let output = PathBuf::from(env::var_os("OUT_DIR").context("output")?);
    for name in [
        "MOONSHINE_VERSION",
        "MOONSHINE_DIR",
        "MOONSHINE_ORT_CACHE",
        "MOONSHINE_OFFLINE",
        "CARGO_NET_OFFLINE",
        "HF_HUB_OFFLINE",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    for name in [
        "core-source.tar.gz",
        "provenance.json",
        "ort-artifacts.json",
        "build_support.rs",
    ] {
        println!("cargo:rerun-if-changed={name}");
    }
    let provenance: Provenance =
        serde_json::from_slice(&fs::read(manifest.join("provenance.json"))?)?;
    let source = source_root(&manifest, &output, &provenance)?;
    let core = source.join("core");
    if env::var_os("DOCS_RS").is_none() {
        build_and_link(&manifest, &output, &source, &provenance)?;
    }
    bindgen::Builder::default()
        .header(core.join("moonshine-c-api.h").to_string_lossy())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .context("Moonshine bindings")?
        .write_to_file(output.join("bindings.rs"))?;
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        panic!("Vendored Moonshine build failed (no stock fallback): {error:#}");
    }
}
