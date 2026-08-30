use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=NCCL_ROOT");

    let linux = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");
    let nvidia = env::var_os("CARGO_FEATURE_NVIDIA").is_some();
    let transcribe_cpp = env::var_os("CARGO_FEATURE_TRANSCRIBE_CPP").is_some();

    // transcribe.cpp's host decoder calls cblas_sgemv/gemm when CMake finds a
    // system BLAS, but upstream emits the BLAS library through
    // `cargo:rustc-link-arg` from the -sys build script, which never propagates
    // to downstream binaries — so the final link fails with
    // `undefined symbol: cblas_sgemm`. Emit the missing link directive here,
    // where it does propagate. If pkg-config can't find a C BLAS, the decoder
    // may have been configured with a scalar fallback (TRANSCRIBE_USE_SYSTEM_BLAS
    // =OFF), so absence is only a warning, not an error.
    if linux && transcribe_cpp {
        match pkg_config::Config::new().atleast_version("3").probe("cblas") {
            Ok(_) => println!("cargo:warning=Linking system cblas required by transcribe.cpp host decoder"),
            Err(err) => println!(
                "cargo:warning=cblas not found through pkg-config ({err}); assuming transcribe.cpp used a BLAS-free build or the linker resolves cblas_sgemm elsewhere"
            ),
        }
    }

    if !(linux && nvidia && transcribe_cpp) {
        return;
    }

    // ggml enables NCCL whenever CMake discovers it. Its static ggml-cuda
    // archive records NCCL as a PRIVATE dependency, so transcribe.cpp's Rust
    // link manifest cannot propagate -lnccl to the final Cargo binary. Probe
    // the same system library here and emit the missing final-link metadata.
    // If NCCL is absent, ggml compiles without GGML_USE_NCCL and no link is
    // required (the common single-GPU setup remains dependency-free).
    match pkg_config::Config::new().probe("nccl") {
        Ok(_) => println!("cargo:warning=Linking system NCCL required by static ggml-cuda"),
        Err(err) => println!(
            "cargo:warning=NCCL not found through pkg-config ({err}); assuming ggml-cuda was built without NCCL"
        ),
    }
}
