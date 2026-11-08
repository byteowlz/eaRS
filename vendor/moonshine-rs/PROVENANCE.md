# Moonshine Rust wrapper

Unmodified Rust source from crates.io `moonshine-rs 0.2.6`, package SHA-256
`b5757aa28f590918f9418661c9c9d7c72099fe332d0df3da9dd6ae16a9307e0a`.
Original MIT/Apache-2.0 licences and README are retained.

Only the sys dependency in `Cargo.toml` is changed to `../moonshine-sys`.
This direct path matters: a downstream Git/path consumer of eaRS does not
inherit eaRS's root `[patch.crates-io]`. Both CLI builds and library consumers
must therefore select the repaired native builder without a root patch trick.
No audio decoder feature, model-selection behavior or Rust API was changed.
