# Oqtopus shared runtime snapshot

`oqtopus-core` and `oqtopus-render` are a source snapshot of the shared
`oqto-animation` repository at commit `5a98651`. They contain code only, not
animation packs or artwork. They are vendored so eaRS builds from a standalone
checkout before the shared runtime is published as a versioned dependency.
Make runtime changes upstream first, then refresh both directories from the
same upstream revision and run both projects' tests.
