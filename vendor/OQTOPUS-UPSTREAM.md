# Shared motion runtime snapshot

`motion-core`, `motion-render`, `oqtopus-core` and `oqtopus-render` are a
source snapshot of `oqto-animation` commit `ee45429`. The bundled manifests
specify procedural Ears geometry and eaRS's provisional Oqtopus state mapping;
no Oqtopus animation pack or artwork is included. The snapshot keeps eaRS
buildable standalone while the shared runtime remains unpublished. Change the
runtime upstream first, refresh all four directories together, and run both
projects' tests.
