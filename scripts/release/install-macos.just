default: test

check:
    /bin/bash -n install-macos.sh

test: check
    bun test ./install-macos.test.ts
