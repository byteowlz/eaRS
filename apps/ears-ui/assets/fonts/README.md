# Bundled fonts

Regular faces bundled locally; no runtime downloads. The overlay faces are
unmodified. Native Settings also uses Michroma (lossless WOFF2-to-TTF container
conversion) and unmodified Geo from the shared hi-fi reference; these do not
change overlay typography.

| Font | Upstream | SHA-256 |
|---|---|---|
| Departure Mono OTF | https://departuremono.com/ | `4d53f663155cf8bf7ffc8e688776e719625f7bbb80a8d90073438b249261a2e0` |
| JetBrains Mono TTF | https://github.com/JetBrains/JetBrainsMono (fonts/ttf/JetBrainsMono-Regular.ttf) | `e6fd0d7e91550b3ed2b735d4312474362c4716edc4fc0577a0f61ed782d5aed1` |

| Michroma TTF | https://github.com/googlefonts/Michroma-font | `a2bcb620ff1df7d3ae9bafed1d4cd0f38e961941b0b6e81c6995f92666dcddaf` |
| Geo TTF | https://fonts.google.com/specimen/Geo | `8e3332479c52b22f52eb6924a3f31a919baf82df62851f85af5ead2c36810154` |

See the adjacent OFL files. These are selectable overlay faces, not a global UI
font change. Departure Mono was copied from an existing local upstream install;
its exact upstream revision is not asserted. JetBrains regular was fetched from
the upstream repository; the bundled bytes above, not the moving branch, pin it.
