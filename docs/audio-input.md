# Microphone input conversion

Capture uses one stream-owned converter (`src/audio_input.rs`) to produce mono
24 kHz PCM for both recognition and retained recordings. It supports F32/I16/U16
at 8–192 kHz, averages arbitrary interleaved channels, preserves partial channel
frames, and uses the shared windowed-sinc resampler in bounded-ratio stages.
Fixed internal quanta keep output independent of callback partitioning. Native
24 kHz bypasses filtering; other rates retain at most 128 target samples (<6 ms)
of lookahead/buffering at stream shutdown, not at every utterance boundary.

The former callback-local linear interpolator lost samples at noninteger ratios,
aliased high frequencies, mishandled more than two channels, and could panic on
split stereo frames. Nine synthetic tests check rate/phase/partition invariance,
channel/sample formats, bounded lag, passband preservation and alias rejection:

```bash
cargo test --locked --lib audio::tests
```

This is not a gain, denoising or normalization feature. Quiet input remains quiet;
equal channel mixing is not speaker-layout-aware. Native microphone quality,
realtime scheduling and device selection require separate live verification.
Capture-disconnect shutdown/logging remains tracked separately in `trx-ntjy`.

## Choosing a microphone

Settings → Dictation → Input → **Microphone** offers **System default**, named
input devices, and **Refresh**. The picker opens an inline list; Enter/Space opens
or chooses, arrows move, Escape closes, and R refreshes. Device enumeration is
metadata-only: it never builds/plays a capture stream. Choosing an item saves a
preference without starting or restarting microphone/wake activity.

The shared core configuration is:

```toml
[dictation]
input_device = "USB microphone"
```

Omit `input_device` for the system default. `ears-ui --input-devices` returns JSON
metadata without a GUI/capture loop; `ears --list-devices` retains the existing
human-readable CLI listing and legacy numeric capture selection is unchanged.

Names are matched exactly and must be unique. Selection is resolved afresh at
capture startup, so device-list reordering does not change the chosen microphone.
Renamed, disconnected, unnamed or duplicate-name devices require reselection;
there is no silent fallback to another input. This is a name-based preference,
not an immutable hardware ID, and does not change the OS-wide default device.

Switching is live. Picking an input while eaRS runs opens the new device muted
while the old one keeps capturing, then stops and joins the old stream before the
new one is heard, so dictation, the wake tee and recording never receive two
devices interleaved. A brief device gap (milliseconds) is not filled. If the new
device cannot be opened, the previous input keeps running and Settings says so.
The preference is saved only after the engine confirms the switch; when eaRS is
not running (e.g. first-run setup) the choice is saved for the next start. The
engine reports the device actually open (`In use: …`). This is the same engine
command for any front end: `DictationCommand::SetInputDevice(Option<String>)`,
answered by `DictationEvent::InputDevice { requested, active, error }`.

Each capture stream is owned by a stoppable `Capture` on its own thread: dropping
it stops the stream, so engine restarts no longer leave old capture threads
running. A stream that fails at runtime (device unplugged) reports an error
instead of exiting the process; a vanished receiver stops its stream once instead
of logging on every buffer (`trx-ntjy`). The legacy blocking CLI live mode keeps
its exit-on-stream-error behaviour.

The picker atomically edits only this key, retaining other values/comments,
permissions and intentional config symlinks. Save errors/conflicts stay visible;
a disappeared config is not silently recreated as an incomplete document.
Retention metadata identifies an explicitly named input rather than always
claiming the system default. One capture stream still feeds dictation and its
existing wake/recording tee.

Synthetic native fixtures (`--preview-settings`, `--preview-settings-mic-empty`,
`--preview-settings-mic-missing`) use fake device lists. Their selection/Refresh
never enumerates devices, writes preferences or starts an engine. They do not
prove physical microphone/TCC/hotplug or live handoff behavior.
