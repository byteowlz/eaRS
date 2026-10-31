# Native Settings layout

This is an incumbent-theme refinement, not the proposed global Calm Look or the
unselected six-page/sidebar study. The four existing destinations remain intact.

- Equal-width, 32 px-high top tabs fill the available navigation row. Navigation
  stays in a fixed header while only pane content scrolls. Arrow keys and Home/End
  select panes when navigation is focused; Tab/Shift-Tab keep the existing focus cycle.
- The duplicate app heading and implementation-detail footer are removed. The
  native window title remains. Existing780×560 default and640×420 minimum are unchanged.
- Each pane retains its own scroll handle across selection changes. The last
  selected pane is remembered while the app runs. Existing field/profile draft
  entities are not reconstructed when changing tabs.
- A single300px control column defines both edges of form fields and segmented
  groups, not just their right edge. Labels take the remaining width at minimum,
  default and enlarged window sizes. Numeric inputs stay short at the column's
  leading edge; switches remain compact at its trailing edge.
- Choice cells share equal grid tracks within each group. Short codes/numeric
  options stay in compact rows; longer/dense choices use bounded rows rather than
  stretching the form. Native pixel rounding may distribute1px across tracks.
  Opaque long aliases ellipsize visually without changing their actual value.
- Server input and compact Apply button share that same300px compound-control
  width; the field flexes instead of adding a300px input beside a second button.
  Standalone actions remain compact, not full-width.
- Dictionary Add entry is a compact trailing action, with keyboard activation and
  an inactive appearance until both fields contain meaningful values. A failed
  save displays an error and retains the fields; a synthetic preview never writes
  a dictionary. Successful saves alone clear the fields.
- Overlay content (Phrase/Word/Ears only/Hidden), floating appearance (Classic/Black),
  placement and typography are independent. The preview visibility reason is real
  runtime state, not guessed silence or a fake VAD diagnostic.

Research: Design Studio `artifacts/ears/settings-layout-r1/research.md` cites Apple
HIG Settings, Tab views, Segmented controls and Layout; NN/g Tabs/Progressive
Disclosure and WAI Tabs. Those do not mandate full-window tabs universally. Equal
widths are a bounded refinement of the incumbent four-tab route, not selection of
the exploratory six-page/sidebar study or final visual acceptance.

Verification uses synthetic native Settings at default, minimum and light-mode
sizes, measured prepaint bounds and real click/keyboard input. The native probe
aborts if concurrent desktop input or focus changes interfere; it does not force
cursor/focus restoration over new input. Its `--layout-only` mode measures row,
segment and compound-action bounds without pointer or keyboard input. The synthetic
`--preview-settings-wide` case covers1120px in addition to640/780px; explicit fixture
aliases include a long label. No real server connection/config is used. These are not native
accessibility certification or pixel captures. Full per-frame VAD diagnostics,
pause-duration editing and new speaker-setup pages remain separate tracked work;
this change does not invent their controls or claim that broader study completed.
