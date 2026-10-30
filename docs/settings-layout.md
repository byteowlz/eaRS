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
- Form labels align at the top of their controls. Normal text fields use the
  existing300 px width; numeric inputs stay short. The dictionary test no longer
  reserves460 px and squeezes its label at the minimum window size.
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
widths are the owner's accepted bounded alternative for these four categories.

Verification uses synthetic native Settings at default, minimum and light-mode
sizes, measured prepaint bounds and real click/keyboard input. These are not native
accessibility certification or pixel captures. Full per-frame VAD diagnostics,
pause-duration editing and new speaker-setup pages remain separate tracked work;
this change does not invent their controls or claim that broader study completed.
