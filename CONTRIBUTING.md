# Contributing

## Docs ownership (review checklist)

Imajin's user docs have one deliberate rule to keep them from contradicting each
other. When reviewing a PR that touches the docs, check:

- **`docs/analysis_capabilities.md` (the capabilities matrix) is the sole
  authority for the exhaustive set of supported combinations** — which analysis ×
  target × tool × statistics × graph combinations exist. Any change to *what is
  supported* updates the matrix first.
- **`docs/features.md` is a narrative reference**: it describes what each feature
  is for and how a user reaches it. It may describe a feature (e.g. "`compare_groups`
  supports Welch / Mann-Whitney / …"), but it must **not** present itself as the
  authoritative, exhaustive list of supported combinations — where a reader needs
  the full set, it links to the matrix.
- **`docs/getting_started.md` headings are stable anchors** for the in-app
  `get_help` tool. If you rename, add, or remove a section, update the `_TOPICS`
  map in `src/imajin/tools/help.py` (the section↔topic and anchor tests in
  `tests/test_tools_help.py` fail until you do). Keep headings ASCII and
  punctuation-light.

A new capability therefore lands in the matrix; `features.md` gets a sentence only
if the *narrative* changes.
