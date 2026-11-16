# SEN-106: Train–Test Twins Check

### Results

- **With `dst_port` (56 features)**:
  - **Exact twins in train**: `309,400 / 630,181` (**49.1%**) — leaves `320,781` strictly unseen test rows.
  - **Binary LogReg Macro-F1**: `0.9643` (all test) -> **`0.9492`** (without twins, `-0.0151`). Attack recall goes from `95.2%` -> `95.8%`, benign FPR from `1.4%` -> `2.5%`.
  - **Multiclass LogReg Macro-F1**: `0.6245` -> **`0.5787`** (`-0.0458`).

- **Without `dst_port` (55 features)**:
  - **Exact twins in train**: `361,832 / 630,181` (**57.4%**) — leaves `268,349` strictly unseen test rows.
  - **Binary LogReg Macro-F1**: `0.9653` (all test) -> **`0.9547`** (without twins, `-0.0106`). Attack recall goes from `95.2%` -> `97.0%`, benign FPR from `1.4%` -> `2.7%`.
  - **Multiclass LogReg Macro-F1**: `0.5858` -> **`0.5628`** (`-0.0231`).
