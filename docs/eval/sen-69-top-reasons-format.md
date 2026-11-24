# SEN-69: `top_reasons` JSON format (proposal for Humayl)

**Status:** draft — needs Humayl's sign-off before implementation lands.
**Owner (format):** Nima. **Owner (sensor/API wiring):** Humayl.

## What this is

Every alert row in `detections.top_reasons` (JSONB, C2) stores one of these
objects. The same JSON is read in three places (TC-12 requires the API and
Grafana to show the same reasons):

1. `GET /alerts/{id}` (C4)
2. Grafana alert-detail panel (SEN-FR-23)
3. Discord alert line "Top reasons: ..." (C5) — just joins the labels

Keeping one shape for all three means no transformation in the API or
Grafana layer.

## Format

```json
{
  "model_version": "dev-lgbm-noport-v0",
  "method": "tree_shap",
  "class": "PortScan",
  "reasons": [
    {
      "feature": "src2dst_syn_packets",
      "label": "client SYN packets",
      "value": 42.0,
      "shap": 0.4123,
      "direction": "up"
    },
    {
      "feature": "bidirectional_duration_ms",
      "label": "flow duration",
      "value": 15.3,
      "shap": -0.3105,
      "direction": "down"
    }
    // exactly 5 reasons, largest |shap| first
  ]
}
```

### Field notes

- `model_version` — matches `detections.model_version`. Lets Grafana group
  explanations by model and the API include it in the response.
- `method` — `"tree_shap"` for LightGBM TreeSHAP. `null` when no
  explanation is available (see below). Keeps the door open for other
  methods later without a schema change.
- `class` — the predicted class the SHAP values are computed against, i.e.
  `detections.predicted_class`. Having it in the JSON means consumers
  don't need to join rows to interpret the sign.
- `reasons` — always exactly 5 entries when `method == "tree_shap"`,
  sorted by `|shap|` descending. The top-5 cutoff keeps the Discord line
  readable and the JSON small (<= ~500 bytes per row).
- `feature` — the raw NFStream feature name (e.g.
  `bidirectional_rst_packets`). Lets Grafana link to feature docs if we
  add them.
- `label` — short plain-English description from
  `sentinel_ml/feature_labels.py` (e.g. "many reset (RST) packets").
  Neutral wording; direction is carried separately by `direction`. This
  is what Discord joins.
- `value` — that flow's raw feature value (number). After preprocess but
  before any transform; i.e. the number a human would see in a Grafana
  tooltip.
- `shap` — raw SHAP value for that feature on the predicted class.
  Rounded to 6 d.p. Negative means the feature pushed probability DOWN
  for this class (even though the class was still the argmax).
- `direction` — `"up"` if `shap >= 0`, else `"down"`. Discord and
  Grafana use this for a small arrow or colour. Stored redundantly so
  consumers don't need to re-derive it.

### "No explanation available" (UC-2)

If the Explainer can't produce reasons for a row (e.g. the predicted
class isn't in the model's `classes_`, or a future non-TreeSHAP model
has no cheap explainer), store:

```json
{
  "model_version": "dev-lgbm-noport-v0",
  "method": null,
  "class": "PortScan",
  "reasons": [],
  "note": "no explanation available"
}
```

Same top-level keys, empty `reasons`, `method=null`, plus a `note` string
for the UI. The API and Grafana should fall back to a short placeholder
when `method` is null.

## Discord rendering

Discord's alert line in C5 is:
> Top reasons: client SYN packets, shortest gap between packets, smallest packet size

i.e. `", ".join(r["label"] for r in reasons[:5])`. When `reasons` is
empty, it drops the "Top reasons: ..." line entirely (UC-2).

## Why these choices (short version)

- **One shape, three consumers**: avoids per-consumer transforms and the
  sync problems TC-12 is checking for.
- **Store `label` AND `feature`**: `label` is what humans read; `feature`
  is what the schema joins / debugs on.
- **Store `direction`**: cheap, and saves every UI from re-deriving it.
- **Exactly 5 reasons**: SHAP values beyond 5 add little for scanning;
  keeps the JSON small and the Discord line short.

## Open questions for Humayl

1. Discord: drop the "Top reasons:" prefix entirely when `reasons == []`,
   or keep the prefix with "no explanation available"? I've proposed drop.
2. Is adding `label` inside each reason a problem for Grafana, or would
   you prefer a flat `labels: [...]` field alongside? I kept them nested
   so one row of iteration gives feature + label + value + shap.
3. The `value` field is the raw feature value *after* preprocess. For the
   dev stand-in that's identical to the input (preprocess only drops
   dst_port). For future models with scaling we'd need to decide: raw or
   scaled. I'd default to raw (human-readable). OK?

`sentinel_ml/explain.py` is already in this PR and produces the shape above;
I'll adjust it to track any format changes you want. The sensor-side wiring
is just:

```python
reasons = explainer.explain(flows_batch, detections["predicted_class"])
# write reasons[i] into detections.top_reasons for alert rows
```
