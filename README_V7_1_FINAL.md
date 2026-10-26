
# Shadow Protocol RL V7.1 — Final Stabilization Pass

This patch is based on the first full V7 evaluation.

Observed V7 problems:

- SWAT doctrine: 91.67%.
- `utility_smoke`: 0%; policy always selected `USE_FLASH`.
- confirmed visual: 40/40 escapes.
- assault chose `PURSUE` only ~5.5% in that scenario.
- arrest: 40/40 truncations.
- radio updates exploded in the arrest scenario.
- normal guard doctrine was excellent, but full-world phase separation still
  showed some distribution shift.

## Root causes fixed

### 1. Flash and smoke were observationally too similar

The doctrine states are now genuinely distinguishable:

- flash = fresh / local confirmed contact;
- smoke = stale reported contact + corridor/approach exposure.

The observation/action dimensions remain **64 / 14**, so the V7 model can be
loaded and fine-tuned.

### 2. Fresh contact did not outrank routine clearing

For ASSAULT, recent confirmed contact now strongly rewards `PURSUE` and
penalizes continuing routine `CHECK_CORNER/CLEAR_SECTOR/WATCH_ENTRY`.

The neural network still selects the action; this is reward hierarchy, not an
FSM override.

### 3. Arrest was only possible when already standing next to the player

`ARREST` is now a tactical intention:

- if compliant contact is legitimately known by local vision or radio,
- LEADER approaches the reported contact,
- cuff/capture completes only at arrest distance and with another live unit
  available to cover.

### 4. Team compliance was not visible from the radio blackboard

The existing compliance observation slot now means "legitimately known
compliance", including a teammate radio report.

No observation dimension was added.

### 5. Radio spam

Visual contact still refreshes the team blackboard every sighting, but radio
update counters only advance for materially new information.

### 6. Confirmed noncompliant contact had no stopping consequence

The deterministic combat executor now tracks player health and wound slowdown.
A moving, confirmed intruder can be neutralized by sustained legitimate LOS
engagement.

The policy still selects tactical intent; it does not learn individual shots.

### 7. No-contact curriculum rewarded omniscient racing

Before legitimate discovery, the training intruder moves slowly/stealthily
inside the facility instead of immediately sprinting directly to the exit.

After confirmed contact it evades normally.

### 8. Normal guard final rehearsal

A `BalancedGuardEnv` oversamples the still-confused real-world transitions:

- routine,
- repeat_sound,
- visual,
- lost_contact/search,
- theft.

It keeps the V6 observation/action interface at **43 / 16**.

## Files

Replace:

```text
shadow_ai\swat.py
shadow_ai\swat_env.py
shadow_ai\swat_pretrain_env.py
shadow_ai\__init__.py
pytest.ini
```

Add:

```text
shadow_ai\balanced_guard_env.py
train_v7_1_guard_stable.py
train_v7_1_swat_stable.py
train_v7_1_final.py
evaluate_v7_1_stable.py
tests\test_v7_1_stability.py
README_V7_1_FINAL.md
```

## Test

```powershell
python -m pytest -q
```

## Final fine-tune

This starts from the already-trained models. It does NOT retrain from zero.

```powershell
python train_v7_1_final.py --guard-init models_v6_3_2\best_stable\best_model.zip --swat-init models_v7_swat\swat_shared_policy_v7.zip
```

Outputs:

```text
models_v7_1_guard\guard_v7_1_final.zip
models_v7_1_swat\swat_v7_1_final.zip
```

## SWAT stability evaluation

```powershell
python evaluate_v7_1_stable.py models_v7_1_swat\swat_v7_1_final.zip --episodes 40 --doctrine-episodes 100
```

The evaluator ends with an explicit V7.1 stability gate.

## Guard evaluation

Use the existing V6.3.2 evaluator:

```powershell
python evaluate_v6_3_2.py models_v7_1_guard\guard_v7_1_final.zip --episodes 50 --doctrine-episodes 100
```

Target for the final freeze:

- clean doctrine >= 98%;
- robust doctrine >= 98%;
- routine PATROL >= 95%;
- arrest ARREST >= 90%;
- visual reported PURSUE clearly dominant;
- visual unreported RADIO_REPORT clearly dominant;
- search SEARCH_LAST_SEEN dominant;
- theft post-call protection dominant.
