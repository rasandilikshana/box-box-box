---
name: analyze-formula
description: Reverse-engineer the F1 tire degradation formula from historical race data. Use when asked to discover, investigate, or fit the degradation model, compound offsets, or any simulation parameter.
allowed tools: Read, Bash, Grep, Glob, Write
---

# F1 Degradation Formula Discovery

## Goal
Find the exact parameters for:
```
lap_time = base_lap_time + compound_offset[compound] + degradation(tire_age, compound, track_temp)
```

## Data Sources
- Historical races: `data/historical_races/races_XXXXX-XXXXX.json` (30 files × 1,000 races)
- Analysis scripts: `analysis/` directory
- Known mechanics: `docs/regulations.md`, `docs/faq.md`
- Pseudocode skeleton: `solution/race_simulator.pseudocode`

## Workflow

### 1. Load a sample race
```python
import json
with open('data/historical_races/races_00001-01000.json') as f:
    races = json.load(f)
race = races[0]
# Keys: race_config, strategies, finishing_positions
```

### 2. Compute ground-truth total times
For each driver, simulate their strategy and compare computed time ranking vs `finishing_positions`.

### 3. Isolate degradation signal
- Use races where only compound differs between two drivers (same pit laps)
- Hold `track_temp` constant to find compound offsets first
- Then vary `track_temp` to find temperature coefficient

### 4. Known formula structure (from `docs/regulations.md`)
- SOFT: fastest base, highest degradation
- MEDIUM: balanced
- HARD: slowest base, lowest degradation
- Degradation is a function of `tire_age` (resets to 1 after pit, starts at 1 on lap 1)
- `track_temp` scales the degradation rate

### 5. Validate
Run `./test_runner.sh` — target is 100/100 exact matches.
Check `docs/faq.md` for hints at different accuracy thresholds (e.g., 60/100, 80/100).

## Output
Write discovered parameters to `analysis/lp_params.json` and implement in `solution/race_simulator.py`.
