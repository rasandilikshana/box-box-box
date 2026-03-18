# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run all 100 test cases:**
```bash
./test_runner.sh
```

**Run a single test manually:**
```bash
python solution/race_simulator.py < data/test_cases/inputs/test_001.json
# or for JS: node solution/race_simulator.js < data/test_cases/inputs/test_001.json
```

**Change solution language/entrypoint:** Update `solution/run_command.txt` with the new command (e.g., `node solution/race_simulator.js`). The test runner reads this file to know how to execute the solution.

## Architecture

This is an F1 Pit Strategy Optimization Challenge — a reverse-engineering competition. The goal is to analyze 30,000 historical race records to discover the exact simulation mechanics, then build a predictor that passes 100 test cases.

### Input → Output Contract
- **Input (stdin):** JSON with `race_config` (track, total_laps, base_lap_time, pit_lane_time, track_temp) and `strategies` for 20 drivers (starting tire compound + pit stop schedule)
- **Output (stdout):** JSON with `finishing_positions` — array of 20 driver IDs ordered 1st to 20th by total race time
- Scoring: entire finishing order must match exactly (no partial credit per test case)

### Core Simulation Mechanics
The race is a pure time trial — no car-to-car interaction. Final position is determined by total time:

```
total_time = sum of all lap times + pit stop penalties
lap_time   = base_lap_time + compound_offset + degradation(tire_age, compound, track_temp)
pit stop   = adds pit_lane_time penalty on the lap the stop occurs
```

Key constraints to simulate correctly:
- **Lap-by-lap** (not aggregate) — tire age increments each lap
- **Tire age** resets to 1 after each pit stop; starts at 1 on lap 1
- **Three compounds:** SOFT (fastest, degrades quickly), MEDIUM (balanced), HARD (slowest, durable)
- **track_temp** affects degradation rate
- **Mandatory rule:** each driver must use ≥2 different tire compounds

### Data Layout
- `data/historical_races/races_XXXXX-XXXXX.json` — 30 files × 1,000 races each; each record has `race_config`, `strategies`, and `finishing_positions` (ground truth for reverse-engineering)
- `data/test_cases/inputs/test_NNN.json` — 100 test inputs (no finishing_positions)
- `data/test_cases/expected_outputs/test_NNN.json` — 100 expected outputs

### Discovering the Formula
The exact degradation model is not documented — it must be reverse-engineered from historical data. See `docs/regulations.md` for the qualitative rules and `docs/faq.md` for debugging guidance at different accuracy thresholds. The pseudocode skeleton at `solution/race_simulator.pseudocode` shows the expected simulation structure.
