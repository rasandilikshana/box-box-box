---
name: run-tests
description: Run and interpret the 100 F1 race simulation test cases. Use when asked to test the solution, check accuracy, or debug failing cases.
allowed tools: Bash, Read, Grep
---

# Running & Interpreting Tests

## Commands

### Run all 100 tests
```bash
./test_runner.sh
```
Outputs pass/fail per test and a final score (e.g., `87/100`).

### Run a single test
```bash
python solution/race_simulator.py < data/test_cases/inputs/test_001.json
```
Compare against: `data/test_cases/expected_outputs/test_001.json`

### Debug a specific failing test
```bash
python solution/race_simulator.py < data/test_cases/inputs/test_042.json > /tmp/my_out.json
diff /tmp/my_out.json data/test_cases/expected_outputs/test_042.json
```

## Interpreting Results

| Score | Likely Issue | See |
|-------|-------------|-----|
| < 20/100 | Wrong formula structure | `solution/race_simulator.pseudocode` |
| 20–60/100 | Wrong compound offsets or degradation scale | `docs/faq.md` |
| 60–80/100 | track_temp coefficient off | `docs/faq.md` |
| 80–99/100 | Edge cases: pit lap timing, tire age reset | `docs/regulations.md` |
| 100/100 | Done! | Submit |

## Scoring Rule
Entire `finishing_positions` array must match exactly — no partial credit per test.

## Solution Entrypoint
Controlled by `solution/run_command.txt`. Change to switch between Python/JS/Java:
```
python solution/race_simulator.py      # Python
node solution/race_simulator.js        # JavaScript
java -cp solution Main                 # Java
```
