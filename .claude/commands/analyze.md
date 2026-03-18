# /analyze

Reverse-engineer the tire degradation formula from historical race data.

Steps:
1. Load a sample of races from `data/historical_races/`
2. For each race, simulate all 20 drivers using current formula and compare predicted order to `finishing_positions`
3. Identify which parameter (compound_offset, degradation_scale, temp_coefficient) causes the most errors
4. Propose updated parameter values and test them
5. Write best-fit parameters to `analysis/lp_params.json`
6. Update `solution/race_simulator.py` with the new parameters
7. Run `/test` to measure improvement
