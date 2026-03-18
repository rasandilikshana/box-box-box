#!/usr/bin/env python3
"""
Investigate: when drivers have identical strategies, what determines their finishing order?
This reveals the tiebreaker mechanic (or per-driver modifier).
"""
import json, re
from collections import defaultdict

DATA = '/home/ubuntu/box-box-box/data/historical_races/races_00000-00999.json'
with open(DATA) as f:
    races = json.load(f)

# Find race R05618
target = None
for r in races:
    if r['race_id'] == 'R05618':
        target = r
        break

if not target:
    # Search all files
    import glob
    for fname in sorted(glob.glob('/home/ubuntu/box-box-box/data/historical_races/*.json'))[:10]:
        with open(fname) as f:
            rs = json.load(f)
        for r in rs:
            if r['race_id'] == 'R05618':
                target = r
                break
        if target:
            break

if not target:
    print("Race R05618 not found in first 10 files")
else:
    rc = target['race_config']
    print(f"Race R05618: {rc['track']}, temp={rc['track_temp']}, laps={rc['total_laps']}")
    print(f"Finishing order: {target['finishing_positions']}")
    print()

    # Print all strategies
    print("All strategies:")
    finish_pos = {d: i+1 for i, d in enumerate(target['finishing_positions'])}
    for pk, s in sorted(target['strategies'].items()):
        did = s['driver_id']
        pits = [(p['lap'], p['to_tire']) for p in s['pit_stops']]
        grid_num = int(pk.replace('pos',''))
        print(f"  {pk:5s} (grid={grid_num:2d}) {did} → finish={finish_pos[did]:2d}: {s['starting_tire']:6s} pits={pits}")

print()
print("=" * 60)
print("ANALYSIS: Same-strategy pairs")
print("For each pair, note: grid position of each driver")
print("=" * 60)

# Find all same-strategy pairs in race R05618
if target:
    strats = list(target['strategies'].items())  # [(pos_key, strategy), ...]

    # Build lookup: driver_id -> grid position
    driver_to_grid = {}
    for pk, s in target['strategies'].items():
        grid_num = int(pk.replace('pos', ''))
        driver_to_grid[s['driver_id']] = grid_num

    finish_order = {d: i for i, d in enumerate(target['finishing_positions'])}

    def strat_key(s):
        return (s['starting_tire'],
                tuple(sorted([(p['lap'], p['from_tire'], p['to_tire']) for p in s['pit_stops']])))

    # Group by strategy
    by_strat = defaultdict(list)
    for pk, s in target['strategies'].items():
        by_strat[strat_key(s)].append(s['driver_id'])

    for sk, drivers in by_strat.items():
        if len(drivers) > 1:
            print(f"\nStrategy {sk[0]} → {[p[2] for p in sk[1]]} at laps {[p[0] for p in sk[1]]}:")
            for did in sorted(drivers, key=lambda d: finish_order[d]):
                grid = driver_to_grid[did]
                d_num = int(did.replace('D', ''))
                print(f"    {did} (D_num={d_num:3d}, grid={grid:2d}) → finish pos {finish_order[did]+1}")

print()
print("HYPOTHESIS:")
print("1. Lower driver ID number finishes first?")
print("2. Lower grid position (pos1 > pos20) finishes first?")
print("3. Higher grid position finishes first?")

# General analysis across many races
print("\n" + "=" * 60)
print("GENERAL ANALYSIS: 200 races")
print("=" * 60)

results = []  # (did_A_num, did_B_num, grid_A, grid_B, A_finishes_first)

for race in races[:200]:
    strats = list(race['strategies'].values())
    finish_order = {d: i for i, d in enumerate(race['finishing_positions'])}
    driver_to_grid = {}
    for pk, s in race['strategies'].items():
        driver_to_grid[s['driver_id']] = int(pk.replace('pos', ''))

    by_strat = defaultdict(list)
    for s in strats:
        key = strat_key(s)
        by_strat[key].append(s['driver_id'])

    for sk, drivers in by_strat.items():
        if len(drivers) == 2:
            da, db = drivers[0], drivers[1]
            if finish_order[da] < finish_order[db]:
                winner, loser = da, db
            else:
                winner, loser = db, da
            w_num = int(winner.replace('D', ''))
            l_num = int(loser.replace('D', ''))
            w_grid = driver_to_grid[winner]
            l_grid = driver_to_grid[loser]
            results.append({
                'winner': winner, 'loser': loser,
                'w_num': w_num, 'l_num': l_num,
                'w_grid': w_grid, 'l_grid': l_grid,
                'lower_id_wins': w_num < l_num,
                'lower_grid_wins': w_grid < l_grid,
            })

if results:
    lower_id_correct = sum(1 for r in results if r['lower_id_wins'])
    lower_grid_correct = sum(1 for r in results if r['lower_grid_wins'])
    n = len(results)
    print(f"\nFound {n} same-strategy pairs across 200 races")
    print(f"Lower driver ID wins: {lower_id_correct}/{n} = {lower_id_correct/n*100:.1f}%")
    print(f"Lower grid position wins: {lower_grid_correct}/{n} = {lower_grid_correct/n*100:.1f}%")
    print(f"Higher grid position wins: {n-lower_grid_correct}/{n} = {(n-lower_grid_correct)/n*100:.1f}%")

    # Show some examples
    print("\nSome examples:")
    for r in results[:10]:
        print(f"  Winner={r['winner']}(id={r['w_num']}, grid={r['w_grid']}) vs Loser={r['loser']}(id={r['l_num']}, grid={r['l_grid']})")
        print(f"    lower_id_wins={r['lower_id_wins']}, lower_grid_wins={r['lower_grid_wins']}")
