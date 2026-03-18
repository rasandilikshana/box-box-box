#!/usr/bin/env python3
"""
Deep analysis to understand the formula structure.
Key questions:
1. Does temperature change finishing ORDER (or just scale times)?
2. What is the right degradation scale?
3. Manual calculation of race 0 with large deg values
"""
import json, numpy as np
from collections import defaultdict

DATA = '/home/ubuntu/box-box-box/data/historical_races/races_00000-00999.json'

with open(DATA) as f:
    races = json.load(f)

# ============================================================
# Q1: Does temperature change ORDER?
# Compare races with identical strategies, different temps
# ============================================================
print("=== Q1: Does temperature change ORDER? ===")

# Build signature for a race's strategies (to find duplicate strategies)
def strat_signature(race):
    """Canonical representation of all 20 driver strategies."""
    strats = sorted([(s['driver_id'], s['starting_tire'],
                      tuple(sorted([(p['lap'], p['from_tire'], p['to_tire'])
                                    for p in s['pit_stops']])))
                     for s in race['strategies'].values()])
    return tuple(strats)

# Group races by strategy signature
print("Grouping races by strategy signatures...")
sig_to_races = defaultdict(list)
for r in races:
    sig = strat_signature(r)
    sig_to_races[sig].append(r)

# Find sigs with multiple races (same strategies, possibly different temps)
multi = [(sig, rs) for sig, rs in sig_to_races.items() if len(rs) > 1]
print(f"Found {len(multi)} strategy groups with multiple races")

for sig, rs in multi[:5]:
    temps = [r['race_config']['track_temp'] for r in rs]
    finishes = [r['finishing_positions'] for r in rs]
    same_finish = all(f == finishes[0] for f in finishes)
    diff_temps = len(set(temps)) > 1
    if diff_temps:
        print(f"\n  Same strategies, temps={temps}")
        for r in rs[:3]:
            print(f"    temp={r['race_config']['track_temp']}: {r['finishing_positions'][:5]}...")
        if same_finish:
            print("    ✓ Same finishing order despite different temps")
        else:
            print("    ✗ DIFFERENT finishing orders!")

# ============================================================
# Q2: For race 0, what params give correct ordering?
# ============================================================
print("\n=== Q2: Manual race 0 analysis ===")

r0 = races[0]
rc = r0['race_config']
print(f"Race 0: base={rc['base_lap_time']}, pit={rc['pit_lane_time']}, temp={rc['track_temp']}, laps={rc['total_laps']}")
print(f"Actual order: {r0['finishing_positions']}")

def sim_all(race, off_S, off_H, deg_S, deg_M, deg_H, formula='l'):
    rc = race['race_config']
    total_laps = rc['total_laps']
    base = rc['base_lap_time']
    pit_t = rc['pit_lane_time']
    temp = rc['track_temp']
    off = {'SOFT': off_S, 'MEDIUM': 0.0, 'HARD': off_H}
    deg = {'SOFT': deg_S, 'MEDIUM': deg_M, 'HARD': deg_H}
    times = {}
    for s in race['strategies'].values():
        did = s['driver_id']
        pm = {p['lap']: p['to_tire'] for p in s['pit_stops']}
        tire = s['starting_tire']
        age = 0
        t = 0.0
        for lap in range(1, total_laps+1):
            age += 1
            if formula == 'l':
                d = deg[tire] * age
            elif formula == 'lt':
                d = deg[tire] * age * temp
            elif formula == 'tn':
                d = deg[tire] * age * (temp/30.0)
            t += base + off[tire] + d
            if lap in pm:
                t += pit_t
                tire = pm[lap]
                age = 0
        times[did] = t
    return times

# Test with large deg values on race 0
print("\n--- Testing with large deg_S values ---")
for ds in [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6]:
    for dh in [0.01, 0.02, 0.03]:
        for dm in [0.03, 0.05, 0.07]:
            for so in [-1.0, -1.5, -2.0]:
                for ho in [0.5, 1.0, 1.5]:
                    times = sim_all(r0, so, ho, ds, dm, dh, 'l')
                    pred = sorted(times, key=lambda d: times[d])
                    if pred == r0['finishing_positions']:
                        print(f"  EXACT MATCH: off=[{so},0,{ho}] deg=[{ds},{dm},{dh}] formula=l")
                        break

# Also try formula with temp
print("\n--- Testing lt formula with wider range ---")
for ds in [0.005, 0.008, 0.01, 0.015, 0.02, 0.025]:
    for dh in [0.001, 0.002, 0.003]:
        for dm in [0.002, 0.003, 0.005]:
            for so in [-1.0, -1.5, -2.0]:
                for ho in [0.5, 1.0, 1.5]:
                    times = sim_all(r0, so, ho, ds, dm, dh, 'lt')
                    pred = sorted(times, key=lambda d: times[d])
                    if pred == r0['finishing_positions']:
                        print(f"  EXACT MATCH: off=[{so},0,{ho}] deg=[{ds},{dm},{dh}] formula=lt")

# ============================================================
# Q3: Does the formula involve tire_age at all?
# Try formula with NO degradation - just compound offsets
# ============================================================
print("\n=== Q3: Test pure compound offset (no degradation) ===")

def sim_nodeg(race, off_S, off_H):
    rc = race['race_config']
    total_laps = rc['total_laps']
    base = rc['base_lap_time']
    pit_t = rc['pit_lane_time']
    off = {'SOFT': off_S, 'MEDIUM': 0.0, 'HARD': off_H}
    times = {}
    for s in race['strategies'].values():
        did = s['driver_id']
        pm = {p['lap']: p['to_tire'] for p in s['pit_stops']}
        tire = s['starting_tire']
        t = 0.0
        for lap in range(1, total_laps+1):
            t += base + off[tire]
            if lap in pm:
                t += pit_t
                tire = pm[lap]
        times[did] = t
    return times

best_nd = 0
for so in np.arange(-3.0, 0.1, 0.1):
    for ho in np.arange(-0.1, 3.1, 0.1):
        ok = 0
        for r in races[:100]:
            times = sim_nodeg(r, so, ho)
            pred = sorted(times, key=lambda d: times[d])
            if pred == r['finishing_positions']:
                ok += 1
        if ok > best_nd:
            best_nd = ok
            print(f"  No-deg best: {ok}/100 | off_S={so:.1f} off_H={ho:.1f}")
print(f"No-degradation max accuracy: {best_nd}/100")

# ============================================================
# Q4: Look at pairs of drivers with nearly identical strategies
# Find min time difference to understand scale
# ============================================================
print("\n=== Q4: Strategy pair analysis ===")

# Find pairs with identical strategies in the same race
for r in races[:200]:
    strats = list(r['strategies'].values())
    for i in range(len(strats)):
        for j in range(i+1, len(strats)):
            a, b = strats[i], strats[j]
            if (a['starting_tire'] == b['starting_tire'] and
                len(a['pit_stops']) == len(b['pit_stops'])):
                # Compare pit stop structures
                pa_laps = tuple(p['lap'] for p in a['pit_stops'])
                pb_laps = tuple(p['lap'] for p in b['pit_stops'])
                pa_tos = tuple(p['to_tire'] for p in a['pit_stops'])
                pb_tos = tuple(p['to_tire'] for p in b['pit_stops'])
                if pa_laps == pb_laps and pa_tos == pb_tos:
                    # Identical strategies! They should tie.
                    finish = r['finishing_positions']
                    da, db = a['driver_id'], b['driver_id']
                    if da in finish and db in finish:
                        pos_a = finish.index(da)
                        pos_b = finish.index(db)
                        print(f"\n  IDENTICAL STRATEGIES in {r['race_id']}:")
                        print(f"    {da} (pos={pos_a+1}) vs {db} (pos={pos_b+1})")
                        print(f"    Strategy: {a['starting_tire']} pits={pa_laps} to={pa_tos}")
                        print(f"    Positions differ: {abs(pos_a - pos_b)} spots apart")

# ============================================================
# Q5: Print race 0 with computed times for debugging
# ============================================================
print("\n=== Q5: Race 0 detailed time calculation ===")
print("Using params: off_S=-1.5, off_M=0, off_H=1.0, deg_S=0.35, deg_M=0.05, deg_H=0.02")

off_S, off_H = -1.5, 1.0
deg_S, deg_M, deg_H = 0.35, 0.05, 0.02

times = sim_all(r0, off_S, off_H, deg_S, deg_M, deg_H, 'l')
sorted_drivers = sorted(times.items(), key=lambda x: x[1])
actual = r0['finishing_positions']

print(f"{'Driver':>8} {'Actual':>6} {'Pred':>6} {'Time':>10} {'Match':>5}")
for i, (did, t) in enumerate(sorted_drivers):
    actual_pos = actual.index(did) + 1
    match = "✓" if actual_pos == i+1 else "✗"
    print(f"  {did:>6}  {actual_pos:>6}  {i+1:>6}  {t:>10.3f}  {match}")
