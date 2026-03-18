#!/usr/bin/env python3
"""Quick targeted test with smarter parameter estimates."""
import json, numpy as np

DATA = '/home/ubuntu/box-box-box/data/historical_races/races_00000-00999.json'

with open(DATA) as f:
    races = json.load(f)

def simulate(strategy, rc, off_S, off_M, off_H, deg_S, deg_M, deg_H, formula='lt'):
    total_laps = rc['total_laps']
    base = rc['base_lap_time']
    pit = rc['pit_lane_time']
    temp = rc['track_temp']
    pit_map = {p['lap']: p['to_tire'] for p in strategy.get('pit_stops', [])}
    tire = strategy['starting_tire']
    age = 0
    t = 0.0
    off = {'SOFT': off_S, 'MEDIUM': off_M, 'HARD': off_H}
    deg = {'SOFT': deg_S, 'MEDIUM': deg_M, 'HARD': deg_H}
    for lap in range(1, total_laps + 1):
        age += 1
        d_rate = deg[tire]
        if formula == 'lt':    # deg*age*temp
            d = d_rate * age * temp
        elif formula == 'l':   # deg*age
            d = d_rate * age
        elif formula == 'tn':  # deg*age*temp/30
            d = d_rate * age * temp / 30
        t += base + off[tire] + d
        if lap in pit_map:
            t += pit
            tire = pit_map[lap]
            age = 0
    return t

def check_n(races, off_S, off_M, off_H, deg_S, deg_M, deg_H, formula='lt', n=200):
    ok = 0
    for r in races[:n]:
        rc = r['race_config']
        times = {s['driver_id']: simulate(s, rc, off_S, off_M, off_H, deg_S, deg_M, deg_H, formula)
                 for s in r['strategies'].values()}
        pred = sorted(times, key=lambda d: times[d])
        if pred == r['finishing_positions']:
            ok += 1
    return ok / min(n, len(races))

# Manual calculation for race 0 to calibrate
r0 = races[0]
rc = r0['race_config']
print(f"Race 0: temp={rc['track_temp']}, base={rc['base_lap_time']}, laps={rc['total_laps']}")
print(f"Actual order (1st-5th): {r0['finishing_positions'][:5]}")
print()

# Print each driver's strategy and position
fin = {d: i+1 for i, d in enumerate(r0['finishing_positions'])}
for pk, s in sorted(r0['strategies'].items()):
    did = s['driver_id']
    pits = [(p['lap'], p['to_tire']) for p in s['pit_stops']]
    print(f"  {did} pos={fin[did]:2d}: {s['starting_tire']:6s} → {pits}")

# Test my calculated params
print("\n--- Testing calibrated params ---")
# off_S=-1.5, off_M=0, off_H=1.0, deg_S=0.01, deg_M=0.003, deg_H=0.001, formula=lt
params_to_test = [
    # off_S, off_M, off_H, deg_S, deg_M, deg_H, formula
    (-1.5, 0.0, 1.0, 0.010, 0.003, 0.001, 'lt'),
    (-1.5, 0.0, 1.0, 0.008, 0.003, 0.001, 'lt'),
    (-1.5, 0.0, 1.0, 0.012, 0.004, 0.001, 'lt'),
    (-1.0, 0.0, 1.0, 0.010, 0.003, 0.001, 'lt'),
    (-2.0, 0.0, 1.0, 0.010, 0.003, 0.001, 'lt'),
    (-1.5, 0.0, 1.5, 0.010, 0.003, 0.001, 'lt'),
    (-1.5, 0.0, 1.0, 0.010, 0.003, 0.0015, 'lt'),
    (-1.5, 0.0, 1.0, 0.010, 0.003, 0.001, 'tn'),
    (-1.5, 0.0, 1.0, 0.30, 0.09, 0.03, 'tn'),
]
for p in params_to_test:
    so, mo, ho, sd, md, hd, fmt = p
    # Test on first 100 races
    acc = check_n(races, so, mo, ho, sd, md, hd, fmt, 100)
    print(f"  {fmt}: off=[{so},{mo},{ho}] deg=[{sd},{md},{hd}] → {acc*100:.1f}%")

# Brute force search with better range
print("\n--- Systematic search (better range) ---")
best, bp = 0.0, None
count = 0
for so in [-0.5, -1.0, -1.5, -2.0, -3.0]:
    for ho in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for sd in [0.005, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020]:
            for md in [0.001, 0.002, 0.003, 0.004, 0.005]:
                for hd in [0.0005, 0.001, 0.0015, 0.002]:
                    p = (so, 0.0, ho, sd, md, hd, 'lt')
                    acc = check_n(races, so, 0.0, ho, sd, md, hd, 'lt', 100)
                    count += 1
                    if acc > best:
                        best, bp = acc, p
                        print(f"  NEW BEST {best*100:.1f}%: {bp}")
print(f"\nBest lt: {best*100:.1f}% | {bp}")
print(f"Tested {count} combinations")

# Now try with temp/30 normalization on best found
print("\n--- Systematic search (temp/30 normalization) ---")
best2, bp2 = 0.0, None
for so in [-0.5, -1.0, -1.5, -2.0]:
    for ho in [0.5, 1.0, 1.5, 2.0]:
        for sd in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
            for md in [0.03, 0.05, 0.07, 0.1]:
                for hd in [0.01, 0.02, 0.03]:
                    acc = check_n(races, so, 0.0, ho, sd, md, hd, 'tn', 100)
                    if acc > best2:
                        best2, bp2 = acc, (so, 0.0, ho, sd, md, hd, 'tn')
                        print(f"  NEW BEST {best2*100:.1f}%: {bp2}")
print(f"Best tn: {best2*100:.1f}% | {bp2}")
