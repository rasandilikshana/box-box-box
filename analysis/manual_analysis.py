#!/usr/bin/env python3
"""Manual analysis of formula - print race data and test specific formulas quickly."""
import json

DATA = '/home/ubuntu/box-box-box/data/historical_races/races_00000-00999.json'

with open(DATA) as f:
    races = json.load(f)

def simulate(strategy, rc, off_S, off_M, off_H, deg_S, deg_M, deg_H, deg_formula='linear_temp'):
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
        if deg_formula == 'linear_temp':  # deg * age * temp
            degradation = d_rate * age * temp
        elif deg_formula == 'linear':     # deg * age
            degradation = d_rate * age
        elif deg_formula == 'quad':       # deg * age^2
            degradation = d_rate * age * age
        elif deg_formula == 'temp_norm':  # deg * age * temp/30
            degradation = d_rate * age * temp / 30.0
        elif deg_formula == 'temp_div20':
            degradation = d_rate * age * temp / 20.0
        t += base + off[tire] + degradation
        if lap in pit_map:
            t += pit
            tire = pit_map[lap]
            age = 0
    return t

def check_race(race, off_S, off_M, off_H, deg_S, deg_M, deg_H, dfmt='linear_temp'):
    rc = race['race_config']
    times = {}
    for s in race['strategies'].values():
        times[s['driver_id']] = simulate(s, rc, off_S, off_M, off_H, deg_S, deg_M, deg_H, dfmt)
    pred = sorted(times, key=lambda d: times[d])
    return pred == race['finishing_positions'], pred, times

def check_all(races, off_S, off_M, off_H, deg_S, deg_M, deg_H, dfmt='linear_temp'):
    return sum(1 for r in races if check_race(r, off_S, off_M, off_H, deg_S, deg_M, deg_H, dfmt)[0]) / len(races)

# Print race 0 strategies
r0 = races[0]
rc = r0['race_config']
print(f"Race 0: temp={rc['track_temp']}, base={rc['base_lap_time']}, pit={rc['pit_lane_time']}, laps={rc['total_laps']}")
print(f"Finishing: {r0['finishing_positions']}\n")
for pk, s in sorted(r0['strategies'].items()):
    did = s['driver_id']
    pits = [(p['lap'], p['to_tire'][0]) for p in s['pit_stops']]
    print(f"  {pk} {did}: {s['starting_tire'][0]} → {pits}")

# Find a race where we can isolate compound differences
# Look for a race where two drivers have SAME pit laps but DIFFERENT compound transitions
print("\n\n=== Looking for comparable driver pairs ===")
for r in races[:100]:
    rc = r['race_config']
    strategies = list(r['strategies'].values())
    found = []
    for i in range(len(strategies)):
        for j in range(i+1, len(strategies)):
            a, b = strategies[i], strategies[j]
            pa = a['pit_stops']
            pb = b['pit_stops']
            if len(pa) == len(pb) == 1:
                if pa[0]['lap'] == pb[0]['lap']:
                    if a['starting_tire'] == b['starting_tire']:
                        if pa[0]['to_tire'] != pb[0]['to_tire']:
                            found.append((a, b, pa[0]['lap'], pa[0]['to_tire'], pb[0]['to_tire']))
    if found:
        pos = {d: i+1 for i, d in enumerate(r['finishing_positions'])}
        a, b, pit_lap, ta, tb = found[0]
        print(f"\nRace {r['race_id']}: temp={rc['track_temp']} laps={rc['total_laps']} base={rc['base_lap_time']}")
        print(f"  {a['driver_id']} (pos={pos[a['driver_id']]}): {a['starting_tire']}→lap{pit_lap}→{ta}")
        print(f"  {b['driver_id']} (pos={pos[b['driver_id']]}): {b['starting_tire']}→lap{pit_lap}→{tb}")
        # Difference: only in compound from lap pit_lap+1 onwards
        after_laps = rc['total_laps'] - pit_lap
        print(f"  Laps on different compound: {after_laps}")
        print(f"  If {a['driver_id']} faster: {ta} better than {tb}")
        if len(found) > 1:
            break

# Now: quick benchmark of all formula types on first 50 races
print("\n\n=== Quick benchmark 50 races ===")
sample50 = races[:50]

# Known reasonable params from analysis
test_params = [
    # off_S, off_M, off_H, deg_S, deg_M, deg_H
    [-1.0, 0.0, 1.0, 0.003, 0.001, 0.0005, 'linear_temp'],
    [-1.0, 0.0, 1.0, 0.05, 0.02, 0.01, 'linear'],
    [-1.0, 0.0, 1.0, 0.005, 0.002, 0.001, 'quad'],
    [-1.0, 0.0, 1.0, 0.1, 0.04, 0.02, 'temp_norm'],
    [-0.5, 0.0, 0.5, 0.1, 0.04, 0.02, 'linear'],
    [-0.5, 0.0, 0.5, 0.003, 0.001, 0.0005, 'linear_temp'],
    [-2.0, 0.0, 2.0, 0.05, 0.02, 0.01, 'linear'],
    [-2.0, 0.0, 2.0, 0.003, 0.001, 0.0005, 'linear_temp'],
    [-0.5, 0.0, 1.0, 0.003, 0.001, 0.0005, 'linear_temp'],
    [-1.5, 0.0, 1.5, 0.003, 0.001, 0.0005, 'linear_temp'],
    [-1.0, 0.0, 1.0, 0.002, 0.001, 0.0005, 'linear_temp'],
    [-1.0, 0.0, 1.0, 0.004, 0.0015, 0.0007, 'linear_temp'],
]

for p in test_params:
    so, mo, ho, sd, md, hd, dfmt = p
    acc = check_all(sample50, so, mo, ho, sd, md, hd, dfmt)
    if acc > 0.1:
        print(f"  {dfmt}: off=[{so},{mo},{ho}] deg=[{sd},{md},{hd}] → {acc*100:.1f}%")

# Focused search on linear_temp (seems most likely for F1)
print("\n\n=== Fine search: linear_temp formula ===")
best, bp = 0, None
for so in [-0.3, -0.5, -0.7, -1.0, -1.2, -1.5, -2.0]:
    for ho in [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]:
        for sd in [0.001, 0.002, 0.003, 0.004, 0.005]:
            for md in [0.0005, 0.001, 0.0015, 0.002]:
                for hd in [0.0002, 0.0003, 0.0005, 0.0007, 0.001]:
                    acc = check_all(sample50, so, 0, ho, sd, md, hd, 'linear_temp')
                    if acc > best:
                        best, bp = acc, [so, 0, ho, sd, md, hd]
print(f"Best linear_temp: {best*100:.1f}% {bp}")

print("\n=== Fine search: linear (no temp) formula ===")
best2, bp2 = 0, None
for so in [-0.3, -0.5, -1.0, -1.5, -2.0]:
    for ho in [0.3, 0.5, 1.0, 1.5, 2.0]:
        for sd in [0.02, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15]:
            for md in [0.01, 0.015, 0.02, 0.03, 0.04]:
                for hd in [0.005, 0.007, 0.01, 0.015, 0.02]:
                    acc = check_all(sample50, so, 0, ho, sd, md, hd, 'linear')
                    if acc > best2:
                        best2, bp2 = acc, [so, 0, ho, sd, md, hd]
print(f"Best linear(no temp): {best2*100:.1f}% {bp2}")

print("\n=== Fine search: temp_norm (temp/30) formula ===")
best3, bp3 = 0, None
for so in [-0.3, -0.5, -1.0, -1.5, -2.0]:
    for ho in [0.3, 0.5, 1.0, 1.5, 2.0]:
        for sd in [0.02, 0.03, 0.05, 0.07, 0.1]:
            for md in [0.01, 0.015, 0.02, 0.03]:
                for hd in [0.005, 0.007, 0.01, 0.015]:
                    acc = check_all(sample50, so, 0, ho, sd, md, hd, 'temp_norm')
                    if acc > best3:
                        best3, bp3 = acc, [so, 0, ho, sd, md, hd]
print(f"Best temp_norm: {best3*100:.1f}% {bp3}")

print("\n=== SUMMARY ===")
print(f"linear_temp: {best*100:.1f}% {bp}")
print(f"linear:      {best2*100:.1f}% {bp2}")
print(f"temp_norm:   {best3*100:.1f}% {bp3}")
