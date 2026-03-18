#!/usr/bin/env python3
"""Targeted formula discovery - fast and direct."""
import json, os, sys

DATA_DIR = '/home/ubuntu/box-box-box/data/historical_races'
TEST_IN = '/home/ubuntu/box-box-box/data/test_cases/inputs'
TEST_OUT = '/home/ubuntu/box-box-box/data/test_cases/expected_outputs'

with open(f'{DATA_DIR}/races_00000-00999.json') as f:
    races = json.load(f)

# Print race 0 in detail
r0 = races[0]
rc = r0['race_config']
print(f"Race 0: {r0['race_id']} | {rc['track']} | laps={rc['total_laps']} base={rc['base_lap_time']} pit={rc['pit_lane_time']} temp={rc['track_temp']}")
print(f"Finishing: {r0['finishing_positions']}")
print()
finish_pos = {d: i+1 for i, d in enumerate(r0['finishing_positions'])}
for pk, s in sorted(r0['strategies'].items()):
    did = s['driver_id']
    pits = [(p['lap'], p['to_tire'][0]) for p in s['pit_stops']]
    print(f"  {pk} {did} (fin={finish_pos[did]}): {s['starting_tire'][0]} pits={pits}")

# Now test all formula variants on first 500 races
def simulate(strategy, rc, off, deg_fn):
    total_laps = rc['total_laps']
    base = rc['base_lap_time']
    pit_time = rc['pit_lane_time']
    temp = rc['track_temp']
    pit_map = {p['lap']: p['to_tire'] for p in strategy.get('pit_stops', [])}
    tire = strategy['starting_tire']
    age = 0
    t = 0.0
    for lap in range(1, total_laps + 1):
        age += 1
        t += base + off[tire] + deg_fn(tire, age, temp)
        if lap in pit_map:
            t += pit_time
            tire = pit_map[lap]
            age = 0
    return t

def check(races, off, deg_fn):
    ok = 0
    for r in races:
        rc = r['race_config']
        times = {s['driver_id']: simulate(s, rc, off, deg_fn) for s in r['strategies'].values()}
        pred = sorted(times, key=lambda d: times[d])
        if pred == r['finishing_positions']:
            ok += 1
    return ok / len(races)

sample = races[:500]
print()
print("=== Testing Formula Variants ===")

# Variant 1: base + off[c] + deg[c] * age * temp
print("\n-- Formula 1: off + deg*age*temp --")
best1, bp1 = 0, None
for so in [-0.5, -1.0, -1.5, -2.0]:
    for ho in [0.5, 1.0, 1.5, 2.0]:
        for sd in [0.001, 0.002, 0.003, 0.005]:
            for md in [0.0005, 0.001, 0.0015]:
                for hd in [0.0002, 0.0005, 0.001]:
                    off = {'SOFT': so, 'MEDIUM': 0.0, 'HARD': ho}
                    def make1(a,b,c):
                        def f(comp, age, temp):
                            return {'SOFT':a,'MEDIUM':b,'HARD':c}[comp]*age*temp
                        return f
                    acc = check(sample[:100], off, make1(sd,md,hd))
                    if acc > best1:
                        best1, bp1 = acc, [so,0,ho,sd,md,hd]
                        if acc > 0.3: print(f"  {acc*100:.1f}% {bp1}")
print(f"Best F1: {best1*100:.1f}% {bp1}")

# Variant 2: base + off[c] + deg[c] * age (no temp)
print("\n-- Formula 2: off + deg*age (no temp) --")
best2, bp2 = 0, None
for so in [-0.3, -0.5, -1.0, -1.5, -2.0]:
    for ho in [0.3, 0.5, 1.0, 1.5, 2.0]:
        for sd in [0.02, 0.05, 0.1, 0.15, 0.2]:
            for md in [0.01, 0.02, 0.03, 0.05]:
                for hd in [0.005, 0.01, 0.015, 0.02]:
                    off = {'SOFT': so, 'MEDIUM': 0.0, 'HARD': ho}
                    def make2(a,b,c):
                        def f(comp, age, temp):
                            return {'SOFT':a,'MEDIUM':b,'HARD':c}[comp]*age
                        return f
                    acc = check(sample[:100], off, make2(sd,md,hd))
                    if acc > best2:
                        best2, bp2 = acc, [so,0,ho,sd,md,hd]
                        if acc > 0.3: print(f"  {acc*100:.1f}% {bp2}")
print(f"Best F2: {best2*100:.1f}% {bp2}")

# Variant 3: base + off[c] + deg[c] * age^2
print("\n-- Formula 3: off + deg*age^2 --")
best3, bp3 = 0, None
for so in [-0.5, -1.0, -1.5]:
    for ho in [0.5, 1.0, 1.5]:
        for sd in [0.001, 0.002, 0.005, 0.01]:
            for md in [0.0005, 0.001, 0.002]:
                for hd in [0.0002, 0.0005, 0.001]:
                    off = {'SOFT': so, 'MEDIUM': 0.0, 'HARD': ho}
                    def make3(a,b,c):
                        def f(comp, age, temp):
                            return {'SOFT':a,'MEDIUM':b,'HARD':c}[comp]*age*age
                        return f
                    acc = check(sample[:100], off, make3(sd,md,hd))
                    if acc > best3:
                        best3, bp3 = acc, [so,0,ho,sd,md,hd]
                        if acc > 0.3: print(f"  {acc*100:.1f}% {bp3}")
print(f"Best F3: {best3*100:.1f}% {bp3}")

# Variant 4: base + off[c] + deg[c] * age * (1 + tc*temp) -- additive temp effect
print("\n-- Formula 4: off + deg*age*(1+tc*temp) --")
best4, bp4 = 0, None
for so in [-0.5, -1.0, -1.5]:
    for ho in [0.5, 1.0, 1.5]:
        for sd in [0.02, 0.05, 0.1]:
            for md in [0.01, 0.02]:
                for hd in [0.005, 0.01]:
                    for tc in [0.01, 0.02, 0.03, 0.05]:
                        off = {'SOFT': so, 'MEDIUM': 0.0, 'HARD': ho}
                        def make4(a,b,c,tc_):
                            def f(comp, age, temp):
                                return {'SOFT':a,'MEDIUM':b,'HARD':c}[comp]*age*(1+tc_*temp)
                            return f
                        acc = check(sample[:100], off, make4(sd,md,hd,tc))
                        if acc > best4:
                            best4, bp4 = acc, [so,0,ho,sd,md,hd,tc]
                            if acc > 0.3: print(f"  {acc*100:.1f}% {bp4}")
print(f"Best F4: {best4*100:.1f}% {bp4}")

# Variant 5: base + off[c] + (deg_base[c] + temp * temp_deg[c]) * age
print("\n-- Formula 5: off + (base_deg + temp_deg*temp)*age --")
best5, bp5 = 0, None
for so in [-0.5, -1.0]:
    for ho in [0.5, 1.0]:
        for sd_b in [0.01, 0.02, 0.05]:
            for sd_t in [0.001, 0.002, 0.003]:
                for md_b in [0.005, 0.01, 0.02]:
                    for md_t in [0.0005, 0.001]:
                        for hd_b in [0.002, 0.005, 0.01]:
                            for hd_t in [0.0002, 0.0005]:
                                off = {'SOFT': so, 'MEDIUM': 0.0, 'HARD': ho}
                                def make5(a,at,b,bt,c,ct):
                                    def f(comp, age, temp):
                                        r = {'SOFT':a+at*temp,'MEDIUM':b+bt*temp,'HARD':c+ct*temp}
                                        return r[comp]*age
                                    return f
                                acc = check(sample[:50], off, make5(sd_b,sd_t,md_b,md_t,hd_b,hd_t))
                                if acc > best5:
                                    best5, bp5 = acc, [so,0,ho,sd_b,sd_t,md_b,md_t,hd_b,hd_t]
                                    if acc > 0.3: print(f"  {acc*100:.1f}% {bp5}")
print(f"Best F5: {best5*100:.1f}% {bp5}")

print("\n=== SUMMARY ===")
print(f"F1 (deg*age*temp): {best1*100:.1f}% {bp1}")
print(f"F2 (deg*age noT):  {best2*100:.1f}% {bp2}")
print(f"F3 (deg*age^2):    {best3*100:.1f}% {bp3}")
print(f"F4 (deg*age*(1+tc*temp)): {best4*100:.1f}% {bp4}")
print(f"F5 (base_deg+temp_deg)*age: {best5*100:.1f}% {bp5}")
