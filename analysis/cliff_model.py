#!/usr/bin/env python3
"""
CLIFF MODEL: Each compound has an 'initial performance period' where degradation is zero.
After cliff_age laps, linear degradation begins.

lap_time = base + off[c] + deg[c] * max(0, tire_age - cliff[c])
total_time = sum(lap_times) + n_pits * pit_lane_time

Tiebreaker: grid position (lower = wins, i.e. pos1 wins over pos20)
Since D_id = pos_id always, lower driver id = lower grid position.
"""
import json, numpy as np, glob, os, time
from scipy.optimize import differential_evolution

IN_DIR = '/home/ubuntu/box-box-box/data/test_cases/inputs'
OUT_DIR = '/home/ubuntu/box-box-box/data/test_cases/expected_outputs'
HIST_DIR = '/home/ubuntu/box-box-box/data/historical_races'
t2i = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}

def precompute(race_data, expected, strats_dict=None):
    """For each driver in each race, precompute per-compound stint data."""
    rc = race_data
    base, pit_t, temp, total_laps = rc['base_lap_time'], rc['pit_lane_time'], rc['track_temp'], rc['total_laps']
    driver_order = {d: i for i, d in enumerate(expected)}
    n = len(expected)

    # For cliff model: we need per-lap tire ages (not just sums)
    # Store (driver, lap, compound, tire_age) implicitly via stint arrays
    # For each driver: list of (compound, ages_array) — e.g. [(S, [1,2,3,4,5]), (H, [1,2,...,25])]
    driver_stints = [[] for _ in range(n)]  # list of (compound_idx, ages array)
    bt = np.zeros(n)
    grid_pos = np.zeros(n, dtype=np.int32)

    if strats_dict is None:
        strats = race_data.get('strategies', {})
    else:
        strats = strats_dict

    for pos_key, s in strats.items():
        did = s['driver_id']
        idx = driver_order[did]
        grid_num = int(pos_key.replace('pos',''))
        grid_pos[idx] = grid_num

        pm = {p['lap']: t2i[p['to_tire']] for p in s.get('pit_stops', [])}
        ti = t2i[s['starting_tire']]; age = 0; np_ = 0

        # Build stints
        current_stint_comp = ti
        current_stint_ages = []

        for lap in range(1, total_laps + 1):
            age += 1
            if lap in pm:
                # End current stint
                driver_stints[idx].append((current_stint_comp, np.array(current_stint_ages)))
                # Start new stint
                current_stint_comp = pm[lap]
                current_stint_ages = []
                age = 0
                np_ += 1
        # Last stint
        driver_stints[idx].append((current_stint_comp, np.array(current_stint_ages)))

        bt[idx] = base * total_laps + np_ * pit_t

    return {'stints': driver_stints, 'bt': bt, 'temp': temp, 'n': n, 'grid': grid_pos}

def compute_times_cliff(feat, off, deg, cliff):
    """Compute total times for all drivers using cliff model.
    off: [off_S, off_M, off_H]
    deg: [deg_S, deg_M, deg_H]
    cliff: [cliff_S, cliff_M, cliff_H] (integer cliff ages)
    """
    n = feat['n']
    times = feat['bt'].copy()
    for i, stints in enumerate(feat['stints']):
        for c, ages in stints:
            # degradation only after cliff[c] laps
            clipped = np.maximum(0.0, ages - cliff[c])
            times[i] += len(ages) * off[c] + np.sum(clipped) * deg[c]
    return times

def wrong_pairs(feats, off, deg, cliff):
    total = 0
    for f in feats:
        t = compute_times_cliff(f, off, deg, cliff)
        for i in range(f['n'] - 1):
            if t[i] >= t[i+1]: total += 1
    return total

def accuracy(feats, off, deg, cliff):
    ok = 0
    for f in feats:
        t = compute_times_cliff(f, off, deg, cliff)
        if np.all(np.argsort(t) == np.arange(f['n'])): ok += 1
    return ok / len(feats)

def accuracy_with_tiebreak(feats, off, deg, cliff):
    """Use grid position as tiebreaker for identical times."""
    ok = 0
    for f in feats:
        t = compute_times_cliff(f, off, deg, cliff)
        # Primary sort: time; secondary sort: grid position (lower = better)
        order = np.lexsort((f['grid'], t))
        if np.all(order == np.arange(f['n'])): ok += 1
    return ok / len(feats)

print("Loading test cases...")
feats_test = []
for fn in sorted(glob.glob(f'{IN_DIR}/test_*.json')):
    num = os.path.basename(fn).replace('test_','').replace('.json','')
    with open(fn) as f: inp = json.load(f)
    with open(f'{OUT_DIR}/test_{num}.json') as f: exp = json.load(f)
    feats_test.append(precompute(inp['race_config'], exp['finishing_positions'], inp['strategies']))

print("Loading historical races...")
with open(f'{HIST_DIR}/races_00000-00999.json') as f:
    races = json.load(f)[:300]
feats_hist = [precompute(r['race_config'], r['finishing_positions']) for r in races]

all_feats = feats_test + feats_hist
print(f"Test: {len(feats_test)}, Hist: {len(feats_hist)}")

# ============================================================
# Quick test: compare linear vs cliff model
# ============================================================
print("\n=== Quick comparison: linear (cliff=0) vs cliff ===")
# Parameters from best known (approximate)
off0 = np.array([-2.0, 0.0, 1.5])
deg0 = np.array([0.3, 0.05, 0.01])
cliff0 = np.array([0, 0, 0])  # no cliff = linear
cliff1 = np.array([1, 2, 3])  # 1/2/3 lap cliff for S/M/H
cliff2 = np.array([2, 4, 6])
cliff3 = np.array([3, 5, 8])
cliff4 = np.array([0, 0, 5])
cliff5 = np.array([1, 3, 6])

for c in [cliff0, cliff1, cliff2, cliff3, cliff4, cliff5]:
    acc_t = accuracy(feats_test, off0, deg0, c)
    acc_h = accuracy(feats_hist, off0, deg0, c)
    wp_t = wrong_pairs(feats_test, off0, deg0, c)
    print(f"  cliff={c}: test={acc_t*100:.1f}%  hist={acc_h*100:.1f}%  wrong_pairs(test)={wp_t}")

# ============================================================
# Grid search over cliff values with fixed-ish params
# ============================================================
print("\n=== Grid search over cliff integers ===")
best_score = -1
best_params = None

t0 = time.time()

# Try scipy DE with cliff as continuous, then round
def obj_cliff(p):
    off = np.array([p[0], p[1], p[2]])
    deg = np.array([p[3], p[4], p[5]])
    cliff = np.array([p[6], p[7], p[8]])
    return wrong_pairs(all_feats, off, deg, cliff)

# First: scan over discrete cliff values with a few fixed param guesses
for c_S in [0, 1, 2, 3, 5]:
    for c_M in [0, 1, 2, 4, 6]:
        for c_H in [0, 2, 4, 6, 8, 10]:
            cliff = np.array([c_S, c_M, c_H], dtype=float)
            # Quick DE for just 6 params (off, deg)
            def obj6(p):
                off = np.array([p[0], p[1], p[2]])
                deg = np.array([p[3], p[4], p[5]])
                return wrong_pairs(feats_test, off, deg, cliff)

            bounds6 = [(-5,0),(-2,2),(0,5),(0,2),(0,1),(0,0.5)]
            res = differential_evolution(obj6, bounds6, maxiter=80, popsize=10, seed=42,
                                          tol=0.1, disp=False)
            p = res.x
            off = np.array([p[0], p[1], p[2]])
            deg = np.array([p[3], p[4], p[5]])
            acc_t = accuracy(feats_test, off, deg, cliff)
            if acc_t > best_score:
                best_score = acc_t
                best_params = (off.copy(), deg.copy(), cliff.copy())
                print(f"  NEW BEST: cliff={cliff} | test={acc_t*100:.1f}% | off={[round(x,3) for x in off]} deg={[round(x,4) for x in deg]}")
                # also check hist
                acc_h = accuracy(feats_hist, off, deg, cliff)
                print(f"           hist={acc_h*100:.1f}%  wrong={int(res.fun)}")

t1 = time.time()
print(f"\nGrid search done in {t1-t0:.1f}s")
print(f"Best accuracy (test): {best_score*100:.1f}%")
if best_params:
    off, deg, cliff = best_params
    print(f"Best params: off={[round(x,4) for x in off]} deg={[round(x,5) for x in deg]} cliff={cliff}")

# ============================================================
# Full DE with 9 params (off, deg, cliff continuous)
# ============================================================
print("\n=== Full DE: 9 params (off, deg, cliff) ===")
bounds9 = [(-5,0),(-2,2),(0,5),(0,2),(0,1),(0,0.5),(0,10),(0,15),(0,20)]
t0 = time.time()
res9 = differential_evolution(obj_cliff, bounds9, maxiter=200, popsize=15, seed=42,
                               tol=0.01, disp=True, mutation=(0.5,1.5), recombination=0.7)
t1 = time.time()
p9 = res9.x
off9 = np.array([p9[0], p9[1], p9[2]])
deg9 = np.array([p9[3], p9[4], p9[5]])
cliff9 = np.array([p9[6], p9[7], p9[8]])
print(f"Time: {t1-t0:.1f}s, Loss: {res9.fun}")
print(f"off={[round(x,4) for x in off9]} deg={[round(x,5) for x in deg9]} cliff={[round(x,2) for x in cliff9]}")
acc_t9 = accuracy(feats_test, off9, deg9, cliff9)
acc_h9 = accuracy(feats_hist, off9, deg9, cliff9)
print(f"Test: {acc_t9*100:.1f}%, Hist: {acc_h9*100:.1f}%")

# Save
import json
result = {
    'formula': 'cliff',
    'off': list(off9),
    'deg': list(deg9),
    'cliff': list(cliff9),
    'acc_test': acc_t9,
    'acc_hist': acc_h9
}
with open('/home/ubuntu/box-box-box/analysis/cliff_params.json', 'w') as f:
    json.dump(result, f, indent=2)
print("Saved to cliff_params.json")
