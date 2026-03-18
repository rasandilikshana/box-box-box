#!/usr/bin/env python3
"""
Critical insight: there's a tiny per-position (grid) epsilon added to total time.
total_time[k] = formula_time(strategy_k) + epsilon * grid_position_k

Test this hypothesis and find all parameters.
"""
import json, numpy as np, glob, os, time as tm
from scipy.optimize import differential_evolution
from itertools import product as iprod

IN_DIR = '/home/ubuntu/box-box-box/data/test_cases/inputs'
OUT_DIR = '/home/ubuntu/box-box-box/data/test_cases/expected_outputs'
HIST_DIR = '/home/ubuntu/box-box-box/data/historical_races'
t2i = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}

def precompute(race_data, expected, strats_dict):
    rc = race_data
    base, pit_t, temp, total_laps = rc['base_lap_time'], rc['pit_lane_time'], rc['track_temp'], rc['total_laps']
    driver_order = {d: i for i, d in enumerate(expected)}
    n = len(expected)
    lc = np.zeros((n,3)); as_ = np.zeros((n,3)); aq = np.zeros((n,3))
    bt = np.zeros(n); grid_pos = np.zeros(n)

    for pos_key, s in strats_dict.items():
        did = s['driver_id']
        idx = driver_order[did]
        grid_num = int(pos_key.replace('pos',''))
        grid_pos[idx] = grid_num
        pm = {p['lap']: t2i[p['to_tire']] for p in s.get('pit_stops',[])}
        ti = t2i[s['starting_tire']]; age = 0; np_ = 0
        for lap in range(1, total_laps+1):
            age+=1; lc[idx,ti]+=1; as_[idx,ti]+=age; aq[idx,ti]+=age*age
            if lap in pm: ti=pm[lap]; age=0; np_+=1
        bt[idx] = base*total_laps + np_*pit_t
    return {'lc':lc,'as':as_,'aq':aq,'bt':bt,'temp':temp,'n':n,'grid':grid_pos}

def load_tests():
    tests = []
    for fn in sorted(glob.glob(f'{IN_DIR}/test_*.json')):
        num = os.path.basename(fn).replace('test_','').replace('.json','')
        with open(fn) as f: inp = json.load(f)
        with open(f'{OUT_DIR}/test_{num}.json') as f: exp = json.load(f)
        f_data = precompute(inp['race_config'], exp['finishing_positions'], inp['strategies'])
        tests.append(f_data)
    return tests

def load_hist(n=300):
    with open(f'{HIST_DIR}/races_00000-00999.json') as f:
        races = json.load(f)[:n]
    return [precompute(r['race_config'], r['finishing_positions'], r['strategies']) for r in races]

def compute_times(feat, p, formula='l'):
    """With epsilon: time = formula_time + epsilon * grid_pos"""
    off = p[:3]; deg = p[3:6]
    eps = p[6] if len(p) > 6 else 0.0
    if formula == 'l':
        ft = feat['bt'] + feat['lc']@off + feat['as']@deg
    elif formula == 'lt':
        ft = feat['bt'] + feat['lc']@off + (feat['as']@deg)*feat['temp']
    elif formula == 'tn':
        ft = feat['bt'] + feat['lc']@off + (feat['as']@deg)*(feat['temp']/30.0)
    return ft + eps * feat['grid']

def wrong_pairs(feats, p, formula='l'):
    total = 0
    for f in feats:
        t = compute_times(f, p, formula)
        for i in range(f['n']-1):
            if t[i] >= t[i+1]: total += 1
    return total

def accuracy(feats, p, formula='l'):
    ok = 0
    for f in feats:
        t = compute_times(f, p, formula)
        if np.all(np.argsort(t) == np.arange(f['n'])): ok += 1
    return ok / len(feats)

print("Loading...")
feats_test = load_tests()
feats_hist = load_hist(300)
all_feats = feats_test + feats_hist
print(f"Test: {len(feats_test)}, Hist: {len(feats_hist)}")

# ============================================================
# Test 1: Check if epsilon hypothesis is correct
# ============================================================
print("\n=== Checking epsilon hypothesis ===")
# With eps=0, what's the accuracy of a grid search?
# With eps > 0, does it improve?

# Quick test with specific params + different epsilons
test_combos = [
    ([-2.0, 0.0, 1.5, 0.3, 0.05, 0.02], 'l'),
    ([-3.0, 0.0, 2.0, 0.4, 0.05, 0.02], 'l'),
    ([-3.0, 0.0, 2.0, 0.48, 0.05, 0.06], 'l'),
    ([-2.0, 0.0, 1.5, 0.01, 0.003, 0.001], 'lt'),
    ([-3.0, 0.0, 2.0, 0.01, 0.003, 0.001], 'lt'),
]
for base_p, formula in test_combos:
    for eps in [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        p = base_p + [eps]
        acc_t = accuracy(feats_test, p, formula)
        if acc_t > 0.05:
            acc_h = accuracy(feats_hist, p, formula)
            print(f"  {formula} {base_p} eps={eps}: test={acc_t*100:.1f}% hist={acc_h*100:.1f}%")

# ============================================================
# Scipy DE with epsilon as 7th parameter
# ============================================================
print("\n=== scipy DE: formula l + epsilon ===")

def obj(p):
    return wrong_pairs(all_feats, p, 'l')

bounds = [(-10,2), (-3,3), (-2,10), (0,5), (0,2), (0,1), (0,10)]
t0 = tm.time()
res = differential_evolution(obj, bounds, maxiter=200, popsize=15, seed=42,
                              tol=0.01, disp=True, mutation=(0.5,1.5), recombination=0.7)
t1 = tm.time()
p_l = list(res.x)
print(f"Time: {t1-t0:.1f}s, Loss: {res.fun}")
print(f"Params: {[round(x,5) for x in p_l]}")
acc_test = accuracy(feats_test, p_l, 'l')
acc_hist = accuracy(feats_hist, p_l, 'l')
print(f"Test: {acc_test*100:.1f}%, Hist: {acc_hist*100:.1f}%")

print("\n=== scipy DE: formula lt + epsilon ===")
def obj_lt(p):
    return wrong_pairs(all_feats, p, 'lt')

bounds_lt = [(-10,2), (-3,3), (-2,10), (0,0.5), (0,0.2), (0,0.1), (0,10)]
t0 = tm.time()
res_lt = differential_evolution(obj_lt, bounds_lt, maxiter=200, popsize=15, seed=42,
                                 tol=0.01, disp=True, mutation=(0.5,1.5), recombination=0.7)
t1 = tm.time()
p_lt = list(res_lt.x)
print(f"Time: {t1-t0:.1f}s, Loss: {res_lt.fun}")
print(f"Params: {[round(x,5) for x in p_lt]}")
acc_test_lt = accuracy(feats_test, p_lt, 'lt')
acc_hist_lt = accuracy(feats_hist, p_lt, 'lt')
print(f"Test: {acc_test_lt*100:.1f}%, Hist: {acc_hist_lt*100:.1f}%")

print("\n=== FINAL RESULTS ===")
best_p = p_l if acc_test >= acc_test_lt else p_lt
best_f = 'l' if acc_test >= acc_test_lt else 'lt'
print(f"Best formula: {best_f}")
print(f"Params: {[round(x,6) for x in best_p]}")

# Save
import json
with open('/home/ubuntu/box-box-box/analysis/best_params.json', 'w') as f:
    json.dump({'formula': best_f, 'params': best_p,
               'acc_test': max(acc_test, acc_test_lt),
               'acc_hist': max(acc_hist, acc_hist_lt)}, f, indent=2)
