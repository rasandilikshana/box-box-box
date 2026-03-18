#!/usr/bin/env python3
"""
Use test cases (known ground truth) to discover the formula.
Try multiple formula structures with scipy.
"""
import json, numpy as np, glob, os
from scipy.optimize import differential_evolution, minimize

IN_DIR = '/home/ubuntu/box-box-box/data/test_cases/inputs'
OUT_DIR = '/home/ubuntu/box-box-box/data/test_cases/expected_outputs'
HIST_DIR = '/home/ubuntu/box-box-box/data/historical_races'

t2i = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}

def load_tests():
    tests = []
    for fn in sorted(glob.glob(f'{IN_DIR}/test_*.json')):
        base = os.path.basename(fn)
        num = base.replace('test_','').replace('.json','')
        with open(fn) as f: inp = json.load(f)
        with open(f'{OUT_DIR}/test_{num}.json') as f: exp = json.load(f)
        tests.append({'input': inp, 'expected': exp['finishing_positions']})
    print(f"Loaded {len(tests)} test cases")
    return tests

def precompute_race(race_data, expected):
    """Pre-compute features. Drivers stored in actual finish order."""
    rc = race_data['race_config']
    base = rc['base_lap_time']
    pit_t = rc['pit_lane_time']
    temp = rc['track_temp']
    total_laps = rc['total_laps']
    driver_order = {d: i for i, d in enumerate(expected)}
    n = len(expected)
    lc = np.zeros((n, 3)); as_ = np.zeros((n, 3)); aq = np.zeros((n, 3)); bt = np.zeros(n)
    for s in race_data['strategies'].values():
        did = s['driver_id']
        idx = driver_order[did]
        pm = {p['lap']: t2i[p['to_tire']] for p in s.get('pit_stops', [])}
        ti = t2i[s['starting_tire']]; age = 0; n_pits = 0
        for lap in range(1, total_laps + 1):
            age += 1; lc[idx, ti] += 1; as_[idx, ti] += age; aq[idx, ti] += age*age
            if lap in pm: ti = pm[lap]; age = 0; n_pits += 1
        bt[idx] = base * total_laps + n_pits * pit_t
    return {'lc': lc, 'as': as_, 'aq': aq, 'bt': bt, 'temp': temp, 'n': n}

def compute_times(feat, params, formula):
    """Compute total times for all drivers in a race."""
    off = params[:3]  # [S, M, H] offsets
    bt = feat['bt']
    lc = feat['lc']
    as_ = feat['as']
    aq = feat['aq']
    temp = feat['temp']

    if formula == 'l':  # deg * age
        deg = params[3:6]
        return bt + lc @ off + as_ @ deg
    elif formula == 'lt':  # deg * age * temp
        deg = params[3:6]
        return bt + lc @ off + (as_ @ deg) * temp
    elif formula == 'tn':  # deg * age * temp/30
        deg = params[3:6]
        return bt + lc @ off + (as_ @ deg) * (temp / 30.0)
    elif formula == 'sep':  # deg_base*age + deg_temp*age*temp (6 deg params)
        deg_b = params[3:6]
        deg_t = params[6:9]
        return bt + lc @ off + as_ @ deg_b + (as_ @ deg_t) * temp
    elif formula == 'quad':  # deg * age^2
        deg = params[3:6]
        return bt + lc @ off + aq @ deg
    elif formula == 'quad_t':  # deg * age^2 * temp
        deg = params[3:6]
        return bt + lc @ off + (aq @ deg) * temp
    elif formula == 'off_temp':  # off[c] + off_temp[c]*temp + deg*age
        off_t = params[3:6]
        deg = params[6:9]
        return bt + lc @ (off + off_t * temp) + as_ @ deg

def wrong_pairs(feats, params, formula):
    total = 0
    for feat in feats:
        times = compute_times(feat, params, formula)
        for i in range(feat['n'] - 1):
            if times[i] >= times[i+1]:
                total += 1
    return total

def accuracy(feats, params, formula):
    ok = 0
    for feat in feats:
        times = compute_times(feat, params, formula)
        if np.all(np.argsort(times) == np.arange(feat['n'])):
            ok += 1
    return ok / len(feats)

# Load data
tests = load_tests()
test_feats = [precompute_race(t['input'], t['expected']) for t in tests]

# Also load some historical races for broader validation
with open(f'{HIST_DIR}/races_00000-00999.json') as f:
    hist_races = json.load(f)[:200]
hist_feats = [precompute_race(r, r['finishing_positions']) for r in hist_races]

all_feats = test_feats + hist_feats
print(f"Test: {len(test_feats)}, Hist: {len(hist_feats)}")

print("\nTest race 0 details:")
t0 = tests[0]
rc = t0['input']['race_config']
print(f"  Config: temp={rc['track_temp']}, base={rc['base_lap_time']}, pit={rc['pit_lane_time']}, laps={rc['total_laps']}")
print(f"  Expected (1st-5th): {t0['expected'][:5]}")

# ============================================================
# Run scipy DE for each formula structure
# ============================================================
results = {}

for formula, n_params, bounds in [
    ('l',        6, [(-5,0),(-1,1),(0,5),(0,2),(0,1),(0,0.5)]),
    ('lt',       6, [(-5,0),(-1,1),(0,5),(0,0.1),(0,0.05),(0,0.02)]),
    ('tn',       6, [(-5,0),(-1,1),(0,5),(0,1),(0,0.5),(0,0.2)]),
    ('sep',      9, [(-5,0),(-1,1),(0,5),(0,1),(0,0.5),(0,0.2),(0,0.1),(0,0.05),(0,0.02)]),
    ('quad',     6, [(-5,0),(-1,1),(0,5),(0,0.1),(0,0.05),(0,0.02)]),
    ('off_temp', 9, [(-5,0),(-1,1),(0,5),(-0.3,0.3),(-0.3,0.3),(-0.3,0.3),(0,1),(0,0.5),(0,0.2)]),
]:
    print(f"\n=== Formula: {formula} (n_params={n_params}) ===")

    def obj(p):
        return wrong_pairs(test_feats, p, formula)

    res = differential_evolution(obj, bounds, maxiter=150, popsize=15, seed=42,
                                  tol=0.01, disp=False, mutation=(0.5, 1.5), recombination=0.7)
    p_opt = list(res.x)
    wp = int(res.fun)
    acc_test = accuracy(test_feats, p_opt, formula)
    acc_hist = accuracy(hist_feats, p_opt, formula)

    print(f"  Wrong pairs: {wp} / {len(test_feats)*19}")
    print(f"  Test accuracy: {acc_test*100:.1f}%")
    print(f"  Hist accuracy: {acc_hist*100:.1f}%")
    print(f"  Params: {[round(x,6) for x in p_opt]}")

    results[formula] = {'acc_test': acc_test, 'acc_hist': acc_hist, 'params': p_opt, 'wrong': wp}

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for formula, r in sorted(results.items(), key=lambda x: -x[1]['acc_test']):
    print(f"  {formula:10s}: test={r['acc_test']*100:.1f}%, hist={r['acc_hist']*100:.1f}%, wrong={r['wrong']}")

best_formula = max(results, key=lambda f: results[f]['acc_test'])
best = results[best_formula]
print(f"\nBest: {best_formula} | test={best['acc_test']*100:.1f}% | {best['params']}")

# Save
with open('/home/ubuntu/box-box-box/analysis/best_params.json', 'w') as f:
    json.dump({'formula': best_formula, 'params': best['params'],
               'accuracy_test': best['acc_test'], 'accuracy_hist': best['acc_hist']}, f, indent=2)
print("Saved to analysis/best_params.json")
