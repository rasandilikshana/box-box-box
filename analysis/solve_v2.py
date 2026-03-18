#!/usr/bin/env python3
"""
Broad scipy optimization with all offsets free + wide bounds.
Also try constraining the sign of the degradation temp sensitivity.
"""
import json, numpy as np, glob, os
from scipy.optimize import differential_evolution

IN_DIR = '/home/ubuntu/box-box-box/data/test_cases/inputs'
OUT_DIR = '/home/ubuntu/box-box-box/data/test_cases/expected_outputs'
HIST_DIR = '/home/ubuntu/box-box-box/data/historical_races'
t2i = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}

def load_all():
    tests = []
    for fn in sorted(glob.glob(f'{IN_DIR}/test_*.json')):
        num = os.path.basename(fn).replace('test_','').replace('.json','')
        with open(fn) as f: inp = json.load(f)
        with open(f'{OUT_DIR}/test_{num}.json') as f: exp = json.load(f)
        tests.append({'rc': inp['race_config'], 'strats': inp['strategies'], 'expected': exp['finishing_positions']})
    return tests

def precompute(tests):
    feats = []
    for t in tests:
        rc = t['rc']; expected = t['expected']
        base, pit_t, temp, total_laps = rc['base_lap_time'], rc['pit_lane_time'], rc['track_temp'], rc['total_laps']
        driver_order = {d: i for i, d in enumerate(expected)}
        n = len(expected)
        lc = np.zeros((n,3)); as_ = np.zeros((n,3)); aq = np.zeros((n,3)); bt = np.zeros(n)
        for s in t['strats'].values():
            did = s['driver_id']; idx = driver_order[did]
            pm = {p['lap']: t2i[p['to_tire']] for p in s.get('pit_stops',[])}
            ti = t2i[s['starting_tire']]; age = 0; np_ = 0
            for lap in range(1, total_laps+1):
                age+=1; lc[idx,ti]+=1; as_[idx,ti]+=age; aq[idx,ti]+=age*age
                if lap in pm: ti=pm[lap]; age=0; np_+=1
            bt[idx] = base*total_laps + np_*pit_t
        feats.append({'lc':lc,'as':as_,'aq':aq,'bt':bt,'temp':temp,'n':n})
    return feats

def wrong_pairs_lt(feats, p):
    """Formula: base + off[c] + deg[c]*age*temp"""
    off = p[:3]; deg = p[3:6]
    total = 0
    for f in feats:
        times = f['bt'] + f['lc']@off + (f['as']@deg)*f['temp']
        for i in range(f['n']-1):
            if times[i] >= times[i+1]: total += 1
    return total

def wrong_pairs_l(feats, p):
    """Formula: base + off[c] + deg[c]*age"""
    off = p[:3]; deg = p[3:6]
    total = 0
    for f in feats:
        times = f['bt'] + f['lc']@off + f['as']@deg
        for i in range(f['n']-1):
            if times[i] >= times[i+1]: total += 1
    return total

def wrong_pairs_sep(feats, p):
    """Formula: base + off[c] + (deg_b[c] + deg_t[c]*temp)*age"""
    off = p[:3]; deg_b = p[3:6]; deg_t = p[6:9]
    total = 0
    for f in feats:
        t_ = f['temp']
        eff_deg = deg_b + deg_t * t_
        times = f['bt'] + f['lc']@off + f['as']@eff_deg
        for i in range(f['n']-1):
            if times[i] >= times[i+1]: total += 1
    return total

def accuracy(feats, fn, p):
    ok = 0
    for f in feats:
        times = fn(f, p)
        if np.all(np.argsort(times) == np.arange(f['n'])): ok += 1
    return ok / len(feats)

def time_lt(f, p):
    return f['bt'] + f['lc']@p[:3] + (f['as']@p[3:6])*f['temp']
def time_l(f, p):
    return f['bt'] + f['lc']@p[:3] + f['as']@p[3:6]
def time_sep(f, p):
    eff = p[3:6] + p[6:9]*f['temp']
    return f['bt'] + f['lc']@p[:3] + f['as']@eff

print("Loading...")
tests = load_all()
feats_test = precompute(tests)
print(f"Test cases: {len(feats_test)}")

# Load some historical races
with open(f'{HIST_DIR}/races_00000-00999.json') as f: hist = json.load(f)[:500]
hist_tests = [{'rc': r['race_config'], 'strats': r['strategies'], 'expected': r['finishing_positions']} for r in hist]
feats_hist = precompute(hist_tests)

all_feats = feats_test + feats_hist
print(f"Total: {len(all_feats)}")

def obj(p):
    return wrong_pairs_lt(all_feats, p)

def obj_l(p):
    return wrong_pairs_l(all_feats, p)

def obj_sep(p):
    return wrong_pairs_sep(all_feats, p)

# Very wide bounds, off_M free
# Formula lt: [off_S, off_M, off_H, deg_S, deg_M, deg_H]
bounds_lt = [(-10,2), (-5,5), (-2,10), (-0.1,0.3), (-0.05,0.15), (-0.02,0.06)]
# Formula l: [off_S, off_M, off_H, deg_S, deg_M, deg_H]
bounds_l  = [(-10,2), (-5,5), (-2,10), (-2,5), (-1,3), (-0.5,2)]
# Formula sep: [off_S, off_M, off_H, deg_b_S, deg_b_M, deg_b_H, deg_t_S, deg_t_M, deg_t_H]
bounds_sep = [(-10,2),(-5,5),(-2,10), (-2,5),(-1,3),(-0.5,2), (-0.1,0.3),(-0.05,0.15),(-0.02,0.06)]

print("\n=== Formula lt: deg*age*temp (wide bounds, off_M free) ===")
res_lt = differential_evolution(obj, bounds_lt, maxiter=300, popsize=20, seed=42,
                                 tol=0.001, disp=True, mutation=(0.5,1.5), recombination=0.7,
                                 workers=1)
p_lt = res_lt.x
print(f"Loss: {res_lt.fun}")
print(f"Params: {[round(x,5) for x in p_lt]}")
acc_test_lt = accuracy(feats_test, time_lt, p_lt)
acc_hist_lt = accuracy(feats_hist, time_lt, p_lt)
print(f"Test acc: {acc_test_lt*100:.1f}%, Hist acc: {acc_hist_lt*100:.1f}%")

print("\n=== Formula l: deg*age (wide bounds) ===")
res_l = differential_evolution(obj_l, bounds_l, maxiter=300, popsize=20, seed=42,
                                tol=0.001, disp=True, mutation=(0.5,1.5), recombination=0.7)
p_l = res_l.x
print(f"Loss: {res_l.fun}")
print(f"Params: {[round(x,5) for x in p_l]}")
acc_test_l = accuracy(feats_test, time_l, p_l)
acc_hist_l = accuracy(feats_hist, time_l, p_l)
print(f"Test acc: {acc_test_l*100:.1f}%, Hist acc: {acc_hist_l*100:.1f}%")

print("\n=== Formula sep: (deg_b + deg_t*temp)*age (9 params) ===")
res_sep = differential_evolution(obj_sep, bounds_sep, maxiter=300, popsize=20, seed=42,
                                  tol=0.001, disp=True, mutation=(0.5,1.5), recombination=0.7)
p_sep = res_sep.x
print(f"Loss: {res_sep.fun}")
print(f"Params: {[round(x,5) for x in p_sep]}")
acc_test_sep = accuracy(feats_test, time_sep, p_sep)
acc_hist_sep = accuracy(feats_hist, time_sep, p_sep)
print(f"Test acc: {acc_test_sep*100:.1f}%, Hist acc: {acc_hist_sep*100:.1f}%")

print("\n=== SUMMARY ===")
print(f"lt:  test={acc_test_lt*100:.1f}% hist={acc_hist_lt*100:.1f}%  loss={int(res_lt.fun)}")
print(f"l:   test={acc_test_l*100:.1f}%  hist={acc_hist_l*100:.1f}%  loss={int(res_l.fun)}")
print(f"sep: test={acc_test_sep*100:.1f}% hist={acc_hist_sep*100:.1f}%  loss={int(res_sep.fun)}")
