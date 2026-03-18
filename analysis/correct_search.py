#!/usr/bin/env python3
"""
Corrected parameter search with proper ranges.
Key insight: deg_S needs to be MUCH larger (0.1-0.8 range).
Also: tiebreaker = lower driver ID number wins.
"""
import json, numpy as np, time
from itertools import product as iprod

DATA = '/home/ubuntu/box-box-box/data/historical_races/races_00000-00999.json'
print("Loading...")
with open(DATA) as f:
    races = json.load(f)
N = 300

t2i = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}

def precompute(races_list):
    race_lc, race_as, race_bt, race_t = [], [], [], []
    for race in races_list:
        rc = race['race_config']
        base, pit_t, temp, total_laps = rc['base_lap_time'], rc['pit_lane_time'], rc['track_temp'], rc['total_laps']
        actual = race['finishing_positions']
        # Sort drivers by actual finishing position
        driver_order = {d: i for i, d in enumerate(actual)}
        n = len(actual)
        lc = np.zeros((n, 3)); as_ = np.zeros((n, 3)); bt = np.zeros(n)
        for s in race['strategies'].values():
            did = s['driver_id']
            idx = driver_order[did]
            pm = {p['lap']: t2i[p['to_tire']] for p in s.get('pit_stops', [])}
            ti = t2i[s['starting_tire']]; age = 0; n_pits = 0
            for lap in range(1, total_laps + 1):
                age += 1
                lc[idx, ti] += 1; as_[idx, ti] += age
                if lap in pm: ti = pm[lap]; age = 0; n_pits += 1
            bt[idx] = base * total_laps + n_pits * pit_t
        race_lc.append(lc); race_as.append(as_); race_bt.append(bt); race_t.append(temp)
    return np.stack(race_lc), np.stack(race_as), np.stack(race_bt), np.array(race_t)

print(f"Precomputing {N} races...")
lc_arr, as_arr, bt_arr, t_arr = precompute(races[:N])
print(f"Done. Arrays: {lc_arr.shape}")

exp_order = np.tile(np.arange(20), (N, 1))

def batch_score(off_b, deg_b, formula='l'):
    K = off_b.shape[0]
    CHUNK = 200
    ok = np.zeros(K, dtype=np.int32)
    for s in range(0, K, CHUNK):
        e = min(s+CHUNK, K)
        oc, dc = off_b[s:e], deg_b[s:e]
        C = e - s
        off_c = np.einsum('ndc,kc->ndk', lc_arr, oc)
        if formula == 'l':
            deg_c = np.einsum('ndc,kc->ndk', as_arr, dc)
        elif formula == 'lt':
            deg_c = np.einsum('ndc,kc->ndk', as_arr, dc) * t_arr[:, np.newaxis, np.newaxis]
        elif formula == 'tn':
            deg_c = np.einsum('ndc,kc->ndk', as_arr, dc) * (t_arr/30.0)[:, np.newaxis, np.newaxis]
        times = bt_arr[:, :, np.newaxis] + off_c + deg_c
        pred = np.argsort(times, axis=1)
        ok[s:e] = np.all(pred == exp_order[:, :, np.newaxis], axis=1).sum(axis=0)
    return ok / N

# ============================================================
# Search with CORRECT parameter ranges
# Key insight: deg_S needs to be 0.1-0.8 for 'l' formula
# ============================================================
print("\n=== Formula l (no temp): extended deg range ===")

so_v = np.array([-0.5, -1.0, -1.5, -2.0, -3.0])
ho_v = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
ds_v = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8])
dm_v = np.array([0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15])
dh_v = np.array([0.005, 0.01, 0.015, 0.02, 0.03, 0.05])

combos = list(iprod(so_v, ho_v, ds_v, dm_v, dh_v))
print(f"Testing {len(combos)} combinations...")
off_b = np.array([[so, 0.0, ho] for so, ho, *_ in combos])
deg_b = np.array([[sd, md, hd] for _, _, sd, md, hd in combos])

t0 = time.time()
scores = batch_score(off_b, deg_b, 'l')
t1 = time.time()

top_idx = np.argsort(scores)[-10:][::-1]
print(f"Time: {t1-t0:.1f}s")
print("Top 10:")
for i in top_idx:
    print(f"  {scores[i]*100:.1f}% | off=[{combos[i][0]},0,{combos[i][1]}] deg=[{combos[i][2]},{combos[i][3]},{combos[i][4]}]")

best_l = scores[top_idx[0]]
best_combo_l = combos[top_idx[0]]

# ============================================================
# Search lt formula with adjusted scale (temp factor)
# ============================================================
print("\n=== Formula lt (deg*age*temp): extended range ===")

ds_lt = np.array([0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020, 0.025, 0.030])
dm_lt = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.007])
dh_lt = np.array([0.0003, 0.0005, 0.001, 0.0015, 0.002])

combos_lt = list(iprod(so_v, ho_v, ds_lt, dm_lt, dh_lt))
off_lt = np.array([[so, 0.0, ho] for so, ho, *_ in combos_lt])
deg_lt = np.array([[sd, md, hd] for _, _, sd, md, hd in combos_lt])
print(f"Testing {len(combos_lt)} combinations...")
t0 = time.time()
scores_lt = batch_score(off_lt, deg_lt, 'lt')
t1 = time.time()

top_lt = np.argsort(scores_lt)[-10:][::-1]
print(f"Time: {t1-t0:.1f}s")
print("Top 10:")
for i in top_lt:
    print(f"  {scores_lt[i]*100:.1f}% | off=[{combos_lt[i][0]},0,{combos_lt[i][1]}] deg=[{combos_lt[i][2]},{combos_lt[i][3]},{combos_lt[i][4]}]")

best_lt = scores_lt[top_lt[0]]
best_combo_lt = combos_lt[top_lt[0]]

# ============================================================
# Fine search around best
# ============================================================
best_overall = max(best_l, best_lt)
print(f"\n=== Best overall: {best_overall*100:.1f}% ===")
if best_l >= best_lt:
    print(f"Best formula: l | {best_combo_l}")
    so, ho, sd, md, hd = best_combo_l
    so_f = np.linspace(so-0.3, so+0.3, 13)
    ho_f = np.linspace(max(0.1, ho-0.3), ho+0.3, 13)
    sd_f = np.linspace(max(0.01, sd*0.7), sd*1.3, 13)
    md_f = np.linspace(max(0.005, md*0.7), md*1.3, 9)
    hd_f = np.linspace(max(0.001, hd*0.7), hd*1.3, 9)
    combos_f = list(iprod(so_f, ho_f, sd_f, md_f, hd_f))
    print(f"Fine search: {len(combos_f)} combos...")
    off_f = np.array([[s, 0.0, h] for s, h, *_ in combos_f])
    deg_f = np.array([[s, m, h] for _, _, s, m, h in combos_f])
    scores_f = batch_score(off_f, deg_f, 'l')
    best_fi = np.argmax(scores_f)
    print(f"Fine best: {scores_f[best_fi]*100:.1f}% | {combos_f[best_fi]}")
    best_p = list(combos_f[best_fi])
    formula = 'l'
else:
    print(f"Best formula: lt | {best_combo_lt}")
    so, ho, sd, md, hd = best_combo_lt
    so_f = np.linspace(so-0.3, so+0.3, 13)
    ho_f = np.linspace(max(0.1, ho-0.3), ho+0.3, 13)
    sd_f = np.linspace(max(0.001, sd*0.7), sd*1.3, 13)
    md_f = np.linspace(max(0.0005, md*0.7), md*1.3, 9)
    hd_f = np.linspace(max(0.0001, hd*0.7), hd*1.3, 9)
    combos_f = list(iprod(so_f, ho_f, sd_f, md_f, hd_f))
    print(f"Fine search: {len(combos_f)} combos...")
    off_f = np.array([[s, 0.0, h] for s, h, *_ in combos_f])
    deg_f = np.array([[s, m, h] for _, _, s, m, h in combos_f])
    scores_f = batch_score(off_f, deg_f, 'lt')
    best_fi = np.argmax(scores_f)
    print(f"Fine best: {scores_f[best_fi]*100:.1f}% | {combos_f[best_fi]}")
    best_p = list(combos_f[best_fi])
    formula = 'lt'

# ============================================================
# Scipy DE refinement
# ============================================================
print("\n=== Scipy DE on full 300 races ===")
from scipy.optimize import differential_evolution

def obj(p):
    """Count wrong adjacent pairs."""
    o = np.array([[p[0], p[1], p[2]]])
    d = np.array([[p[3], p[4], p[5]]])
    off_c = np.einsum('ndc,kc->ndk', lc_arr, o)
    if formula == 'l':
        deg_c = np.einsum('ndc,kc->ndk', as_arr, d)
    elif formula == 'lt':
        deg_c = np.einsum('ndc,kc->ndk', as_arr, d) * t_arr[:, np.newaxis, np.newaxis]
    times = bt_arr[:, :, np.newaxis] + off_c + deg_c  # (N, 20, 1)
    times = times[:, :, 0]  # (N, 20)
    # Count wrong pairs
    wrong = 0
    for n in range(N):
        t = times[n]
        for i in range(19):
            if t[i] >= t[i+1]:
                wrong += 1
    return wrong

# Start from best found
x0 = best_p
if formula == 'l':
    bounds = [(x0[0]-1, x0[0]+1), (-0.5, 0.5), (x0[2]-1, x0[2]+1),
              (max(0.01, x0[3]*0.5), x0[3]*2), (max(0.005, x0[4]*0.5), x0[4]*2),
              (max(0.001, x0[5]*0.5), x0[5]*2)]
else:
    bounds = [(x0[0]-1, x0[0]+1), (-0.5, 0.5), (x0[2]-1, x0[2]+1),
              (max(0.001, x0[3]*0.5), x0[3]*2), (max(0.0005, x0[4]*0.5), x0[4]*2),
              (max(0.0001, x0[5]*0.5), x0[5]*2)]

res = differential_evolution(obj, bounds, maxiter=100, popsize=15, seed=42, tol=0.01, disp=True, x0=x0)
p_opt = list(res.x)
print(f"\nDE result: {p_opt}")

# Compute accuracy
o = np.array([[p_opt[0], p_opt[1], p_opt[2]]])
d = np.array([[p_opt[3], p_opt[4], p_opt[5]]])
final_acc = batch_score(o, d, formula)
print(f"Accuracy on {N} races: {final_acc[0]*100:.1f}%")
print(f"Formula: {formula}")
print(f"Params: off_S={p_opt[0]:.4f}, off_M={p_opt[1]:.4f}, off_H={p_opt[2]:.4f}")
print(f"        deg_S={p_opt[3]:.6f}, deg_M={p_opt[4]:.6f}, deg_H={p_opt[5]:.6f}")
