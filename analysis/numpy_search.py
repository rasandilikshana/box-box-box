#!/usr/bin/env python3
"""
Fully vectorized numpy search - test thousands of parameter combinations simultaneously.
For each race, we vectorize over all param combinations at once.
"""
import json, numpy as np, time

DATA = '/home/ubuntu/box-box-box/data/historical_races/races_00000-00999.json'

print("Loading data...")
with open(DATA) as f:
    races = json.load(f)

N_RACES = 200
races = races[:N_RACES]
print(f"Using {len(races)} races")

# Pre-compute features
# For each race: store (lc, as_, aq, base_t) as numpy arrays
# lc: (20, 3), as_: (20, 3), aq: (20, 3), base_t: (20,)
# Drivers are stored IN ACTUAL FINISHING ORDER (so correct order = 0,1,...,19)

print("Precomputing features...")
t2i = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}

race_lc = []   # list of (20,3) arrays
race_as = []   # list of (20,3) arrays
race_bt = []   # list of (20,) base times
race_t  = []   # track temps

for race in races:
    rc = race['race_config']
    base = rc['base_lap_time']
    pit_t = rc['pit_lane_time']
    temp = rc['track_temp']
    total_laps = rc['total_laps']
    actual = race['finishing_positions']
    driver_order = {d: i for i, d in enumerate(actual)}
    n = len(actual)

    lc = np.zeros((n, 3))
    as_ = np.zeros((n, 3))
    bt = np.zeros(n)

    for s in race['strategies'].values():
        did = s['driver_id']
        idx = driver_order[did]
        pit_map = {p['lap']: t2i[p['to_tire']] for p in s.get('pit_stops', [])}
        tire_i = t2i[s['starting_tire']]
        age = 0
        n_pits = 0

        for lap in range(1, total_laps + 1):
            age += 1
            lc[idx, tire_i] += 1
            as_[idx, tire_i] += age
            if lap in pit_map:
                tire_i = pit_map[lap]
                age = 0
                n_pits += 1

        bt[idx] = base * total_laps + n_pits * pit_t

    race_lc.append(lc)
    race_as.append(as_)
    race_bt.append(bt)
    race_t.append(temp)

# Convert to numpy arrays
# race_lc_arr: (N_RACES, 20, 3)
# race_as_arr: (N_RACES, 20, 3)
# race_bt_arr: (N_RACES, 20)
# race_t_arr:  (N_RACES,)
race_lc_arr = np.stack(race_lc)  # (N, 20, 3)
race_as_arr = np.stack(race_as)  # (N, 20, 3)
race_bt_arr = np.stack(race_bt)  # (N, 20)
race_t_arr  = np.array(race_t)   # (N,)

expected_order = np.tile(np.arange(20), (N_RACES, 1))  # (N, 20) - all should be [0,1,...,19]

print(f"Feature arrays: {race_lc_arr.shape}, {race_as_arr.shape}")

def batch_score(off_batch, deg_batch, formula='lt'):
    """
    off_batch: (K, 3)
    deg_batch: (K, 3)
    Returns: (K,) array of accuracy scores (fraction of races correct)

    formula: 'lt' = deg*age*temp, 'l' = deg*age, 'tn' = deg*age*temp/30
    """
    K = off_batch.shape[0]

    # Compute time for each race and each param combo
    # race_lc_arr: (N, 20, 3)
    # off_batch: (K, 3)
    # offset contribution: (N, 20, 3) @ (3, K) → but we want sum over compounds
    # = einsum('ndc,kc->ndk', race_lc_arr, off_batch)
    # then sum... actually: (N, 20, 3) @ (K, 3).T doesn't give right shape

    # Better: sum_c[lc[n,d,c] * off_batch[k,c]]
    # = np.einsum('ndc,kc->ndk', race_lc_arr, off_batch)  → shape (N, 20, K), sum over... no
    # Actually: for offset contrib:
    # off_contrib[n, d, k] = sum_c lc[n,d,c] * off[k,c]
    # = np.tensordot(race_lc_arr, off_batch, axes=[[2],[1]])  → (N, 20, K)

    # To avoid memory issues, process in chunks of K
    CHUNK = 500

    correct_counts = np.zeros(K, dtype=np.int32)

    for start in range(0, K, CHUNK):
        end = min(start + CHUNK, K)
        o_chunk = off_batch[start:end]  # (C, 3)
        d_chunk = deg_batch[start:end]  # (C, 3)
        C = end - start

        # off_contrib: (N, 20, C) = race_lc_arr @ o_chunk.T
        # Using einsum
        off_c = np.einsum('ndc,kc->ndk', race_lc_arr, o_chunk)  # (N, 20, C)

        # deg_contrib: depends on formula
        if formula == 'lt':
            # temp * (race_as_arr @ d_chunk.T)
            deg_c = np.einsum('ndc,kc->ndk', race_as_arr, d_chunk)  # (N, 20, C)
            # multiply by temp: race_t_arr (N,) → need to broadcast
            deg_c = deg_c * race_t_arr[:, np.newaxis, np.newaxis]  # (N, 20, C)
        elif formula == 'l':
            deg_c = np.einsum('ndc,kc->ndk', race_as_arr, d_chunk)
        elif formula == 'tn':
            deg_c = np.einsum('ndc,kc->ndk', race_as_arr, d_chunk)
            deg_c = deg_c * (race_t_arr / 30.0)[:, np.newaxis, np.newaxis]

        # total time: race_bt_arr (N, 20) + off_c + deg_c
        # times (N, 20, C) = base (N, 20, 1) + off_c + deg_c
        times = race_bt_arr[:, :, np.newaxis] + off_c + deg_c  # (N, 20, C)

        # For each race and param combo, argsort times and check if it equals [0,1,...,19]
        pred = np.argsort(times, axis=1)  # (N, 20, C)
        # Check each: pred[n, :, k] == [0,1,...,19]
        correct = np.all(pred == expected_order[:, :, np.newaxis], axis=1)  # (N, C)
        correct_counts[start:end] = correct.sum(axis=0)

    return correct_counts / N_RACES

print("Starting parameter search...\n")

# ============================================================
# Formula lt: deg * age * temp
# Test wider range of parameters
# ============================================================
print("=== Formula: deg*age*temp (lt) ===")

off_S_vals = np.array([-0.5, -1.0, -1.5, -2.0, -3.0])
off_H_vals = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
deg_S_vals = np.array([0.003, 0.005, 0.007, 0.010, 0.012, 0.015, 0.020])
deg_M_vals = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
deg_H_vals = np.array([0.0003, 0.0005, 0.001, 0.0015, 0.002])

# Build all combinations
from itertools import product as iprod
combos = list(iprod(off_S_vals, off_H_vals, deg_S_vals, deg_M_vals, deg_H_vals))
print(f"Testing {len(combos)} combinations...")

off_b = np.array([[so, 0.0, ho] for so, ho, *_ in combos])
deg_b = np.array([[sd, md, hd] for _, _, sd, md, hd in combos])

t0 = time.time()
scores = batch_score(off_b, deg_b, 'lt')
t1 = time.time()

idx_best = np.argmax(scores)
print(f"Time: {t1-t0:.1f}s")
print(f"Best lt: {scores[idx_best]*100:.1f}% | {combos[idx_best]}")
print(f"Top 5:")
top5 = np.argsort(scores)[-5:][::-1]
for i in top5:
    print(f"  {scores[i]*100:.1f}% | {combos[i]}")

# ============================================================
# Formula l: deg * age (no temp)
# ============================================================
print("\n=== Formula: deg*age (no temp, l) ===")

deg_S_l = np.array([0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20])
deg_M_l = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
deg_H_l = np.array([0.005, 0.008, 0.01, 0.015])

combos_l = list(iprod(off_S_vals, off_H_vals, deg_S_l, deg_M_l, deg_H_l))
off_bl = np.array([[so, 0.0, ho] for so, ho, *_ in combos_l])
deg_bl = np.array([[sd, md, hd] for _, _, sd, md, hd in combos_l])

t0 = time.time()
scores_l = batch_score(off_bl, deg_bl, 'l')
t1 = time.time()

idx_bl = np.argmax(scores_l)
print(f"Time: {t1-t0:.1f}s")
print(f"Best l: {scores_l[idx_bl]*100:.1f}% | {combos_l[idx_bl]}")

# ============================================================
# Formula tn: deg * age * temp/30
# ============================================================
print("\n=== Formula: deg*age*temp/30 (tn) ===")

deg_S_n = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
deg_M_n = np.array([0.02, 0.03, 0.05, 0.07, 0.1])
deg_H_n = np.array([0.01, 0.015, 0.02, 0.03])

combos_n = list(iprod(off_S_vals, off_H_vals, deg_S_n, deg_M_n, deg_H_n))
off_bn = np.array([[so, 0.0, ho] for so, ho, *_ in combos_n])
deg_bn = np.array([[sd, md, hd] for _, _, sd, md, hd in combos_n])

t0 = time.time()
scores_n = batch_score(off_bn, deg_bn, 'tn')
t1 = time.time()

idx_bn = np.argmax(scores_n)
print(f"Time: {t1-t0:.1f}s")
print(f"Best tn: {scores_n[idx_bn]*100:.1f}% | {combos_n[idx_bn]}")

# ============================================================
# Summary
# ============================================================
print("\n=== SUMMARY ===")
results = [
    ('lt', scores[idx_best], combos[idx_best]),
    ('l', scores_l[idx_bl], combos_l[idx_bl]),
    ('tn', scores_n[idx_bn], combos_n[idx_bn]),
]
results.sort(key=lambda x: -x[1])
for name, acc, params in results:
    print(f"  {name}: {acc*100:.1f}% | {params}")

# Fine search around best
best_name, best_acc, best_params = results[0]
print(f"\nBest formula: {best_name} ({best_acc*100:.1f}%)")

if best_name == 'lt':
    so, ho, sd, md, hd = best_params
    # Fine grid
    so_f = np.linspace(so - 0.5, so + 0.5, 11)
    ho_f = np.linspace(ho - 0.5, ho + 0.5, 11)
    sd_f = np.linspace(max(0.001, sd * 0.5), sd * 1.5, 11)
    md_f = np.linspace(max(0.0005, md * 0.5), md * 1.5, 11)
    hd_f = np.linspace(max(0.0001, hd * 0.5), hd * 1.5, 11)

    combos_f = list(iprod(so_f, ho_f, sd_f, md_f, hd_f))
    print(f"Fine search: {len(combos_f)} combos...")
    off_f = np.array([[s, 0.0, h] for s, h, *_ in combos_f])
    deg_f = np.array([[s, m, h] for _, _, s, m, h in combos_f])
    scores_f = batch_score(off_f, deg_f, 'lt')
    idx_f = np.argmax(scores_f)
    print(f"Fine best: {scores_f[idx_f]*100:.1f}% | {combos_f[idx_f]}")

    # Validate on all 1000 races
    print("\nValidating on all 1000 races...")
    with open(DATA) as ff:
        all_races = json.load(ff)
    # Add more races
    for race in all_races[N_RACES:]:
        rc = race['race_config']
        base = rc['base_lap_time']
        pit_t = rc['pit_lane_time']
        temp = rc['track_temp']
        total_laps = rc['total_laps']
        actual = race['finishing_positions']
        driver_order = {d: i for i, d in enumerate(actual)}
        n = len(actual)

        lc = np.zeros((n, 3))
        as_ = np.zeros((n, 3))
        bt = np.zeros(n)

        for s in race['strategies'].values():
            did = s['driver_id']
            idx = driver_order[did]
            pit_map = {p['lap']: t2i[p['to_tire']] for p in s.get('pit_stops', [])}
            tire_i = t2i[s['starting_tire']]
            age = 0
            n_pits = 0
            for lap in range(1, total_laps + 1):
                age += 1
                lc[idx, tire_i] += 1
                as_[idx, tire_i] += age
                if lap in pit_map:
                    tire_i = pit_map[lap]
                    age = 0
                    n_pits += 1
            bt[idx] = base * total_laps + n_pits * pit_t

        race_lc.append(lc)
        race_as.append(as_)
        race_bt.append(bt)
        race_t.append(temp)

    race_lc_arr = np.stack(race_lc)
    race_as_arr = np.stack(race_as)
    race_bt_arr = np.stack(race_bt)
    race_t_arr = np.array(race_t)
    expected_order = np.tile(np.arange(20), (len(race_lc), 1))
    N_RACES_ALL = len(race_lc)

    # Temporarily update N_RACES for the scoring function
    # Score best params on all 1000
    best_combo = combos_f[idx_f]
    so2, ho2, sd2, md2, hd2 = best_combo
    p_off = np.array([[so2, 0.0, ho2]])
    p_deg = np.array([[sd2, md2, hd2]])

    def score_single(p_o, p_d, formula, n_races):
        global N_RACES, expected_order
        old = N_RACES
        N_RACES = n_races
        # Need to recalculate expected_order for this
        exp = np.tile(np.arange(20), (n_races, 1))
        # Use einsum directly
        off_c = np.einsum('ndc,kc->ndk', race_lc_arr[:n_races], p_o)
        if formula == 'lt':
            deg_c = np.einsum('ndc,kc->ndk', race_as_arr[:n_races], p_d)
            deg_c = deg_c * race_t_arr[:n_races, np.newaxis, np.newaxis]
        times = race_bt_arr[:n_races, :, np.newaxis] + off_c + deg_c
        pred = np.argsort(times, axis=1)
        correct = np.all(pred == exp[:, :, np.newaxis], axis=1)
        N_RACES = old
        return correct.sum() / n_races

    acc_all = score_single(p_off, p_deg, 'lt', len(race_lc))
    print(f"All 1000 races accuracy: {acc_all*100:.1f}%")
