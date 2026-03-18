#!/usr/bin/env python3
"""Fast formula check using pre-computed race features."""
import json, numpy as np, sys

DATA = '/home/ubuntu/box-box-box/data/historical_races/races_00000-00999.json'

def load_and_precompute(path, n=200):
    with open(path) as f:
        races = json.load(f)[:n]

    processed = []
    for r in races:
        rc = r['race_config']
        base = rc['base_lap_time']
        pit_t = rc['pit_lane_time']
        temp = rc['track_temp']
        total_laps = rc['total_laps']
        actual = r['finishing_positions']

        driver_feats = {}
        for s in r['strategies'].values():
            did = s['driver_id']
            pit_map = {p['lap']: p['to_tire'] for p in s.get('pit_stops', [])}
            tire = s['starting_tire']
            age = 0
            n_pits = 0
            # Per compound: lap_count, age_sum, age_sq_sum
            lc = {'S': 0, 'M': 0, 'H': 0}  # lap count
            as_ = {'S': 0, 'M': 0, 'H': 0}  # age sum
            aq = {'S': 0, 'M': 0, 'H': 0}  # age^2 sum
            t2c = {'SOFT': 'S', 'MEDIUM': 'M', 'HARD': 'H'}

            for lap in range(1, total_laps + 1):
                age += 1
                c = t2c[tire]
                lc[c] += 1
                as_[c] += age
                aq[c] += age * age
                if lap in pit_map:
                    tire = pit_map[lap]
                    age = 0
                    n_pits += 1

            driver_feats[did] = {
                'lc': lc, 'as': as_, 'aq': aq,
                'n_pits': n_pits,
                'base_time': base * total_laps + n_pits * pit_t  # fixed component
            }

        processed.append({
            'temp': temp,
            'pit_t': pit_t,
            'base': base,
            'total_laps': total_laps,
            'feats': driver_feats,
            'actual': actual,
        })
    return processed

def score_all(races_proc, off_S, off_M, off_H, deg_S, deg_M, deg_H, deg_fn):
    """
    deg_fn: function(agesum, agesq, temp, deg_rate) -> degradation contribution
    """
    correct = 0
    for r in races_proc:
        temp = r['temp']
        times = {}
        for did, f in r['feats'].items():
            lc, as_, aq = f['lc'], f['as'], f['aq']
            t = f['base_time']
            t += off_S * lc['S'] + off_M * lc['M'] + off_H * lc['H']
            t += deg_fn(as_['S'], aq['S'], temp, deg_S)
            t += deg_fn(as_['M'], aq['M'], temp, deg_M)
            t += deg_fn(as_['H'], aq['H'], temp, deg_H)
            times[did] = t
        pred = sorted(times, key=lambda d: times[d])
        if pred == r['actual']:
            correct += 1
    return correct / len(races_proc)

print("Loading and precomputing...")
races = load_and_precompute(DATA, 500)
print(f"Loaded {len(races)} races")

# Quick print of race 0
r = races[0]
print(f"\nRace 0: temp={r['temp']}, base={r['base']}, pit={r['pit_t']}, laps={r['total_laps']}")
print(f"Actual: {r['actual']}")
print("Sample driver features:")
for did in r['actual'][:3]:
    f = r['feats'][did]
    print(f"  {did}: lc={f['lc']}, as={f['as']}, pits={f['n_pits']}, base_time={f['base_time']:.1f}")

print("\n=== Testing formula structures ===\n")

# FORMULA A: deg * age_sum * temp (linear in age, linear in temp)
# deg contribution = deg_rate * sum(ages) * temp
def fa(agesum, agesq, temp, rate):
    return rate * agesum * temp

# FORMULA B: deg * age_sum (no temp)
def fb(agesum, agesq, temp, rate):
    return rate * agesum

# FORMULA C: deg * age^2 sum (quadratic in age)
def fc(agesum, agesq, temp, rate):
    return rate * agesq

# FORMULA D: deg * age_sum * (temp/30) normalized
def fd(agesum, agesq, temp, rate):
    return rate * agesum * (temp / 30.0)

# FORMULA E: deg * age_sum * (1 + 0.01*temp) additive temp
def fe_factory(alpha):
    def fe(agesum, agesq, temp, rate):
        return rate * agesum * (1 + alpha * temp)
    return fe

# For each formula, do grid search
best_results = {}
sample = races[:100]

print("Testing F_A (agesum * temp)...")
bA, pA = 0, None
for so in [-0.5, -1.0, -1.5, -2.0, -3.0]:
    for ho in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for sd in [0.001, 0.002, 0.003, 0.005]:
            for md in [0.0005, 0.001, 0.0015, 0.002]:
                for hd in [0.0002, 0.0005, 0.001]:
                    acc = score_all(sample, so, 0.0, ho, sd, md, hd, fa)
                    if acc > bA:
                        bA, pA = acc, [so, 0, ho, sd, md, hd]
print(f"  Best FA: {bA*100:.1f}% {pA}")
best_results['FA (agesum*temp)'] = (bA, pA)

print("Testing F_B (agesum, no temp)...")
bB, pB = 0, None
for so in [-0.3, -0.5, -1.0, -1.5, -2.0]:
    for ho in [0.3, 0.5, 1.0, 1.5, 2.0]:
        for sd in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
            for md in [0.005, 0.01, 0.02, 0.03, 0.05]:
                for hd in [0.002, 0.005, 0.01, 0.015]:
                    acc = score_all(sample, so, 0.0, ho, sd, md, hd, fb)
                    if acc > bB:
                        bB, pB = acc, [so, 0, ho, sd, md, hd]
print(f"  Best FB: {bB*100:.1f}% {pB}")
best_results['FB (agesum, no temp)'] = (bB, pB)

print("Testing F_C (age^2 sum)...")
bC, pC = 0, None
for so in [-0.3, -0.5, -1.0, -1.5]:
    for ho in [0.3, 0.5, 1.0, 1.5]:
        for sd in [0.001, 0.003, 0.005, 0.01, 0.02]:
            for md in [0.0005, 0.001, 0.003, 0.005]:
                for hd in [0.0002, 0.0005, 0.001, 0.002]:
                    acc = score_all(sample, so, 0.0, ho, sd, md, hd, fc)
                    if acc > bC:
                        bC, pC = acc, [so, 0, ho, sd, md, hd]
print(f"  Best FC: {bC*100:.1f}% {pC}")
best_results['FC (age^2)'] = (bC, pC)

print("Testing F_D (agesum * temp/30)...")
bD, pD = 0, None
for so in [-0.3, -0.5, -1.0, -1.5, -2.0]:
    for ho in [0.3, 0.5, 1.0, 1.5, 2.0]:
        for sd in [0.01, 0.02, 0.03, 0.05, 0.1]:
            for md in [0.005, 0.01, 0.02, 0.03]:
                for hd in [0.002, 0.005, 0.01, 0.015]:
                    acc = score_all(sample, so, 0.0, ho, sd, md, hd, fd)
                    if acc > bD:
                        bD, pD = acc, [so, 0, ho, sd, md, hd]
print(f"  Best FD: {bD*100:.1f}% {pD}")
best_results['FD (agesum*temp/30)'] = (bD, pD)

print("\n=== RESULTS ===")
for name, (acc, params) in sorted(best_results.items(), key=lambda x: -x[1][0]):
    print(f"  {name}: {acc*100:.1f}% | {params}")

# Take best formula and refine with scipy
best_name = max(best_results, key=lambda k: best_results[k][0])
best_acc, best_p = best_results[best_name]
print(f"\nBest formula: {best_name} ({best_acc*100:.1f}%)")
print("Running scipy optimization on best formula...")

# Determine which formula to optimize
if 'FA' in best_name:
    deg_fn = fa
elif 'FB' in best_name:
    deg_fn = fb
elif 'FC' in best_name:
    deg_fn = fc
else:
    deg_fn = fd

from scipy.optimize import differential_evolution

def obj(p):
    wrong = 0
    for r in races[:200]:
        t_ = {}
        for did, f in r['feats'].items():
            lc, as_, aq = f['lc'], f['as'], f['aq']
            temp = r['temp']
            t = f['base_time']
            t += p[0]*lc['S'] + p[1]*lc['M'] + p[2]*lc['H']
            t += deg_fn(as_['S'], aq['S'], temp, p[3])
            t += deg_fn(as_['M'], aq['M'], temp, p[4])
            t += deg_fn(as_['H'], aq['H'], temp, p[5])
            t_[did] = t
        pred = sorted(t_, key=lambda d: t_[d])
        for i in range(len(r['actual'])-1):
            if t_[r['actual'][i]] >= t_[r['actual'][i+1]]:
                wrong += 1
    return wrong

bounds = [(-5,0),(-1,1),(0,5),(1e-5,1.0),(1e-5,0.5),(1e-5,0.2)]
res = differential_evolution(obj, bounds, maxiter=150, popsize=15, seed=42, tol=0.001, disp=True)
p_opt = list(res.x)
acc_opt = score_all(races[:200], *p_opt, deg_fn)
print(f"Scipy opt: {acc_opt*100:.1f}% | {p_opt}")

# Test on 500 races
acc500 = score_all(races[:500], *p_opt, deg_fn)
print(f"On 500 races: {acc500*100:.1f}%")

print(f"\nFINAL PARAMS: off_S={p_opt[0]:.4f} off_M={p_opt[1]:.4f} off_H={p_opt[2]:.4f} deg_S={p_opt[3]:.6f} deg_M={p_opt[4]:.6f} deg_H={p_opt[5]:.6f}")
print(f"Formula: lap_time = base + off[c] + {best_name} * deg[c]")
