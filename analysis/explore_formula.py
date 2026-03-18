#!/usr/bin/env python3
"""
Direct formula exploration: look at specific races to find the formula structure.
"""
import json, os, glob, numpy as np
from itertools import combinations

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'historical_races')
TEST_INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'test_cases', 'inputs')
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'test_cases', 'expected_outputs')

def get_all_strategies(race):
    """Return list of (driver_id, strategy) tuples."""
    return [(s['driver_id'], s) for s in race['strategies'].values()]

def compute_driver_time(strategy, race_config, off, deg_fn):
    """
    off: dict {SOFT, MEDIUM, HARD} -> float (compound speed offset)
    deg_fn: function(compound, tire_age, track_temp) -> float (degradation per lap)
    """
    total_laps = race_config['total_laps']
    base = race_config['base_lap_time']
    pit = race_config['pit_lane_time']
    temp = race_config['track_temp']

    pit_map = {stop['lap']: stop['to_tire'] for stop in strategy.get('pit_stops', [])}
    current_tire = strategy['starting_tire']
    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        deg = deg_fn(current_tire, tire_age, temp)
        total_time += base + off[current_tire] + deg

        if lap in pit_map:
            total_time += pit
            current_tire = pit_map[lap]
            tire_age = 0

    return total_time

def test_formula(races, off, deg_fn, label):
    """Test a formula on a set of races."""
    correct = 0
    for race in races:
        rc = race['race_config']
        times = {}
        for did, strat in get_all_strategies(race):
            times[did] = compute_driver_time(strat, rc, off, deg_fn)
        predicted = sorted(times.keys(), key=lambda d: times[d])
        if predicted == race['finishing_positions']:
            correct += 1
    acc = correct / len(races)
    print(f"  {label}: {acc*100:.1f}%")
    return acc

def main():
    print("Loading race data...")
    with open(os.path.join(DATA_DIR, 'races_00000-00999.json')) as f:
        races = json.load(f)

    sample = races[:200]

    # ============================================================
    # Step 1: Print the first race to understand it manually
    # ============================================================
    r = races[0]
    rc = r['race_config']
    print(f"\nRace 0: {r['race_id']} - {rc['track']}")
    print(f"Config: laps={rc['total_laps']}, base={rc['base_lap_time']}, pit={rc['pit_lane_time']}, temp={rc['track_temp']}")
    print(f"Finishing order: {r['finishing_positions']}")
    print("\nAll driver strategies:")
    finishing = {d: i+1 for i, d in enumerate(r['finishing_positions'])}
    for pos_key, strat in sorted(r['strategies'].items()):
        did = strat['driver_id']
        pits = [(p['lap'], p['from_tire'], p['to_tire']) for p in strat['pit_stops']]
        print(f"  {pos_key} {did} pos={finishing[did]}: {strat['starting_tire']} → {pits}")

    # ============================================================
    # Step 2: Try different formula structures
    # ============================================================
    print("\n--- Testing Formula Structures ---")

    # STRUCTURE 1: Simple linear with temp multiplier on degradation
    # lap_time = base + off[c] + deg[c] * tire_age * temp
    for s_off in [-0.5, -1.0, -1.5, -2.0]:
        for h_off in [0.5, 1.0, 1.5, 2.0]:
            for s_deg in [0.001, 0.002, 0.005]:
                for m_deg in [0.0005, 0.001, 0.002]:
                    for h_deg in [0.0002, 0.0005, 0.001]:
                        off = {'SOFT': s_off, 'MEDIUM': 0.0, 'HARD': h_off}
                        def make_deg(sd, md, hd):
                            def deg_fn(c, age, temp):
                                rates = {'SOFT': sd, 'MEDIUM': md, 'HARD': hd}
                                return rates[c] * age * temp
                            return deg_fn
                        p = [s_off, 0, h_off, s_deg, m_deg, h_deg]
                        acc = sum(1 for race in sample[:100] if sorted(
                            [s['driver_id'] for s in race['strategies'].values()],
                            key=lambda d: compute_driver_time(
                                next(s for s in race['strategies'].values() if s['driver_id']==d),
                                race['race_config'], off, make_deg(s_deg, m_deg, h_deg)
                            )
                        ) == race['finishing_positions']) / 100
                        if acc > 0.5:
                            print(f"  STRUCT1 (off*age*temp): {p} → {acc*100:.0f}%")

    # STRUCTURE 2: lap_time = base + off[c] + deg[c] * tire_age (no temp!)
    print("\nTesting without temperature:")
    best_no_temp = 0
    best_no_temp_params = None
    for s_off in [-0.3, -0.5, -1.0, -1.5, -2.0, -3.0]:
        for h_off in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
            for s_deg in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
                for m_deg in [0.01, 0.02, 0.03, 0.05, 0.07]:
                    for h_deg in [0.005, 0.01, 0.015, 0.02, 0.03]:
                        off = {'SOFT': s_off, 'MEDIUM': 0.0, 'HARD': h_off}
                        def make_deg_notemp(sd, md, hd):
                            def deg_fn(c, age, temp):
                                return {'SOFT': sd, 'MEDIUM': md, 'HARD': hd}[c] * age
                            return deg_fn
                        acc = sum(1 for race in sample[:100] if sorted(
                            [s['driver_id'] for s in race['strategies'].values()],
                            key=lambda d: compute_driver_time(
                                next(s for s in race['strategies'].values() if s['driver_id']==d),
                                race['race_config'], off, make_deg_notemp(s_deg, m_deg, h_deg)
                            )
                        ) == race['finishing_positions']) / 100
                        if acc > best_no_temp:
                            best_no_temp = acc
                            best_no_temp_params = [s_off, 0, h_off, s_deg, m_deg, h_deg]
    print(f"  Best no-temp: {best_no_temp*100:.1f}% params={best_no_temp_params}")

    # STRUCTURE 3: lap_time = base + off[c] + deg[c] * tire_age^2
    print("\nTesting quadratic tire_age:")
    best_quad = 0
    best_quad_params = None
    for s_off in [-0.3, -0.5, -1.0, -1.5]:
        for h_off in [0.3, 0.5, 1.0, 1.5]:
            for s_deg in [0.001, 0.002, 0.005, 0.01]:
                for m_deg in [0.0005, 0.001, 0.002]:
                    for h_deg in [0.0002, 0.0005, 0.001]:
                        off = {'SOFT': s_off, 'MEDIUM': 0.0, 'HARD': h_off}
                        def make_deg_quad(sd, md, hd):
                            def deg_fn(c, age, temp):
                                return {'SOFT': sd, 'MEDIUM': md, 'HARD': hd}[c] * age * age
                            return deg_fn
                        acc = sum(1 for race in sample[:100] if sorted(
                            [s['driver_id'] for s in race['strategies'].values()],
                            key=lambda d: compute_driver_time(
                                next(s for s in race['strategies'].values() if s['driver_id']==d),
                                race['race_config'], off, make_deg_quad(s_deg, m_deg, h_deg)
                            )
                        ) == race['finishing_positions']) / 100
                        if acc > best_quad:
                            best_quad = acc
                            best_quad_params = [s_off, 0, h_off, s_deg, m_deg, h_deg]
    print(f"  Best quadratic: {best_quad*100:.1f}% params={best_quad_params}")

    # STRUCTURE 4: lap_time = base + off[c] * (1 + temp_c * track_temp) + deg[c] * tire_age
    # (temperature affects base compound speed, not degradation)
    print("\nTesting temp on compound offset:")
    best_temp_off = 0
    best_temp_off_params = None
    for s_off in [-0.5, -1.0, -1.5]:
        for h_off in [0.5, 1.0, 1.5]:
            for s_toff in [0.01, 0.02, 0.03, 0.05]:
                for h_toff in [-0.01, -0.02, -0.03]:
                    for s_deg in [0.05, 0.1]:
                        for h_deg in [0.01, 0.02]:
                            for m_deg in [0.02, 0.03]:
                                def make_deg_tempoff(so, ho, sto, hto, sd, md, hd):
                                    def deg_fn(c, age, temp):
                                        return {'SOFT': sd, 'MEDIUM': md, 'HARD': hd}[c] * age
                                    def t_off(c, temp):
                                        return {'SOFT': so + sto*temp, 'MEDIUM': 0, 'HARD': ho + hto*temp}[c]
                                    return deg_fn, t_off
                                dg, toff = make_deg_tempoff(s_off, h_off, s_toff, h_toff, s_deg, m_deg, h_deg)

                                def compute_with_toff(strategy, rc):
                                    total_laps = rc['total_laps']
                                    base = rc['base_lap_time']
                                    pit = rc['pit_lane_time']
                                    temp = rc['track_temp']
                                    pit_map = {p['lap']: p['to_tire'] for p in strategy.get('pit_stops',[])}
                                    current = strategy['starting_tire']
                                    ta = 0
                                    t = 0
                                    for lap in range(1, total_laps+1):
                                        ta += 1
                                        t += base + toff(current, temp) + dg(current, ta, temp)
                                        if lap in pit_map:
                                            t += pit
                                            current = pit_map[lap]
                                            ta = 0
                                    return t

                                acc = sum(1 for race in sample[:50] if sorted(
                                    [s['driver_id'] for s in race['strategies'].values()],
                                    key=lambda d: compute_with_toff(
                                        next(s for s in race['strategies'].values() if s['driver_id']==d),
                                        race['race_config'])
                                ) == race['finishing_positions']) / 50
                                if acc > best_temp_off:
                                    best_temp_off = acc
                                    best_temp_off_params = [s_off, h_off, s_toff, h_toff, s_deg, m_deg, h_deg]
    print(f"  Best temp-on-offset: {best_temp_off*100:.1f}% params={best_temp_off_params}")

    # ============================================================
    # Step 3: Look at a specific test case to guide analysis
    # ============================================================
    print("\n--- Loading test case 1 to verify ---")
    with open(os.path.join(TEST_INPUT_DIR, 'test_001.json')) as f:
        test1 = json.load(f)
    with open(os.path.join(TEST_OUTPUT_DIR, 'test_001.json')) as f:
        exp1 = json.load(f)

    rc = test1['race_config']
    print(f"Test 1: {rc['track']}, laps={rc['total_laps']}, base={rc['base_lap_time']}, pit={rc['pit_lane_time']}, temp={rc['track_temp']}")
    print(f"Expected: {exp1['finishing_positions'][:5]}...")
    print("\nStrategies (first 5 drivers):")
    exp_pos = {d: i+1 for i, d in enumerate(exp1['finishing_positions'])}
    for pos_key, strat in sorted(test1['strategies'].items())[:5]:
        did = strat['driver_id']
        pits = [(p['lap'], p['from_tire'][:1], p['to_tire'][:1]) for p in strat['pit_stops']]
        print(f"  {did} [pos={exp_pos.get(did,'?')}]: start={strat['starting_tire']}, pits={pits}")

    # ============================================================
    # Step 4: Find races with simple strategies to analyze
    # ============================================================
    print("\n--- Finding simple races (2 drivers, same compound changes) ---")
    for race in races[:1000]:
        strategies_list = list(race['strategies'].values())
        # Find two drivers with only 1 pit stop, same pit lap, same start compound, different new compound
        one_pit = [s for s in strategies_list if len(s.get('pit_stops', [])) == 1]
        for i in range(len(one_pit)):
            for j in range(i+1, len(one_pit)):
                a, b = one_pit[i], one_pit[j]
                pa, pb = a['pit_stops'][0], b['pit_stops'][0]
                if (a['starting_tire'] == b['starting_tire'] and
                    pa['lap'] == pb['lap'] and
                    pa['to_tire'] != pb['to_tire'] and
                    a['starting_tire'] != pa['to_tire'] and
                    b['starting_tire'] != pb['to_tire']):
                    rc = race['race_config']
                    finishing = race['finishing_positions']
                    pos_a = finishing.index(a['driver_id']) + 1
                    pos_b = finishing.index(b['driver_id']) + 1
                    print(f"\n  Race {race['race_id']}: base={rc['base_lap_time']}, temp={rc['temp'] if 'temp' in rc else rc['track_temp']}, laps={rc['total_laps']}")
                    print(f"    A ({a['driver_id']} pos={pos_a}): {a['starting_tire']} → lap{pa['lap']} → {pa['to_tire']}")
                    print(f"    B ({b['driver_id']} pos={pos_b}): {b['starting_tire']} → lap{pb['lap']} → {pb['to_tire']}")
                    # The difference is purely in compound2 performance from lap (pit+1) to end
                    laps_after = rc['total_laps'] - pa['lap']
                    print(f"    Laps after pit: {laps_after}")
                    if len([r for r in races[:20] if r['race_id'] == race['race_id']]) > 0:
                        break
            else:
                continue
            break

    # ============================================================
    # Step 5: Statistical analysis of temp effect
    # ============================================================
    print("\n--- Analyzing temperature effect ---")
    # Group races by temperature and see if there's a pattern
    temp_groups = {}
    for race in races:
        temp = race['race_config']['track_temp']
        temp_groups.setdefault(temp, []).append(race)

    print(f"Temperature distribution: {sorted(temp_groups.keys())}")
    for temp in sorted(temp_groups.keys())[:5]:
        print(f"  temp={temp}: {len(temp_groups[temp])} races")


if __name__ == '__main__':
    main()
