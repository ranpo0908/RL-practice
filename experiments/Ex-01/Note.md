## Day 1 — Random/Fixed Policy Baselines (CartPole-v1)
  
**Script:** `cleanrl/random_cartpole.py`

### Run Commands
1) Random policy, action seed mode 1/2, random seed 0:
- `python cleanrl/random_cartpole.py --episodes 50 --seed 0 --policy random --action-seed-mode 1`
- `python cleanrl/random_cartpole.py --episodes 50 --seed 0 --policy random --action-seed-mode 2`

2) Random policy, action seed mode 1/2, random seed 42:
- `python cleanrl/random_cartpole.py --episodes 50 --seed 42 --policy random --action-seed-mode 1`
- `python cleanrl/random_cartpole.py --episodes 50 --seed 42 --policy random --action-seed-mode 2`

3) Random policy, action seed mode 1/2, random seed 1234:
- `python cleanrl/random_cartpole.py --episodes 50 --seed 1234 --policy random --action-seed-mode 1`
- `python cleanrl/random_cartpole.py --episodes 50 --seed 1234 --policy random --action-seed-mode 2`

4) Left/Right policy, action seed mode '-', random seed 0:
- `python cleanrl/random_cartpole.py --episodes 50 --seed 0 --policy left --action-seed-mode 0`
- `python cleanrl/random_cartpole.py --episodes 50 --seed 0 --policy right --action-seed-mode 0`

5) Left/Right policy, action seed mode '-', random seed 42:
- `python cleanrl/random_cartpole.py --episodes 50 --seed 42 --policy left --action-seed-mode 0`
- `python cleanrl/random_cartpole.py --episodes 50 --seed 42 --policy right --action-seed-mode 0`

6) Left/Right policy, action seed mode '-', random seed 1234:
- `python cleanrl/random_cartpole.py --episodes 50 --seed 1234 --policy left --action-seed-mode 0`
- `python cleanrl/random_cartpole.py --episodes 50 --seed 1234 --policy right --action-seed-mode 0`

### Output Summary (fill these from terminal)
| policy | random seed | action_seed_mode | mean_return (± std) | mean_length |
|---|---:|---:|---:|---|
| random | 0 | 1 | 25.78 ± 13.43 | 25.78 | 
| random | 0 | 2 | 26.32 ± 17.52 | 26.32 |
| random | 42 | 1 | 21.80 ± 13.49 | 21.80 |
| random | 42 | 2 | 26.28 ± 17.13 | 26.28 |
| random | 1234 | 1 | 22.92 ± 10.66 | 22.92 |
| random | 1234 | 2 | 23.04 ± 13.67 | 23.04 |
| left | 0 | 0 | 9.36 ± 0.69 | 9.36 | 
| right | 0 | 0 | 9.22 ± 0.87 | 9.22 |
| left | 42 | 0 | 9.42 ± 0.60 | 9.42 |
| right | 42 | 0 | 9.30 ± 0.73 | 9.30 |
| left | 1234 | 0 | 9.36 ± 0.71 | 9.36 |
| right | 1234 | 0 | 9.36 ± 0.69 | 9.36 |
---

# (i) Observations + Interpretation

## (a) Fixed `--seed`, but action seed in different places (mode 1 vs mode 2)

### Observation 1: For **random policy**, action_seed_mode=2 often gives **larger std** and sometimes higher mean than mode 1.
From my runs (50 episodes each):

- seed=0:
  - mode 1: 25.78 ± 13.43
  - mode 2: 26.32 ± 17.52  (std ↑)

- seed=42:
  - mode 1: 21.80 ± 13.49
  - mode 2: 26.28 ± 17.13  (mean ↑, std ↑)

- seed=1234:
  - mode 1: 22.92 ± 10.66
  - mode 2: 23.04 ± 13.67  (std ↑)

So the consistent pattern is: **mode 2 tends to increase variability (std)**; mean can go up or stay similar, but is not monotone across seeds.

### Observation 2: Switching `--seed` changes the baseline level even for the same mode.
Example:
- mode 1 mean_return: seed 0 → 25.78, seed 42 → 21.80, seed 1234 → 22.92
This suggests the “random baseline performance” itself has meaningful dependence on the random stream (initial states + sampled action sequences).

### Observation 3: For **fixed policies** (always left / always right), results are low and very stable, almost independent of `--seed`.
- left: ~9.36–9.42 with std ~0.6–0.7
- right: ~9.22–9.36 with std ~0.7–0.9
Across seeds 0 / 42 / 1234, differences are tiny.

This makes sense because for a deterministic policy, the only randomness left is mostly from environment reset stochasticity (CartPole starts near upright but with small random perturbation), which has limited impact given the policy is poor.

---

## My interpretation (why these happen)

### Key idea: there are at least two distinct RNG streams in this script
1) **Environment RNG** controlled by `env.reset(seed=...)`  
   - influences initial state (and sometimes dynamics in other envs).
   - In my code, I reset each episode with `seed + ep`, so each episode has a different initial randomness stream.

2) **Action-space RNG** controlled by `env.action_space.seed(...)`  
   - influences `env.action_space.sample()` which generates the random action each step.
   - This is exactly what action_seed_mode is controlling.

### Why mode 1 vs mode 2 changes mean/std
- **Mode 1**: seed action RNG **once** before the episode loop.  
  Then across episodes, the action RNG continues as one long sequence. Even though each episode starts with a new env seed, the sampled actions come from a single continuous RNG stream.

- **Mode 2**: seed action RNG **per episode** using `seed + ep`.  
  Each episode gets its “own” action RNG stream. This can create larger episode-to-episode differences because action sequences are less “coupled” across episodes (each episode effectively restarts the action RNG).  
  Empirically, this matches my observation that mode 2 often has a larger std.

Another way to view it:
- Mode 1 has one global action RNG path; mode 2 has many smaller independent-ish action RNG paths (one per episode), which can increase variance of per-episode returns.

### Why fixed policies barely depend on seed
For `policy=left` or `policy=right`, the action sequence is deterministic. So action RNG is irrelevant, and only the env reset randomness matters. CartPole’s reset randomness is small; plus the policy is bad anyway, so the episode ends quickly regardless, making results stable around ~9.

---

## My thinking / takeaways (what this teaches me about RL experiments)

1) **Seeding is not a single switch** — different components (env reset vs action sampling vs network init vs replay sampling) each have their own RNG, and where I seed them changes the experimental behavior.

2) A “random baseline” is not a fixed number; it can move noticeably with different seeds.  
   So for reporting baselines (and later DQN/PPO), I should use:
   - multiple seeds (e.g., 3–10 seeds),
   - report mean ± std across seeds.

3) Mode 2 (per-episode action seeding) can be helpful for controlled comparisons at the episode level, but it may artificially change the correlation structure of randomness across episodes compared with mode 1.  
   For fairness in later algorithm comparisons, I should pick **one consistent seeding convention** and stick to it.

4) The left/right results are a good sanity check:
   - They are low and stable, which is expected.
   - If a future learning algorithm performs similarly to left/right after training, something is likely wrong.

**Conclusion (one sentence):**
Action seeding location changes the randomness structure of sampled actions; this affects variance (and sometimes mean) of the random-policy baseline, while deterministic policies are mostly insensitive to action seeding and only weakly sensitive to env reset noise.

---

# (ii) Further Q&A

## Q1: What’s the difference between `terminated` and `truncated`? Why do we use `done = terminated or truncated`?
**Answer:**
- `terminated` = true terminal condition in the MDP (success/failure).
- `truncated` = episode ended due to a time limit or external cutoff (not a “true” terminal state).
- We treat both as episode end for logging and resetting; otherwise we may overrun or mis-measure episode stats.

## Q2: Why do seeds matter in RL experiments? What are the main randomness sources?
**Answer:**
- Seeds make experiments reproducible and debuggable.
- Main sources: environment reset RNG, action sampling RNG, network initialization, replay sampling, minibatch order, hardware nondeterminism.

## Q3: Why build a random-policy baseline before implementing DQN/PPO?
**Answer:**
- Sanity-checks the whole pipeline (env loop, done logic, logging).
- Provides a performance floor to compare improvements.
- Helps detect bugs (if “learning” is worse than random, something is wrong).

---