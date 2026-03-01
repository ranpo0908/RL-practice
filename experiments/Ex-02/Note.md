# Ex-02 — DQN CartPole Code Walkthrough + Interview Q&A

This note explains the **CleanRL-style single-file DQN** (`cleanrl/dqn_cartpole.py`) in a way that connects:
- **RL concepts / math objects** (policy, Q-function, TD target),
- **where they appear in code**, and
- **common pitfalls** (including why you may see `NaN` in logs).

---

## 0) What are we “learning” in DQN?

DQN does **not** directly parameterize a policy. It learns a **Q-function**:

$$
Q_\theta(s,a) \approx \mathbb{E}\big[\sum_{t\ge 0}\gamma^t r_t\mid s_0=s, a_0=a\big].
$$

Then the *derived* (greedy) policy is:

$$
\pi_\theta(s)=\arg\max_a Q_\theta(s,a).
$$

**What is learned?** The neural network parameters **$\theta$** of `q_net`.

- `q_net.parameters()` are optimized by gradient descent.
- `target_net` is *not* learned by gradient; it is a periodically-updated copy of `q_net`.

---

## 1) Data: what transition tuple is stored?

Each environment interaction produces a transition:

$$
(s, a, r, s', d)
$$

where $d\in\{0,1\}$ indicates episode termination.

**In code:**
- `next_obs, reward, terminated, truncated, _ = env.step(action)`
- `done = terminated or truncated`
- `rb.add(obs, action, reward, next_obs, done)`

**Why store `done`?** Because terminal transitions should not bootstrap:
$$
y = r \quad \text{if done}=1.
$$

---

## 2) Replay buffer: why it exists

The replay buffer stores past transitions and samples random mini-batches.

**Why?**
1. Breaks strong temporal correlation (stabilizes SGD).
2. Makes learning **off-policy** (learn from older experience).

**In code:**
- `ReplayBuffer.add(...)`
- `ReplayBuffer.sample(...)` uses `np.random.randint(...)`

**Typical bugs:**
- wrong dtype for `actions` (should be `int64` for `.gather`)
- shape mismatches in `.gather`

---

## 3) Behavior policy during training: ε-greedy

During training, DQN typically uses **ε-greedy**:

$$
a=
\begin{cases}
\text{random action}, & \text{w.p. } \varepsilon \\
\arg\max_a Q_\theta(s,a), & \text{w.p. } 1-\varepsilon.
\end{cases}
$$

**In code:**
- `epsilon = linear_schedule(...)`
- `if random.random() < epsilon: action = env.action_space.sample()`
- `else: action = argmax(q_net(obs))`

**Interpretation of logs:**
- early: ε≈1 → mostly random → low return
- later: ε decreases → more exploitation → return should trend upward

---

## 4) Why `loss` is often `NaN` early

The script initializes:
- `loss_val = NaN`
- `eval_val = NaN`

Then it only **starts training** after `learning_starts` steps (default `1000`) and only at certain frequencies.

Training condition:

- `global_step > learning_starts`  
- `global_step % train_frequency == 0`  
- `len(rb) >= batch_size`

So before ~1000 steps, **no gradient update happens**, and the logged `loss` remains `NaN`.  
This is expected and indicates “not computed yet,” not a numerical error.

---

## 5) TD target: the heart of DQN

### 5.1 Vanilla DQN TD target

$$
y = r + \gamma (1-d)\max_{a'} Q_{\bar\theta}(s',a').
$$

- $Q_{\bar\theta}$ is the **target network**.

### 5.2 Loss

$$
L(\theta)=\mathbb{E}\big[(Q_\theta(s,a)-y)^2\big].
$$

**In code (target computation):**
```python
with torch.no_grad():
    next_q = target_net(b_next_obs).max(dim=1).values
    td_target = b_rewards + gamma * (1 - b_dones) * next_q
```

**In code (current Q(s,a)):**
```python
q_values = q_net(b_obs).gather(1, b_actions.view(-1, 1)).squeeze(1)
loss = mse(q_values, td_target)
```

**Why `no_grad()`?** We do not backprop through the TD target.

**Common pitfall:** forgetting `(1 - done)` causes bootstrapping through terminal transitions.

---

## 6) Target network: why it stabilizes learning

If you use `q_net` to compute both the current prediction and the bootstrap target, the target moves every gradient step, creating a feedback loop.

DQN uses a separate target network $Q_{\bar\theta}$ that updates slowly:

- Online network: $Q_\theta$ (`q_net`) updated by SGD
- Target network: $Q_{\bar\theta}$ (`target_net`) updated by periodic copy

**In code:**
```python
if global_step % target_update_frequency == 0:
    target_net.load_state_dict(q_net.state_dict())
```

This reduces instability by making $y$ more “stationary” over short windows.

---

## 7) Evaluation: why `eval_mean_return` can also be `NaN`

Evaluation runs only every `eval_frequency` steps (default `10000`).

So `eval_mean_return` stays `NaN` until the first evaluation step is reached.  
If you want earlier evaluations, run with smaller frequency, e.g. `--eval-frequency 2000`.

---

## 8) Shape sanity (useful for debugging and interviews)

Let batch size be $B$ and action count be $A$.

- `target_net(b_next_obs)` has shape `(B, A)`
- `.max(dim=1).values` has shape `(B,)`
- `td_target` has shape `(B,)`

For current Q values:

- `q_net(b_obs)` has shape `(B, A)`
- `.gather(1, actions[:,None])` has shape `(B,1)`
- `.squeeze(1)` has shape `(B,)`

So MSE is between two `(B,)` vectors.

---

# Interview Q&A (based on the code)

## Q1) When we say we “learn a policy,” what do we really learn?

In DQN, we learn parameters **$\theta$** of a **Q-function** $Q_\theta(s,a)$.  
The policy is derived as $\pi_\theta(s)=\arg\max_a Q_\theta(s,a)$.

---

## Q2) In the code, which network is used during exploitation?

**Answer:** `q_net` (the online network) is used for exploitation to compute `argmax_a Q(s,a)`.

---

## Q3) Why do we use a target network in DQN?

**Answer:** To stabilize TD learning.  
Using a slowly updated `target_net` reduces moving-target instability because the TD target uses $Q_{\bar\theta}$ rather than the constantly-changing online network.

---

## Q4) Why use a replay buffer?

**Answer:** It reduces correlation in training data and allows off-policy learning by sampling random transitions from past experience.

---

## Q5) Why are there many `NaN` values in the CSV early on?

**Answer:** Because `loss` and `eval_mean_return` are only computed after certain conditions:
- `loss` appears only after `learning_starts` and training steps occur.
- `eval_mean_return` appears only after the first multiple of `eval_frequency`.

---

## Q6) What is the “deadly triad” and where does DQN fit?

The deadly triad refers to instability from combining:
1. Function approximation (neural net),
2. Bootstrapping (TD targets),
3. Off-policy learning (replay + ε-greedy experience).

DQN includes all three, which is why tricks like replay buffers and target networks are crucial.

---

## Q7) What changes in Double DQN (high-level)?

Vanilla DQN uses the same max operator for selection and evaluation, causing overestimation bias.  
Double DQN decouples them:
- select action using `q_net` (`argmax`)
- evaluate that action using `target_net` (`gather`)

(Implementable by changing only a few lines in the TD target.)

---

## Notes / TODO
- Consider logging step-based metrics (loss at each train step) if you want fewer `NaN`s.
- For quick progress monitoring, set `--eval-frequency 2000`.
