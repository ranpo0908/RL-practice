import argparse
import csv
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sanitize_token(x: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    if duration <= 0:
        return end_e
    slope = (end_e - start_e) / duration
    return max(end_e, start_e + slope * t)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, obs_dim: int, size: int, device: torch.device):
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False

        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size,), dtype=np.int64)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.dones = np.zeros((size,), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr += 1
        if self.ptr >= self.size:
            self.ptr = 0
            self.full = True

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, batch_size: int):
        assert len(self) >= batch_size
        idx = np.random.randint(0, len(self), size=batch_size)

        obs = torch.tensor(self.obs[idx], device=self.device)
        next_obs = torch.tensor(self.next_obs[idx], device=self.device)
        actions = torch.tensor(self.actions[idx], device=self.device)
        rewards = torch.tensor(self.rewards[idx], device=self.device)
        dones = torch.tensor(self.dones[idx], device=self.device)
        return obs, actions, rewards, next_obs, dones


def make_csv_path(outdir: str, env_id: str, seed: int) -> str:
    metric_dir = os.path.join(outdir, "metric")
    os.makedirs(metric_dir, exist_ok=True)
    env_tok = sanitize_token(env_id)
    name = f"metrics_env-{env_tok}_algo-dqn_seed-{seed}.csv"
    return os.path.join(metric_dir, name)


@torch.no_grad()
def evaluate_greedy(env_id: str, q_net: QNetwork, device: torch.device, seed: int, n_episodes: int = 5) -> float:
    env = gym.make(env_id)
    returns = []
    for k in range(n_episodes):
        obs, _ = env.reset(seed=seed + 10_000 + k)
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            q = q_net(obs_t)
            action = int(torch.argmax(q, dim=1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default=os.path.join("experiments", "Ex-02"))

    # training length
    parser.add_argument("--total-timesteps", type=int, default=120_000)

    # DQN core hyperparams (reasonable CPU defaults)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--target-update-frequency", type=int, default=1_000)

    # epsilon schedule
    parser.add_argument("--start-e", type=float, default=1.0)
    parser.add_argument("--end-e", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.5)

    # eval
    parser.add_argument("--eval-frequency", type=int, default=10_000)  # 0 to disable
    parser.add_argument("--eval-episodes", type=int, default=5)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cpu")

    env = gym.make(args.env_id)
    obs, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n

    q_net = QNetwork(obs_dim, n_actions).to(device)
    target_net = QNetwork(obs_dim, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)
    rb = ReplayBuffer(obs_dim=obs_dim, size=args.buffer_size, device=device)

    csv_path = make_csv_path(args.outdir, args.env_id, args.seed)
    t0 = time.time()

    # episode trackers
    ep = 0
    ep_ret = 0.0
    ep_len = 0

    exploration_duration = int(args.exploration_fraction * args.total_timesteps)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "global_step",
                "episode",
                "episode_return",
                "episode_length",
                "epsilon",
                "loss",
                "eval_mean_return",
                "wall_time_sec",
            ],
        )
        writer.writeheader()

        loss_val = float("nan")
        eval_val = float("nan")

        for global_step in range(1, args.total_timesteps + 1):
            epsilon = linear_schedule(args.start_e, args.end_e, exploration_duration, global_step)

            # ε-greedy action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                    q = q_net(obs_t)
                    action = int(torch.argmax(q, dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            rb.add(obs, action, float(reward), next_obs, float(done))

            ep_ret += float(reward)
            ep_len += 1
            obs = next_obs

            # if episode ended, log & reset
            if done:
                ep += 1
                writer.writerow(
                    {
                        "global_step": global_step,
                        "episode": ep,
                        "episode_return": ep_ret,
                        "episode_length": ep_len,
                        "epsilon": round(epsilon, 6),
                        "loss": loss_val,
                        "eval_mean_return": eval_val,
                        "wall_time_sec": round(time.time() - t0, 3),
                    }
                )
                if ep % 10 == 0 or ep == 1:
                    print(f"[ep {ep:04d} | step {global_step:07d}] return={ep_ret:.1f} len={ep_len} eps={epsilon:.3f} loss={loss_val}")

                obs, _ = env.reset(seed=args.seed + ep)
                ep_ret = 0.0
                ep_len = 0

            # train
            if global_step > args.learning_starts and (global_step % args.train_frequency == 0) and len(rb) >= args.batch_size:
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = rb.sample(args.batch_size)

                with torch.no_grad():
                    next_q = target_net(b_next_obs).max(dim=1).values
                    td_target = b_rewards + args.gamma * (1.0 - b_dones) * next_q

                q_values = q_net(b_obs).gather(1, b_actions.view(-1, 1)).squeeze(1)
                loss = nn.functional.mse_loss(q_values, td_target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

                loss_val = float(loss.item())

            # target network update
            if global_step % args.target_update_frequency == 0:
                target_net.load_state_dict(q_net.state_dict())

            # eval (greedy)
            if args.eval_frequency > 0 and global_step % args.eval_frequency == 0:
                eval_val = evaluate_greedy(
                    env_id=args.env_id,
                    q_net=q_net,
                    device=device,
                    seed=args.seed,
                    n_episodes=args.eval_episodes,
                )
                print(f"[eval @ step {global_step}] greedy_mean_return={eval_val:.1f}")

    env.close()

    print("\n=== DQN Finished ===")
    print(f"saved: {csv_path}")
    print(f"outdir: {args.outdir}")


if __name__ == "__main__":
    main()