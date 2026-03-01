import argparse
import csv
import os
import random
import time

import numpy as np
import gymnasium as gym


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def sanitize_token(x: str) -> str:
    # safe for filenames across OS
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in x)


def make_csv_name(env_id: str, policy: str, action_seed_mode: int, seed: int) -> str:
    env_tok = sanitize_token(env_id)
    pol_tok = sanitize_token(policy)
    return f"metrics_env-{env_tok}_pol-{pol_tok}_aseed-{action_seed_mode}_seed-{seed}.csv"


def select_action(policy: str, env: gym.Env) -> int:
    if policy == "random":
        return env.action_space.sample()
    if policy == "left":
        return 0
    if policy == "right":
        return 1
    raise ValueError(f"Unknown policy={policy}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--outdir", type=str, default=os.path.join("experiments", "Ex-01"))

    # (i) action-space seeding mode: 0/1/2
    # 0: do not seed action_space at all
    # 1: seed action_space once (before episode loop)
    # 2: seed action_space per-episode with (seed + ep)
    parser.add_argument("--action-seed-mode", type=int, choices=[0, 1, 2], default=2)

    # (ii) behavior policy: random/left/right
    parser.add_argument("--policy", type=str, choices=["random", "left", "right"], default="random")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    env = gym.make(args.env_id)

    # Gymnasium uses env.reset(seed=...) for reproducibility
    obs, info = env.reset(seed=args.seed)

    # Apply action-space seeding according to mode
    if args.action_seed_mode == 1:
        env.action_space.seed(args.seed)

    metric_dir = os.path.join(args.outdir, "metric")
    os.makedirs(metric_dir, exist_ok=True)
    csv_name = make_csv_name(
        env_id=args.env_id,
        policy=args.policy,
        action_seed_mode=args.action_seed_mode,
        seed=args.seed,
    )
    csv_path = os.path.join(metric_dir, csv_name)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "episode_return",
                "episode_length",
                "seed",
                "env_id",
                "policy",
                "action_seed_mode",
                "wall_time_sec",
            ],
        )
        writer.writeheader()

        returns = []
        lengths = []
        t0 = time.time()

        for ep in range(1, args.episodes + 1):
            obs, info = env.reset(seed=args.seed + ep)  # vary per-episode seed (env dynamics)

            if args.action_seed_mode == 2:
                env.action_space.seed(args.seed + ep)

            done = False
            ep_ret = 0.0
            ep_len = 0

            while not done:
                action = select_action(args.policy, env)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_ret += float(reward)
                ep_len += 1

            returns.append(ep_ret)
            lengths.append(ep_len)

            writer.writerow(
                {
                    "episode": ep,
                    "episode_return": ep_ret,
                    "episode_length": ep_len,
                    "seed": args.seed,
                    "env_id": args.env_id,
                    "policy": args.policy,
                    "action_seed_mode": args.action_seed_mode,
                    "wall_time_sec": round(time.time() - t0, 3),
                }
            )

            if ep % 10 == 0 or ep == 1:
                print(f"[ep {ep:03d}] return={ep_ret:.1f} len={ep_len}")

    env.close()

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    mean_len = float(np.mean(lengths))

    print("\n=== Baseline Runner ===")
    print(f"env_id: {args.env_id}")
    print(f"episodes: {args.episodes}")
    print(f"seed: {args.seed}")
    print(f"policy: {args.policy}")
    print(f"action_seed_mode: {args.action_seed_mode}")
    print(f"mean_return: {mean_ret:.2f} ± {std_ret:.2f}")
    print(f"mean_length: {mean_len:.2f}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()