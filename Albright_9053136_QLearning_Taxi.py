import os
import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from assignment2_utils import describe_env, simulate_episodes  # professor helper



# Q-Learning Agent (Tabular)

@dataclass
class QParams:
    alpha: float = 0.1      # learning rate (α)
    gamma: float = 0.9      # discount factor (γ)
    epsilon: float = 0.1    # exploration factor (ε)


class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, params: QParams):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.params = params
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    def select_action(self, state: int) -> int:
        """ε-greedy behavior policy (required by simulate_episodes)."""
        if random.random() < self.params.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[int(state)]))

    def train(
        self,
        env: gym.Env,
        episodes: int,
        max_steps: int,
        seed: int = 42,
        visualize: bool = False,
        step_sleep: float = 0.02,
        print_every: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Q-Learning training loop.
        If visualize=True, renders EVERY step of EVERY episode (slow).
        """
        import time

        random.seed(seed)
        np.random.seed(seed)

        returns = np.zeros(episodes, dtype=np.float64)
        steps = np.zeros(episodes, dtype=np.int32)

        for ep in range(episodes):
            s, _ = env.reset(seed=seed + ep)
            s = int(s)

            G = 0.0
            for t in range(max_steps):

                if visualize:
                    env.render()
                    if step_sleep > 0:
                        time.sleep(step_sleep)

                a = self.select_action(s)
                sp, r, terminated, truncated, _ = env.step(a)
                done = bool(terminated or truncated)

                sp = int(sp)
                r = float(r)

                # Q-learning update:
                # Q[s,a] <- Q[s,a] + alpha * (r + gamma*max_a' Q[sp,a'] - Q[s,a])
                q_sa = self.Q[s, a]
                target = r if done else (r + self.params.gamma * float(np.max(self.Q[sp])))
                self.Q[s, a] = q_sa + self.params.alpha * (target - q_sa)

                s = sp
                G += r

                if done:
                    steps[ep] = t + 1
                    break
            else:
                steps[ep] = max_steps

            returns[ep] = G

            if print_every > 0 and (ep + 1) % print_every == 0:
                avg_r = float(np.mean(returns[ep - print_every + 1: ep + 1]))
                avg_s = float(np.mean(steps[ep - print_every + 1: ep + 1]))
                print(f"[Episode {ep+1}] avg_return(last {print_every})={avg_r:.2f} avg_steps={avg_s:.1f}")

        return returns, steps

# Evaluation (greedy policy)

def evaluate_greedy(env: gym.Env, Q: np.ndarray, episodes: int, max_steps: int, seed: int = 123) -> Dict[str, float]:
    random.seed(seed)
    np.random.seed(seed)

    total_return = 0.0
    total_steps = 0

    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        s = int(s)
        G = 0.0

        for t in range(max_steps):
            a = int(np.argmax(Q[s]))
            sp, r, terminated, truncated, _ = env.step(a)
            s = int(sp)
            G += float(r)

            if terminated or truncated:
                total_steps += (t + 1)
                break
        else:
            total_steps += max_steps

        total_return += G

    return {
        "eval_avg_return": total_return / episodes,
        "eval_avg_steps": total_steps / episodes,
    }


# Plotting helpers
def save_plots(returns: np.ndarray, steps: np.ndarray, title: str, out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    x = np.arange(1, len(returns) + 1)

    plt.figure()
    plt.plot(x, returns)
    plt.title(f"{title} - Return per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_returns.png"))
    plt.close()

    plt.figure()
    plt.plot(x, steps)
    plt.title(f"{title} - Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_steps.png"))
    plt.close()


def run_one_setting(
    env_id: str,
    episodes: int,
    max_steps: int,
    params: QParams,
    seed: int,
    visualize_training: bool,
    step_sleep: float,
) -> Tuple[Dict, QLearningAgent, np.ndarray, np.ndarray]:
    """
    One run: train + evaluate.
    If visualize_training=True, env uses render_mode='human' and shows EVERY step during training.
    """
    env = gym.make(env_id, render_mode="human" if visualize_training else None)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = QLearningAgent(n_states, n_actions, params)
    returns, steps = agent.train(
        env,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        visualize=visualize_training,
        step_sleep=step_sleep,
        print_every=200,
    )
    env.close()

    env_eval = gym.make(env_id)
    eval_metrics = evaluate_greedy(env_eval, agent.Q, episodes=200, max_steps=max_steps, seed=seed + 999)
    env_eval.close()

    result = {
        "run_name": "",
        "alpha": params.alpha,
        "epsilon": params.epsilon,
        "gamma": params.gamma,
        "episodes": episodes,
        "avg_return_train": float(np.mean(returns)),
        "avg_steps_train": float(np.mean(steps)),
        **eval_metrics,
    }
    return result, agent, returns, steps


def main():
    env_id = "Taxi-v3"

    # Per assignment:
    # α=0.1, ε=0.1, γ=0.9 baseline + sweeps + choose best and rerun :contentReference[oaicite:4]{index=4}
    episodes = 5000
    max_steps = 200
    seed = 42

    # Set to True if you want to SEE EVERY EPISODE while training (very slow for 5000)
    VISUALIZE_EVERY_EPISODE = False
    STEP_SLEEP = 0.02

    # Show environment info using professor helper :contentReference[oaicite:5]{index=5}
    env = gym.make(env_id)
    describe_env(env)
    env.close()

    out_dir = "assignment2_outputs"
    os.makedirs(out_dir, exist_ok=True)

    all_rows: List[Dict] = []

    # 1) Baseline run
    baseline = QParams(alpha=0.1, epsilon=0.1, gamma=0.9)
    row, agent, returns, steps = run_one_setting(
        env_id, episodes, max_steps, baseline, seed,
        visualize_training=VISUALIZE_EVERY_EPISODE,
        step_sleep=STEP_SLEEP
    )
    row["run_name"] = "baseline"
    all_rows.append(row)
    save_plots(returns, steps, "Baseline (a=0.1, e=0.1, g=0.9)", out_dir, "baseline")

    # 2) Alpha variations (separately) α = [0.01, 0.001, 0.2] :contentReference[oaicite:6]{index=6}
    for a in [0.01, 0.001, 0.2]:
        params = QParams(alpha=a, epsilon=0.1, gamma=0.9)
        row, _, returns, steps = run_one_setting(
            env_id, episodes, max_steps, params, seed,
            visualize_training=VISUALIZE_EVERY_EPISODE,
            step_sleep=STEP_SLEEP
        )
        row["run_name"] = f"alpha_{a}"
        all_rows.append(row)
        save_plots(returns, steps, f"Alpha sweep (a={a}, e=0.1, g=0.9)", out_dir, f"alpha_{a}")

    # 3) Exploration variations (assignment text says "Exploration Factor γ", but that’s ε values) :contentReference[oaicite:7]{index=7}
    for e in [0.2, 0.3]:
        params = QParams(alpha=0.1, epsilon=e, gamma=0.9)
        row, _, returns, steps = run_one_setting(
            env_id, episodes, max_steps, params, seed,
            visualize_training=VISUALIZE_EVERY_EPISODE,
            step_sleep=STEP_SLEEP
        )
        row["run_name"] = f"epsilon_{e}"
        all_rows.append(row)
        save_plots(returns, steps, f"Epsilon sweep (a=0.1, e={e}, g=0.9)", out_dir, f"epsilon_{e}")

    # 4) Best combo & rerun (use eval_avg_return as criterion) :contentReference[oaicite:8]{index=8}
    best = max(all_rows, key=lambda r: r["eval_avg_return"])
    best_params = QParams(alpha=float(best["alpha"]), epsilon=float(best["epsilon"]), gamma=float(best["gamma"]))

    row, best_agent, returns, steps = run_one_setting(
        env_id, episodes, max_steps, best_params, seed,
        visualize_training=VISUALIZE_EVERY_EPISODE,
        step_sleep=STEP_SLEEP
    )
    row["run_name"] = "best_rerun"
    all_rows.append(row)

    save_plots(
        returns,
        steps,
        f"Best re-run (a={best_params.alpha}, e={best_params.epsilon}, g={best_params.gamma})",
        out_dir,
        "best_rerun"
    )

    # Save summary CSV
    csv_path = os.path.join(out_dir, "qlearning_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(all_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print("\n=== Summary (also saved to CSV) ===")
    for r in all_rows:
        print(r)

    print(f"\nWrote outputs to: {out_dir}")
    print(f"- {csv_path}")
    print("- PNG plots: *_returns.png and *_steps.png")

    # Visual simulation (after training) using professor helper :contentReference[oaicite:9]{index=9}
    env_vis = gym.make(env_id, render_mode="human")
    print("\nVisual simulation (best agent, epsilon-greedy behavior):")
    simulate_episodes(env_vis, best_agent, num_episodes=3)
    env_vis.close()


if __name__ == "__main__":
    main()