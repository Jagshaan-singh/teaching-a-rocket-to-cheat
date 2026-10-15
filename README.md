# 🚀 Teaching a Rocket to Cheat

**What happens when a reinforcement learning agent optimizes the reward instead of the goal?**

An experiment in **reward hacking** using PPO, Gymnasium's Lunar Lander, and custom reward functions. I deliberately introduced a flawed reward that incentivized hovering, observed the resulting behavior, and modified the reward to reduce the incentive.

### 🔥 Key Results

- **640 training reward:** The hacked agent's reward increased from 162 to 640.
- **-77.5 evaluation score:** The same seed's performance on the original game fell from 219.8 for the baseline to -77.5.
- **270.9 best fixed run:** The corrected reward reached the environment's solved threshold in 2 of 3 seeds.

Across 10 runs, I used a total training budget of **10 million environment timesteps**, running entirely on a CPU laptop without a GPU.

### 🎬 Watch the Experiments

All 12 recorded clips are available in [`videos/`](videos/).

| Experiment | Videos |
|---|---|
| Baseline | [Seed 0](videos/baseline-episode-0.mp4) · [Seed 1](videos/baseline-episode-1.mp4) · [Seed 2](videos/baseline-episode-2.mp4) |
| Hover hack | [Seed 0](videos/hover_hack-episode-0.mp4) · [Seed 1](videos/hover_hack-episode-1.mp4) · [Seed 2](videos/hover_hack-episode-2.mp4) |
| Fixed reward | [Seed 0](videos/fixed-episode-0.mp4) · [Seed 1](videos/fixed-episode-1.mp4) · [Seed 2](videos/fixed-episode-2.mp4) |
| Time penalty | [Seed 0](videos/time_penalty-episode-0.mp4) · [Seed 1](videos/time_penalty-episode-1.mp4) · [Seed 2](videos/time_penalty-episode-2.mp4) |

### 🧪 Reward Design and Results

All evaluation results use the original Lunar Lander environment, with custom reward wrappers disabled. Each evaluation used 20 deterministic episodes. A mean score of 200 or above counts as solved.

| Variant | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| Baseline | 219.8 ± 12.8 | — | — | 219.8* |
| Time penalty | 133.3 ± 128.1 | 271.0 ± 26.0 | 113.0 ± 114.2 | 172.4 |
| Hover hack | -77.5 ± 27.6 | 114.2 ± 95.9 | 86.7 ± 82.1 | 41.1 |
| Fixed reward | 270.9 ± 30.3 | 158.8 ± 42.8 | 202.3 ± 61.9 | 210.7 |

*Baseline uses one seed; each modified variant uses three. The ± values represent variation across evaluation episodes.*

### 🧠 How It Works

**Stack:** Python · Gymnasium · Box2D · Stable-Baselines3 PPO · PyTorch

The PPO agent uses `MlpPolicy`, a multilayer perceptron neural network, to learn a policy from the lander's observations and rewards.

Four reward configurations were tested:

- **Baseline:** Original environment reward.
- **Time penalty:** `-0.02` per frame to encourage shorter episodes.
- **Hover hack:** `+1` per frame while upright (`|tilt| < 0.2`) and slow (`speed < 0.3`). This unintentionally rewards hovering instead of landing.
- **Fixed reward:** Rewards the change into the upright-and-slow state rather than repeatedly rewarding remaining there, while retaining the time penalty.

The key finding: a high reward under a modified objective does not necessarily mean the agent has learned the intended task.

### 📁 Code

- [`01_random_agent.py`](01_random_agent.py) — Random-agent demonstration
- [`02_train_baseline.py`](02_train_baseline.py) — Baseline PPO training
- [`03_evaluate.py`](03_evaluate.py) — Evaluation
- [`04_record.py`](04_record.py) — Video recording
- [`05_train_shaped.py`](05_train_shaped.py) — Modified reward experiments
- [`wrappers.py`](wrappers.py) — Custom reward functions

### ▶️ Run Locally

```bash
git clone https://github.com/YOUR-USERNAME/teaching-a-rocket-to-cheat.git
cd teaching-a-rocket-to-cheat

python -m venv .venv
.venv\Scripts\activate

pip install "gymnasium[box2d]" stable-baselines3 torch
```

Run the scripts in the order appropriate to your implementation. Training uses 1 million environment timesteps per run and took approximately 13 minutes per run on the experimental hardware.

### 💡 Takeaways

1. Reward is not the same as success.
2. Visual inspection can reveal behavior that a reward curve hides.
3. Evaluating on the original objective helps expose reward exploitation.

### ⚠️ Limitations

Three seeds per modified variant, one baseline seed, and end-of-training checkpoints limit the strength of the conclusions. The time-penalty versus fixed-reward comparison is inconclusive given seed variability. This is a small demonstration, not a universal solution to reward hacking.

---

**The goal wasn't to build a rocket that cheats. It was to understand why an AI might.**
