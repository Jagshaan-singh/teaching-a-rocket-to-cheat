Teaching a Rocket to Cheat

I trained an AI to land a rocket, then gave it a badly designed reward and watched it stop landing.

With the flawed reward, the agent's training score went up (162 → 640) while its real landing score fell from 219.8 to -77.5. It had learned to hover and collect points instead of landing. After I fixed the reward, it recovered and landed again in 2 of 3 runs.

The point: a reward can go up while the agent gets worse at the actual job. Always check the behavior, not just the score.

What I did

I used Gymnasium's LunarLander-v3 and trained PPO (Stable-Baselines3, default MlpPolicy) for 1M steps per run, on a laptop CPU. On top of the game's built-in reward I wrote small reward wrappers and compared four setups:

Setup	Reward change	Why
Baseline	None	Reference
Time penalty	-0.02 per frame	Nudge toward faster landings
Hover hack	+1 per frame while upright and slow	Break it on purpose
Fixed	+1 only for entering the upright-and-slow state, plus -0.02 per frame	Remove the reason to hover

The hack works because the bonus is paid every frame. Landing ends the episode and the bonus stops, so hovering for up to 1,000 frames pays more than landing. The fix pays once for getting into the good state, so sitting there earns nothing extra.

Results

Every score below was measured on the original game with my wrappers turned off, over 20 deterministic episodes. 200 or more counts as solved. The ± is variation across those 20 episodes, not across seeds.

Setup	Seed 0	Seed 1	Seed 2	Mean
Baseline	219.8 ± 12.8	not run	not run	219.8
Time penalty	133.3 ± 128.1	271.0 ± 26.0	113.0 ± 114.2	172.4
Hover hack	-77.5 ± 27.6	114.2 ± 95.9	86.7 ± 82.1	41.1
Fixed	270.9 ± 30.3	158.8 ± 42.8	202.3 ± 61.9	210.7
Hover hack: never reached 200 in any seed. Seed 0 is the clear case: the clip shows it hovering instead of landing.
Fixed: solved in 2 of 3 seeds, and its mean is close to the baseline's.
Time penalty vs fixed: the time-penalty runs vary a lot between seeds, so I can't say which is better.
Be careful with these numbers
The baseline has one seed, the others have three. 10 runs in total (10M steps combined).
The hack was dramatic in seed 0 but milder in seeds 1 and 2.
The fix did not beat the baseline. It brought performance back after the hack.
Models are the final checkpoint, not the best one.

This shows reward hacking in one small environment. It doesn't prove anything general.

Videos

12 clips in videos/, three per setup. If you only watch two, watch the baseline and the hover hack (seed 0).

Run it
bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install swig
pip install "gymnasium[box2d]" stable-baselines3 moviepy

Run everything from the project root, so models/ and videos/ land in one place.

bash
python 02_train_baseline.py
python 05_train_shaped.py hover_hack 0     # also: time_penalty, fixed
python 03_evaluate.py models/hover_hack_s0
python 04_record.py models/hover_hack_s0 hover_hack

01_random_agent.py just shows a random rocket crashing. Reward wrappers live in wrappers.py.

What I took from it
Look at the agent, not only the reward curve.
Evaluate on the original objective, with your own bonuses off.
Run more than one seed before you believe a result.

The agent optimizes the reward it is given. Writing a reward that means what you want is the hard part.