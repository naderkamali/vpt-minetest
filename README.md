# Minetest Gym Environment

The gym env sends actions to the Minetest client and receives back image observations.
Can only be used with [EleutherAI's minetest fork](https://github.com/EleutherAI/minetest).

# Installation

- Clone this repository
- Clone https://github.com/EleutherAI/minetest to `MINETEST_DIR` and switch to the `develop` branch.
- Build minetest

You can check your setup by running
- set `minetest_dir=MINETEST_DIR` in `run_random_agent.py`
- `python run_random_agent.py`

# Run VPT agents

- Download VPT model and weights file from https://github.com/openai/Video-Pre-Training#agent-model-zoo
- `python run_vpt_agent.py --model MODEL_FILE --weights WEIGHTS_FILE --minetest_dir MINETEST_DIR`
