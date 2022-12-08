# Minetest Gym Environment

The gym env sends actions to the Minetest client and receives back  observations.
Can only be used with [EleutherAI's minetest fork](https://github.com/EleutherAI/minetest).

# Installation

- Clone this repository
- Create new virtual/conda environment with `python=3.8` and install dependencies: `pip install -r requirements.txt`
- Clone https://github.com/EleutherAI/minetest to `MINETEST_DIR` and switch to the `develop` branch.
- Build Minetest ([instructions](https://github.com/EleutherAI/minetest#compiling))

You can check your setup by running
- `python run_random_agent.py --minetest_dir MINETEST_DIR`

# Run VPT agents

- Download VPT model and weights file from https://github.com/openai/Video-Pre-Training#agent-model-zoo
- `python run_vpt_agent.py --model MODEL_FILE --weights WEIGHTS_FILE --minetest_dir MINETEST_DIR`
