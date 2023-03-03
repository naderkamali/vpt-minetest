# VPT models for Minetest

Integrates VPT models with [EleutherAI's minetest fork](https://github.com/EleutherAI/minetest).

# Installation

- Clone this repository
- Create new virtual/conda environment with `python=3.8` and install dependencies: `pip install -r requirements.txt`
- Clone https://github.com/EleutherAI/minetest to `MINETEST_DIR` and switch to the `develop` branch.
- Build Minetest and install the gym environment `minetester` ([instructions](https://github.com/EleutherAI/minetest#compiling))

You can check your setup by running
- `python run_random_agent.py --minetest_path MINETEST_DIR/bin/minetest`

# Run VPT agents

- Download VPT model and weights file from https://github.com/openai/Video-Pre-Training#agent-model-zoo
- `python run_vpt_agent.py --model MODEL_FILE --weights WEIGHTS_FILE --minetest_path MINETEST_DIR/bin/minetest`
