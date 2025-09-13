# Welcome to Atari Games DQN implementations

---

This repository allows you to access the training and preview notebooks for different game environments: CartPole, Atari games Space Invaders, Pacman.

DQNs from this repository are trained to play these games as well as Pong better than the human benchmarks reported by "DNADRL human" (Ziyu Wang et. al). This does not mean better than the _best_ human players: best human scores are much higher and higher than what top AIs achieve (see the [comparison](https://eject.com.au/sodeepdude/comparison-of-human-scores-and-human-scores-in-atari/)).

I used a classic MLP DQN for CartPole, and a custom R2D2 implementation for other games. A classic double dueling DQN with reward clipping for Space Invaders is also implemented in-between and represents a decent result. R2D2 ended up being implemented and tested on MPS backend specifically, so might not work for CUDA and other backends.

* **`Atari_Games.ipynb`**: self-contained notebook with multiple DQN variants (DQN/Double Dueling/R2D2). Includes an **episodic replay buffer** (`episodic_replay_buffer.py`) module for R2D2.
* **`Preview_Models.ipynb`**: loads checkpoints from `checkpoints_*` and lets you watch trained agents play Space Invaders / Pacman / CartPole real-time.

## Installation
Installation steps are included in `Atari_Games.ipynb`, but put compactly they are:
```bash
pip install gymnasium ray tianshou pygame stable_baselines3
pip install ale-py autorom
AutoROM --accept-license
pip install "gymnasium[atari, accept-rom-license]"
pip install numba
```

## Usage
**Train:** Run a `Atari_Games.ipynb` cell that defines algorithm and then a cell that runs it, in order to train a model. If you want to train the models from zero, delete the contents of `checkpoints_*` folder.

**To play:** To preview a trained model, just run the cells from `Preview_Models.ipynb`. The latest model checkpoints are included in the folders `checkpoints_*` and are used automatically, so training is not required to preview the models.
