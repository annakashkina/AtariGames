# Atari Games – DQN & R2D2 Implementations

---

This repository contains training and preview notebooks for different game environments: CartPole and Atari games Space Invaders, Pacman and Pong.

DQNs from this repository are trained to play these games better than the human benchmarks reported by "DNADRL human" (Ziyu Wang et. al), as measured by average points over 100 episodes. This does not mean better than the _best_ human: best human scores are much higher and higher than what top AIs achieve (see the [comparison](https://eject.com.au/sodeepdude/comparison-of-human-scores-and-human-scores-in-atari/)).

I used a classic MLP DQN for CartPole, and a custom R2D2 implementation for other games. A classic double dueling DQN with reward clipping for Space Invaders is also implemented in-between and represents a decent result. R2D2 ended up being implemented and tested on MPS backend specifically, so might not work for CUDA and other backends. Models preview should work on different platforms.

* **`Atari_Games.ipynb`**: self-contained notebook with multiple DQN variants (DQN/Double Dueling/R2D2). Includes an **episodic replay buffer** (`episodic_replay_buffer.py`) module for R2D2.
* **`Preview_Models.ipynb`**: loads checkpoints from `checkpoints_*` and lets you watch trained agents play Space Invaders / Pacman / CartPole real-time. Contains fast evaluation at the end.
* **`Preview_Models_OpenCV.ipynb`**: same as `Preview_Models.ipynb`, but uses OpenCV instead of gymnasium's `render_mode='human'` for rendering.

## Installation
```bash
pip install gymnasium pygame
pip install ale-py autorom
AutoROM --accept-license
pip install "gymnasium[atari,accept-rom-license]"
pip install numba
pip install stable_baselines3 # not required
pip install opencv-python # only required for OpenCV preview
```

## Quickstart

1. Launch Jupyter: `jupyter notebook`
2. In `Atari_Games.ipynb`, run the algorithm cell you want, then the run cell to train.
3. To retrain from scratch, clear `checkpoints_*`.

In `Preview_Models.ipynb`, run the preview cells. It will pick the latest/best checkpoints automatically. To measure performance, run the evaluate cells.

## Acknowledgements
I thank the researchers whose work underpins this project - Mnih et al. (DQN), van Hasselt et al. (Double DQN), Wang et al. (Dueling), Schaul et al. (PER), and Kapturowski et al. (R2D2) - and Sutton & Barto for their foundational textbook. I am grateful to the Gymnasium/ALE, PyTorch (including the MPS backend), and Numba communities for the tools that made this implementation possible. Any errors or misinterpretations are mine alone.