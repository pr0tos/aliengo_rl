###Aliengo RL
Training and visualization of the Aliengo quadruped walking using Soft Actor-Critic (SAC) in the Ant-v5 environment (Gymnasium/MuJoCo).
#Installation

Clone the repository:
bash
```
git clone https://github.com/pr0tos/aliengo_rl.git
cd aliengo_rl
```

Create a Conda environment:
bash
```
conda create -n prog_rob python=3.8
conda activate prog_rob
```

Install dependencies:
bash
```
pip install -r requirements.txt
```

#Usage
Training
Train a new SAC model:
bash
```
python aliengo_train.py --use_wandb
```

Saves model to aliengo_models/aliengo_policy.pth.
Logs metrics to WandB (if enabled).

Note: The current model (aliengo_policy.pth) may be incompatible (state_dim=115 vs 113). Training will create a new compatible model.

#Visualization
Visualize the trained policy:
bash
```
python viz_aliengo_walk.py
```

