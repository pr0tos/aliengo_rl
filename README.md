###Aliengo RL###
Training and visualization of the Aliengo quadruped walking using Soft Actor-Critic (SAC) in the Ant-v5 environment (Gymnasium/MuJoCo).
#Installation#

Clone the repository:
bash
```
git clone https://github.com/pr0tos/aliengo_rl.git
cd aliengo_rl
```

Create a Conda environment:
bash
```
conda env create -f environment.yml
conda activate aliengo_rl
pip install --upgrade anyio
```

#Usage#

Train a new SAC model:
bash
```
python aliengo_train.py --use_wandb
```

Saves model to aliengo_models/aliengo_policy.pth.
Logs metrics to WandB (if enabled).

#Visualization#
Visualize the trained policy:
bash
```
python viz_aliengo_walk.py
```
#Known Issues and Fixes#
1. CUDA Initialization Error
Error:
```
   UserWarning: CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero. (Triggered internally at /opt/conda/conda-bld/pytorch_1724789115765/work/c10/cuda/CUDAFunctions.cpp:108.)
   return torch._C._cuda_getDeviceCount() > 0
```
Solution:
```
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```

2. GLFW/OpenGL Configuration Issues
Error:
```
GLFWError: (65542) b'GLX: No GLXFBConfigs returned'  
GLFWError: (65545) b'GLX: Failed to find a suitable GLXFBConfig'  
Assertion `window != NULL' failed.
```
Solution:
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```
