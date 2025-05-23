import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import os
import argparse


# Инициализация среды
env = gym.make(
    'Humanoid-v5',
    xml_file='/home/eidel/projects/prog_rob/unitree_mujoco/unitree_robots/h1/scene.xml',
    render_mode = 'human',
    )
env = Monitor(RescaleAction(env, min_action=-1., max_action=1.))

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(env):
    # Вывод пространства действий для диагностики
    print("Action space:", env.action_space)
    print("Action space low:", env.action_space.low)
    print("Action space high:", env.action_space.high)  

    # Проверка начального наблюдения
    obs = env.reset()[0]
    print("Initial observation:", obs)
    print("Contains NaN:", np.isnan(obs).any())
    
    action_noise = NormalActionNoise(mean=np.zeros(20), sigma=0.1 * np.ones(20))
    model = SAC(
        "MlpPolicy",
        env,
        action_noise=action_noise, 
        verbose=1, 
        device='cuda:0', 
        learning_starts=100, 
        batch_size=256, 
        tensorboard_log=log_dir
        )

    

    TIMESTEPS = 25000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        print(model.action_space)
        model.save(f"{model_dir}/SAC_{TIMESTEPS*iters}")

def test(env, path_to_model):
    model = SAC.load(path_to_model, env=env)
    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        print(action)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break


if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()


    if args.train:
        train(env)

    if(args.test):
        if os.path.isfile(args.test):
            test(env, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')
