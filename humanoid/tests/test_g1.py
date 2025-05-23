import gymnasium as gym
import numpy as np



n_episodes = 10
n_steps = 1000
# Инициализация среды
env = gym.make(
    'Humanoid-v5',
    xml_file='/home/eidel/projects/prog_rob/unitree_mujoco/unitree_robots/h1/scene.xml',
    render_mode = 'human',
    max_episode_steps=n_episodes,
    )

def render(env, n_episodes = 10, n_steps = 1000):
    # Вывод пространства действий для диагностики
    print("Action space:", env.action_space)
    print("Action space low:", env.action_space.low)
    print("Action space high:", env.action_space.high)  
    # Проверка начального наблюдения
    obs = env.reset()[0]
    print("Initial observation:", obs)
    print("Contains NaN:", np.isnan(obs).any())

    
    for _ in range(n_episodes = 100):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        if done:
            observation, _ = env.reset()



if __name__ == '__main__':

    render(env, n_episodes, n_steps)