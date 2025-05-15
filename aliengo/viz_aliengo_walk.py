import gymnasium as gym
import numpy as np
from sac import SAC

# Инициализация среды
env = gym.make(
    'Ant-v5',
    xml_file='/home/eidel/projects/prog_rob/unitree_mujoco/unitree_robots/aliengo/scene.xml',
    forward_reward_weight=1,
    ctrl_cost_weight=0.05,
    contact_cost_weight=5e-4,
    healthy_reward=1,
    main_body=1,
    healthy_z_range=(0.195, 0.75),
    include_cfrc_ext_in_observation=True,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0.1,
    frame_skip=25,
    max_episode_steps=1000,
    render_mode = 'human',
)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print(f"Environment initialized: state_dim={state_dim}, action_dim={action_dim}")

# Инициализация SAC
agent = SAC(
    state_dim=state_dim,
    action_dim=action_dim,
    gamma=0.99,
    alpha=0.1,
    tau=0.005,
    batch_size=256,
    pi_lr=1e-4,
    q_lr=1e-4
)

# Загрузка модели
model_path = "aliengo_models/aliengo_policy_scaled.pth"
try:
    agent.load_model(model_path)
    print(f"Loaded policy from {model_path}")
except RuntimeError as e:
    print(f"Failed to load model: {e}")
    print("Model is incompatible with current state_dim (expected 115, got 113). Please train a new model with aliengo_train.py or use the correct XML file with 115 observations.")
    exit(1)

# Параметры визуализации
max_steps = 1000  # Шаги на эпизод
episode_n = 10    # Количество эпизодов

for episode in range(episode_n):
    state, _ = env.reset()
    total_reward = 0
    
    # Диагностика структуры состояния
    if episode == 0:
        print(f"Initial state shape: {state.shape}")
        print(f"Sample state: {state[:10]} ... {state[-5:]}")
    
    print(f"\nStarting Episode {episode}")
    
    for t in range(max_steps):
        action = agent.get_action(state)
        # Масштабирование действий для ctrlrange из XML
        scaled_action = np.clip(
            action * np.array([35.278, 35.278, 44.4, 35.278, 35.278, 44.4, 35.278, 35.278, 44.4, 35.278, 35.278, 44.4]),
            env.action_space.low,
            env.action_space.high
        )
        next_state, reward, terminated, truncated, _ = env.step(scaled_action)
        done = terminated or truncated

        total_reward += reward
        state = next_state
        
        # Логирование метрик
        height = state[2]  # z-координата (qpos[2])
        com_vel = state[7]  # x-скорость (qvel[0])
        fell = 1 if height < 0.195 else 0
        print(f"Step {t}: Reward = {reward:.2f}, Height = {height:.4f}, COM Vel = {com_vel:.4f}, Fell = {fell}")

        # Рендеринг
        env.render()

        if done:
            print(f"Episode {episode} terminated after {t+1} steps with total reward: {total_reward:.2f}")
            break

# Закрытие среды
env.close()
print("Visualization completed")