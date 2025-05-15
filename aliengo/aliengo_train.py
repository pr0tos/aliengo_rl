import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import wandb
from sac import SAC
# from ppo import PPO
from datetime import datetime

def train(agent, env, episodes_n=20000, max_steps=1000, load_model_path=None, use_wandb=False):
    """
    Обучает агента в заданной среде.

    Args:
        agent: SAC or PPO instance.
        env: Gym environment.
        episodes_n (int): Number of episodes.
        max_steps (int): Maximum steps per episode.
        load_model_path (str): Path to a pre-trained model to load (optional).
        use_wandb (bool): Whether to log to WandB.

    Returns:
        list: Total rewards per episode.
    """
    total_rewards = []
    running_reward = []
    best_avg_reward = -float('inf')

    # Ensure the models directory exists
    os.makedirs("aliengo_models", exist_ok=True)

    # Load pre-trained model if provided
    if load_model_path and os.path.exists(load_model_path):
        try:
            print(f"Loading pre-trained model from {load_model_path}")
            agent.load_model(load_model_path)
        except RuntimeError as e:
            print(f"Failed to load model due to size mismatch: {e}")
            print("Training from scratch instead.")
    else:
        print("No pre-trained model found, training from scratch.")

    for episode in range(episodes_n):
        total_reward = 0
        state, _ = env.reset()
        
        for t in range(max_steps):
            action = agent.get_action(state)
            scaled_action = np.clip(
                action * np.array([35.278, 35.278, 44.4, 35.278, 35.278, 44.4, 35.278, 35.278, 44.4, 35.278, 35.278, 44.4]),
                env.action_space.low,
                env.action_space.high
            )
            next_state, reward, terminated, truncated, _ = env.step(scaled_action)
            done = terminated or truncated

            agent.fit(state, action, reward, done, next_state, use_wandb=use_wandb)
        
            total_reward += reward
            state = next_state

            if done:
                break
        
        total_rewards.append(total_reward)
        running_reward.append(total_reward)
        if len(running_reward) > 10:
            running_reward.pop(0)
        
        # Compute average reward
        avg_reward = np.mean(running_reward)
        
        # Log metrics to WandB if enabled
        if use_wandb:
            wandb.log({
                "episode": episode,
                "total_reward": total_reward,
                "avg_reward": avg_reward,
                "steps_per_episode": t + 1
            })
        
        # Save model if average reward improves
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save_model(f"aliengo_models/aliengo_policy.pth")
            print(f"Episode {episode}: Saved model with avg_reward = {avg_reward:.2f}")
        
        print(f'Episode {episode}: Total Reward = {total_reward}')

    return total_rewards

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train SAC or PPO on Aliengo environment")
    parser.add_argument('--use_wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--agent', choices=['sac', 'ppo'], default='sac', help='Choose agent: sac or ppo')
    args = parser.parse_args()

    # Initialize WandB if enabled
    if args.use_wandb:
        run_name = f"{args.agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project="Aliengo-training", name=run_name, config={
            "agent": args.agent,
            "episodes_n": 20000,
            "max_steps": 1000,
            "reset_noise_scale": 0.1,
            "frame_skip": 25,
        })
    else:
        print("WandB logging disabled.")

    # Initialize environment
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
        frame_skip=5,
        max_episode_steps=1000,
        render_mode = 'human',
    )

    print(f'Environment name: Aliengo({env.spec.id})')
    state_dim = env.observation_space.shape[0]
    print(f'Observation space: {state_dim}')
    action_dim = env.action_space.shape[0]
    print(f'Action space: {action_dim}')

    # Initialize agent
    if args.agent == 'sac':
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
    else:  # ppo
        agent = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=0.99,
            batch_size=128,
            epsilon=0.2,
            epoch_n=30,
            pi_lr=1e-4,
            v_lr=5e-4
        )

    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env)

    # Train the agent
    total_rewards = train(agent, env, 
                         load_model_path=f"aliengo_models/aliengo_policy.pth", 
                         use_wandb=args.use_wandb)

    # Close environment
    env.close()

    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(total_rewards)
    plt.title(f'Total Rewards ({args.agent.upper()})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid()
    plt.savefig(f'aliengo_models/{args.agent}_rewards.png')
    plt.close()

    # Finish WandB run if enabled
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()