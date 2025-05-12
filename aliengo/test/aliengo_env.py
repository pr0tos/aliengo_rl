import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
import os

class AliengoEnv(MujocoEnv, gym.utils.EzPickle):
    def __init__(self, xml_file="/home/eidel/projects/prog_rob/unitree_mujoco/unitree_robots/aliengo/scene.xml", frame_skip=5):
        # Путь к MJCF файлу модели Aliengo
        self.xml_file = xml_file if os.path.exists(xml_file) else os.path.join(
            os.path.dirname(__file__), "/home/eidel/projects/prog_rob/unitree_mujoco/unitree_robots/aliengo/scene.xml"
        )

        # Инициализация MuJoCo среды
        super().__init__(self.xml_file, frame_skip)

        # Определение пространства действий (например, 12 непрерывных значений для суставов)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # Определение пространства наблюдений (положение, скорость и т.д.)
        obs_size = self.data.qpos.size + self.data.qvel.size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        gym.utils.EzPickle.__init__(self)

    def step(self, action):
        # Выполнение действия
        self.do_simulation(action, self.frame_skip)

        # Получение наблюдений
        obs = self._get_obs()

        # Вычисление награды (например, за движение вперед)
        reward = self._compute_reward()

        # Проверка завершения эпизода
        terminated = self._is_terminated()
        truncated = False

        info = {"distance": self._compute_distance()}

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._get_obs()
        info = {"distance": self._compute_distance()}
        return obs, info

    def _get_obs(self):
        # Возвращает наблюдения (положение и скорость)
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def _compute_reward(self):
        # Пример награды: движение вперед
        forward_velocity = self.data.qvel[0]
        return forward_velocity - 0.01 * np.sum(np.square(self.data.ctrl))

    def _is_terminated(self):
        # Завершение, если робот упал (например, высота тела < 0.2)
        return self.data.qpos[2] < 0.2

    def _compute_distance(self):
        # Пример: расстояние до цели (если есть цель)
        return 0.0  # Замените на реальную логику