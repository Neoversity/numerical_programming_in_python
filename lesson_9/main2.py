import gym
import numpy as np

# Параметри Q-навчання
num_episodes = 1000
learning_rate = 0.1
gamma = 0.99
num_buckets = (6, 12)  # Кількість дискретних сегментів для кожної осі в просторі станів

# Створюємо середовище
env = gym.make("CartPole-v1")


# Функція для дискретизації безперервних спостережень
def discretize_state(state):
    upper_bounds = [
        env.observation_space.high[0],
        0.5,
        env.observation_space.high[2],
        np.radians(50),
    ]
    lower_bounds = [
        env.observation_space.low[0],
        -0.5,
        env.observation_space.low[2],
        -np.radians(50),
    ]
    ratios = [
        (state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i])
        for i in range(len(state))
    ]
    new_obs = [int(round((num_buckets[i] - 1) * ratios[i])) for i in range(len(state))]
    new_obs = [min(num_buckets[i] - 1, max(0, new_obs[i])) for i in range(len(state))]
    return tuple(new_obs)


# Ініціалізація Q-таблиці
q_table = np.zeros(num_buckets + (env.action_space.n,))

for episode in range(num_episodes):
    # Витягуємо стан без інформації
    current_state = discretize_state(env.reset(return_info=False))
    done = False

    while not done:
        # Вибір дії за ε-жадібною стратегією
        if np.random.random() < 0.1:
            action = env.action_space.sample()  # випадкова дія
        else:
            action = np.argmax(q_table[current_state])  # дія з найбільшою Q-цінністю

        # Виконуємо дію в середовищі
        obs, reward, done, _ = env.step(action)
        new_state = discretize_state(obs)

        # Оновлення Q-таблиці
        q_table[current_state][action] += learning_rate * (
            reward + gamma * np.max(q_table[new_state]) - q_table[current_state][action]
        )

        current_state = new_state

env.close()
