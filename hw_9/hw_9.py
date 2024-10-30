import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Функція для обчислення функції вартості
def compute_value_function(env, policy, gamma=0.9, theta=1e-6):
    value_function = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for state in range(env.observation_space.n):
            v = 0
            action = policy[state]
            for prob, next_state, reward, done in env.P[state][action]:
                v += prob * (reward + gamma * value_function[next_state])
            delta = max(delta, abs(v - value_function[state]))
            value_function[state] = v
        if delta < theta:
            break
    return value_function


# Функція для виконання ітерації за політикою
def policy_iteration(env, gamma=0.9, theta=1e-6):
    policy = np.random.choice(env.action_space.n, env.observation_space.n)
    while True:
        value_function = compute_value_function(env, policy, gamma, theta)
        policy_stable = True
        for state in range(env.observation_space.n):
            old_action = policy[state]
            action_values = np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                for prob, next_state, reward, done in env.P[state][action]:
                    action_values[action] += prob * (
                        reward + gamma * value_function[next_state]
                    )
            best_action = np.argmax(action_values)
            if best_action != old_action:
                policy_stable = False
            policy[state] = best_action
        if policy_stable:
            break
    return policy, value_function


# Ініціалізація середовища FrozenLake з режимом рендерингу 'rgb_array'
env = gym.make(
    "FrozenLake-v1",
    desc=None,
    map_name="4x4",
    is_slippery=True,
    render_mode="rgb_array",
)

# Запускаємо ітерацію за політикою для отримання оптимальної політики
optimal_policy, optimal_value_function = policy_iteration(env)

print("Оптимальна політика:", optimal_policy)
print("Функція вартості для оптимальної політики:", optimal_value_function)

# Налаштовуємо фігуру для анімації
fig, ax = plt.subplots()
frames = []

# Ініціалізуємо початковий стан
observation, _ = env.reset()
img = env.render()
frames.append([plt.imshow(img, animated=True)])

# Виконуємо дії згідно з оптимальною політикою та зберігаємо кожен кадр
done = False
steps = 0  # Лічильник кроків для обмеження тривалості епізоду
while not done and steps < 100:
    # Перетворюємо observation на ціле число, якщо це необхідно
    if isinstance(observation, (np.ndarray, list)):
        observation = int(observation[0])

    action = optimal_policy[observation]
    observation, reward, done, _, _ = env.step(action)

    # Додаємо новий кадр для анімації
    img = env.render()
    frame = plt.imshow(img, animated=True)
    frames.append([frame])
    steps += 1

    if done:
        print(
            "Агент досягнув мети!"
            if reward > 0
            else "Агент потрапив у пастку або завершив епізод без досягнення мети."
        )

# Створюємо анімацію
ani = animation.ArtistAnimation(fig, frames, interval=500, blit=True, repeat=False)

# Відображаємо анімацію
plt.show()

# Закриваємо середовище
env.close()
