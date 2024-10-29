# import numpy as np

# class MarkovChain(object):
#     def __init__(self, transition_prob):
#         """
#         Initialize the MarkovChain instance.

#         Parameters
#         ----------
#         transition_prob: dict
#             A dict object representing the transition
#             probabilities in Markov Chain.
#             Should be of the form:
#                 {'state1': {'state1': 0.1, 'state2': 0.4},
#                     'state2': {...}}
#         """
#         self.transition_prob = transition_prob
#         self.states = list(transition_prob.keys())

#     def next_state(self, current_state):
#         """
#         Returns the state of the random variable at the next time
#         instance.

#         Parameters
#         ----------
#         current_state: str
#             The current state of the system.
#         """
#         return np.random.choice(
#             self.states,
#             p=[self.transition_prob[current_state][next_state]
#                 for next_state in self.states]
#         )

#     def generate_states(self, current_state, no=10):
#         """
#         Generates the next states of the system.

#         Parameters
#         ----------
#         current_state: str
#             The state of the current random variable.

#         no: int
#             The number of future states to generate.
#         """
#         future_states = []
#         for i in range(no):
#             next_state = self.next_state(current_state)
#             future_states.append(next_state)
#             current_state = next_state
#         return future_states


import numpy as np

# Список можливих станів (символів)
states = ["h", "e", "l", "o", " ", "w", "r", "d"]

# Матриця переходів між станами (ймовірності переходів)
transition_matrix = np.array(
    [
        [0, 0.2, 0.4, 0.2, 0.2, 0, 0, 0],
        [0.2, 0, 0.2, 0.2, 0.2, 0.2, 0, 0],
        [0, 0.2, 0, 0.2, 0.2, 0.2, 0.2, 0],
        [0, 0, 0.2, 0, 0.2, 0.2, 0.2, 0.2],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3],
        [0, 0, 0, 0, 0.2, 0, 0.8, 0],
        [0, 0, 0, 0, 0, 0.4, 0, 0.6],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)

# Початковий стан
initial_state = " "

# Генерація фрази
current_state = initial_state
generated_phrase = [current_state]
num_steps = 20

for _ in range(num_steps):
    next_state = np.random.choice(
        states, p=transition_matrix[states.index(current_state)]
    )
    generated_phrase.append(next_state)
    current_state = next_state

# Об'єднати сгенеровану фразу в одну рядок
generated_phrase = "".join(generated_phrase)

# Вивести результат
print(generated_phrase)
