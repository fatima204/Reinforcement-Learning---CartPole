import gym  # install gym 0.25.2
import numpy as np

# install tensorflow version 2.12.0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam

# install keras.rl version 1.5.0
from rl.agents import DQNAgent  # pip install keras-rl2
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# additional imports:
# protobuf
# pygame to display the visualisation

env = gym.make("CartPole-v1")  # no render mode to prevent display while training

states = env.observation_space.shape[0]
actions = env.action_space.n

print(states)
print(actions)

model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="linear"))

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

print(env)
print(env.observation_space)

agent.compile(Adam(lr=0.001), metrics=["mae"])
agent.fit(env, nb_steps=100000, visualize=False, verbose=1)

results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()