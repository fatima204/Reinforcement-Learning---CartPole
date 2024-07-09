import random
import gym  # version 0.25.2

# additional imports
# pyglet
# pygame

env = gym.make("CartPole-v1", render_mode="human")

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:  # when entering True, then you see how the cart fails completely
        action = random.choice([0, 1])
        n_state, reward, done, info = env.step(action)
        score += reward
        env.render() 

    print(f"Episode; {episode} Score: {score}")

env.close()
