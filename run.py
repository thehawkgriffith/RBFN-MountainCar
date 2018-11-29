import numpy as np
def run(env, agent, epsilon, gamma):
	done = False
	iters = 0
	total_reward = 0
	state = env.reset()
	while not done and iters < 10000:
		action = agent.sample_action(state, epsilon)
		state_prime, reward, done, info = env.step(action)
		actions_prime = agent.predict(state_prime)
		Q = reward + gamma * np.max(actions_prime)
		agent.update(state, action, Q)
		iters += 1
		total_reward += reward
	return total_reward


