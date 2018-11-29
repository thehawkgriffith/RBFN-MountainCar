from sklearn.linear_model import SGDRegressor
import numpy as np

class Agent():

	def __init__(self, env, feature_transformer, learning_rate):
		self.env = env
		self.feature_transformer = feature_transformer
		self.action_models = []
		for _ in range(env.action_space.n):
			action_model = SGDRegressor(learning_rate = learning_rate)
			action_model.partial_fit(self.feature_transformer.transform([self.env.reset()]), [0])
			self.action_models.append(action_model)

	def predict(self, state):
		feat_state = self.feature_transformer.transform([state])
		Q = np.stack([m.predict(feat_state) for m in self.action_models]).T
		return Q

	def update(self, state, action, Qvalue):
		feat_state = self.feature_transformer.transform([state])
		self.action_models[action].partial_fit(feat_state, [Qvalue])

	def sample_action(self, state, epsilon):
		if np.random.random() < epsilon:
			return self.env.action_space.sample()
		else:
			return np.argmax(self.predict(state))




