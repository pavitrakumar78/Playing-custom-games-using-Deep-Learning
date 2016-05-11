import numpy as np
import random

class ReplayMemory:
	def __init__(self, size=1000000, screen_height=84,screen_width=84,history_length=4,batch_size=32,min_reward=-1,max_reward=1):
		self.size = size
		# preallocate memory
		self.actions = np.empty(self.size, dtype = np.uint8)
		self.rewards = np.empty(self.size, dtype = np.integer)
		self.terminals = np.empty(self.size, dtype = np.bool)
		self.history_length = history_length
		self.dims = (screen_height, screen_width)
		self.batch_size = batch_size
		self.min_reward = float(min_reward)
		self.max_reward = float(max_reward)
		self.count = 0
		self.current = 0
		self.screens = np.empty((self.size,screen_height,screen_width), dtype = np.uint8) 

		self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.uint8)
		self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.uint8)


	def add(self, action, reward, screen, terminal):
		assert screen.shape == self.dims
		self.actions[self.current] = action
		# clip reward between -1 and 1
		if self.min_reward and reward < self.min_reward:
			reward = max(reward, self.min_reward)
		if self.max_reward and reward > self.max_reward:
			reward = min(reward, self.max_reward)
		self.rewards[self.current] = reward
		# screen is 84x84 size
		self.screens[self.current, ...] = screen
		self.terminals[self.current] = terminal
		self.count = max(self.count, self.current + 1)
		self.current = (self.current + 1) % self.size 

	def getState(self, index):
		index = index % self.count

		if index >= self.history_length - 1:
			return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
		else:
			indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]

			return self.screens[indexes, ...]

	def getCurrentState(self):
		# this is the input to the model to predict what move to make next
		# reuse first row of prestates in minibatch to minimize memory consumption
		self.prestates[0, ...] = self.getState(self.current - 1)
		#print self.getState(self.current - 1).shape, "is shape of getstate"
		#print self.prestates.shape,"is the shape of current state"
		current_state = self.getState(self.current - 1)
		return current_state

	def getMinibatch(self):
		# memory must include poststate, prestate and history
		assert self.count > self.history_length
		# sample random indexes
		indexes = []
		while len(indexes) < self.batch_size:
			while True:
				index = random.randint(self.history_length, self.count - 1)
				if index >= self.current and index - self.history_length < self.current:
					continue
				if self.terminals[(index - self.history_length):index].any():
					continue
				# otherwise use this index
				break
			
			# fill the "batch"
			self.prestates[len(indexes), ...] = self.getState(index - 1)
			self.poststates[len(indexes), ...] = self.getState(index)
			indexes.append(index)

		# copy actions, rewards and terminals directly
		actions = self.actions[indexes]
		rewards = self.rewards[indexes]
		terminals = self.terminals[indexes]
		return self.prestates, actions, rewards, self.poststates, terminals
