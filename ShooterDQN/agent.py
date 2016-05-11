from memory_store import ReplayMemory
import random
import time
import sys
import logging
import numpy as np
import pygame
import time
import matplotlib.pyplot as plt
import datetime

class Agent:
	def __init__(self, environment, replay_memory, deep_q_network):
		self.env = environment
		self.mem = replay_memory
		self.net = deep_q_network
		self.num_actions = self.env.numActions()
		self.random_starts = 30
		self.history_length = 4

		self.exploration_rate_start = float(1.0)
		self.exploration_rate_end = float(0.1)
		self.exploration_decay_steps = float(1000000)
		self.exploration_rate_test = float(0.05)

		self.total_train_steps = 0

		self.train_frequency = 4
		self.train_repeat = 1

		self.callback = None
		
		
		self.total_score = 0.0
		self.total_moves = 0

	def ALLrestartRandom(self):
		#uncomment whichever is needed
		#for flappy bird and shooter
		self._restartRandomF()

		#for atari games
		#self._restartRandom()

	def _restartRandom(self):
		self.env.restart()
		# perform random number of dummy actions to produce more stochastic games
		for i in xrange(random.randint(self.history_length, self.random_starts) + 1):
			reward = self.env.act(0)
			screen = self.env.getScreen()
			terminal = self.env.isTerminal()
			assert not terminal, "terminal state occurred during random initialization"
			# add dummy states to buffer to guarantee history_length screens
			self.mem.add(0, reward, screen, terminal)
			
	def _restartRandomF(self):
		self.env.restart()
		for i in xrange(5):
			reward = self.env.act(random.randrange(self.num_actions))
			screen = self.env.getScreen()
			terminal = self.env.isTerminal()
			assert not terminal, "terminal state occurred during random initialization"
			# add dummy states to buffer to guarantee history_length screens
			self.mem.add(0, reward, screen, terminal)		

			
	def _exploration_rate(self):
		# calculate decaying exploration rate - very slowly slowly go from 0.1 to 1.0 over decay_steps steps
		# exploration rate controls when to use the model to predict and when to randomly select a action
		if self.total_train_steps < self.exploration_decay_steps:
			return self.exploration_rate_start - self.total_train_steps * (self.exploration_rate_start - self.exploration_rate_end) / float(self.exploration_decay_steps)
		else:
			return self.exploration_rate_end

	def step(self, exploration_rate,play=False):
		#perform a single step (a single action)

		self.total_moves += 1
		if random.random() < exploration_rate:
			# randomly select a action based on exploration_rate
			action = random.randrange(self.num_actions)
		else:
			# choose action with highest Q-value
			state = self.mem.getCurrentState()
			# for convenience getCurrentState() returns minibatch
			action = self.net.choose_action(state,0)
			#print "Prediction chosen",action 
			moves_print = "choosing to predict stuff! Exploration rate:	"+str(exploration_rate)+" action is: "+str(action)
			#print moves_print
			#moves_chosen = open("moves.txt","a")
			#moves_chosen.write(moves_print+"\n")
			#moves_chosen.close()
		
		reward = self.env.act(action)
		screen = self.env.getScreen()
		terminal = self.env.isTerminal()

		self.total_score += reward

		self.mem.add(action, reward, screen, terminal)

		# if game over, then restart it
		if terminal:
			a = datetime.datetime.now().replace(microsecond=0)
			print_stats = "Time: "+str(a)+"	Total moves made in a game: "+str(self.total_moves)+"		Total Score: "+str(self.total_score)
			#print print_stats
			log = open("log_stuff.txt","a")
			log.write(print_stats+"\n")
			log.close()
			self.total_score = 0
			self.total_moves = 0
			self.ALLrestartRandom()

		# call callback to record statistics
		if self.callback:
			self.callback.on_step(action, reward, terminal, screen, exploration_rate)

		return terminal

	def play_random(self, random_steps):
	# randomly play for given no. of steps
		for i in xrange(random_steps):
			# use exploration rate 1 = completely random
			self.step(1)
			print "random step",i

	def train(self, train_steps, epoch = 0):
		traincount = 0
		assert self.mem.count >= self.history_length, "Not enough history in replay memory, increase random steps."
		# play given number of steps
		for i in xrange(train_steps):
			# perform a single game step
			self.step(self._exploration_rate())
			# train after every train_frequency (4)	steps
			if i % self.train_frequency == 0:
				# train for train_repeat times
				for j in xrange(self.train_repeat):
					# sample minibatch
					minibatch = self.mem.getMinibatch()
					loss = self.net.train(minibatch[0],minibatch[1],minibatch[2],minibatch[3],minibatch[4])	
					traincount +=1
					log = open("model_loss.txt","a")
					log.write("Train Count: "+str(traincount)+"			Loss: "+str(loss)+"\n")
					log.close()
			log = open("model_loss.txt","a")
			#log.write("---------------------------------------------------------------------------------\n")
			log.write("TOTAL TRAIN STEP: "+str(self.total_train_steps)+"\n")
			log.close()
			self.total_train_steps += 1

	def test(self, test_steps, epoch = 0):
		self.ALLrestartRandom()
		# play given number of steps
		for i in xrange(test_steps):
			# perform game step
			self.step(self.exploration_rate_test)

	def play(self, num_games):
		self.ALLrestartRandom()
		for i in xrange(num_games):
			# play until terminal state
			while not self.step(self.exploration_rate_test,play=True):
				pass
