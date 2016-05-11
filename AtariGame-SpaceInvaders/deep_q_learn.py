import lasagne
import numpy as np
import theano
import theano.tensor as T
from statistics import Statistics

class DeepQLearner:
	"""
	Deep Q-learning network using Lasagne.
	"""

	def __init__(self, input_width, input_height, num_actions,
				 num_frames, discount, learning_rate, rho,
				 rms_epsilon, momentum, clip_delta, freeze_interval,
				 batch_size, network_type, update_rule,
				 batch_accumulator, rng, input_scale=255.0):

		self.input_width = input_width
		self.input_height = input_height
		self.num_actions = num_actions
		self.num_frames = num_frames
		self.batch_size = batch_size
		self.discount = discount
		self.rho = rho
		self.lr = learning_rate
		self.rms_epsilon = rms_epsilon
		self.momentum = momentum
		self.clip_delta = clip_delta
		self.freeze_interval = freeze_interval
		self.rng = rng

		self.callback = None

		lasagne.random.set_rng(self.rng) #set the seed

		self.update_counter = 0

		# build the network as described in the presentation (PPT)
		self.l_out = self.build_network(network_type, input_width, input_height,
										num_actions, num_frames, batch_size)

		# 4-dimensional ndarray (similar to prestates in memory_store)
		states = T.tensor4('states')
		# 4-dimensional ndarray (similar to poststates in memory_store)
		next_states = T.tensor4('next_states') 
		rewards = T.col('rewards')
		actions = T.icol('actions')
		terminals = T.icol('terminals')

		# creating a shared object is like declaring global - it has be shared between functions that it appears in.
		# similar to prestates matrix construction in memory_store
		self.states_shared = theano.shared(
			np.zeros((batch_size, num_frames, input_height, input_width),
					 dtype=theano.config.floatX))

		self.next_states_shared = theano.shared(
			np.zeros((batch_size, num_frames, input_height, input_width),
					 dtype=theano.config.floatX))

		self.rewards_shared = theano.shared(
			np.zeros((batch_size, 1), dtype=theano.config.floatX),
			broadcastable=(False, True))

		self.actions_shared = theano.shared(
			np.zeros((batch_size, 1), dtype='int32'),
			broadcastable=(False, True))

		self.terminals_shared = theano.shared(
			np.zeros((batch_size, 1), dtype='int32'),
			broadcastable=(False, True))

		# compute an expression for the output of a single layer given its input
		# scaling turns grayscale (or) black and white to 1s and 0s (black OR white)

		q_vals = lasagne.layers.get_output(self.l_out, states / input_scale)
		
		"""shape of q_val? - controls?"""

		next_q_vals = lasagne.layers.get_output(self.l_out, next_states / input_scale)

		next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

		"""how or what does the gradient function do?"""
		
		#perform this step: Q(st,a) = rimm + gamma*[ max(a{t+1}) Q(s{t+1}, a{t+1})]
		#					col. of ones with same dim. as terminals								max element in each row?
		target = (rewards + (T.ones_like(terminals) - terminals) * self.discount * T.max(next_q_vals, axis=1, keepdims=True))
		
		#										col. matrix into row matrix|row. matrix into col matrix
		diff = target - q_vals[T.arange(batch_size), actions.reshape((-1,))].reshape((-1, 1))
		
		# basically, we need to choose that 'a' (action) which maximizes Q(s,a)
		""" Maybe this is why out flappy bird using rms didn't work.. try giving clip_delta as 1 or 2.."""
		if self.clip_delta > 0: 
			quadratic_part = T.minimum(abs(diff), self.clip_delta)
			linear_part = abs(diff) - quadratic_part
			loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
		else:
			loss = 0.5 * diff ** 2

		loss = T.mean(loss)

		""" find out what is in params """
		params = lasagne.layers.helper.get_all_params(self.l_out)  
		givens = {
			states: self.states_shared,
			next_states: self.next_states_shared,
			rewards: self.rewards_shared,
			actions: self.actions_shared,
			terminals: self.terminals_shared
		}

		if update_rule == 'rmsprop':
			""" learn more """
			updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho, self.rms_epsilon)
		elif update_rule == 'sgd':
			# param := param - learning_rate * gradient
			updates = lasagne.updates.sgd(loss, params, self.lr)
		else:
			print "Unrecognized update rule"
			sys.exit(1)

		if self.momentum > 0:
			updates = lasagne.updates.apply_momentum(updates, None, self.momentum)

		#							inputs,outputs
		self._train = theano.function([], [loss, q_vals], updates=updates,
									  givens=givens)
		self._q_vals = theano.function([], q_vals,
									   givens={states: self.states_shared})

	def build_network(self, network_type, input_width, input_height, output_dim, num_frames, batch_size):
		if network_type == "nips_cuda":
			return self.build_nips_network(input_width, input_height,
										   output_dim, num_frames, batch_size)
		elif network_type == "nips_dnn":
			return self.build_nips_network_dnn(input_width, input_height,
											   output_dim, num_frames,
											   batch_size)
		elif network_type == "linear":
			return self.build_linear_network(input_width, input_height,
											 output_dim, num_frames, batch_size)
		else:
			print "Unrecognized network"
			sys.exit(1)

	def build_nips_network(self, input_width, input_height, output_dim, num_frames, batch_size):
		"""
		Build a network based on google atari deep learning paper
		"""
		from lasagne.layers import cuda_convnet

		l_in = lasagne.layers.InputLayer(
			shape=(batch_size, num_frames, input_width, input_height)
		)

		l_conv1 = cuda_convnet.Conv2DCCLayer(
			l_in, #previous layer
			num_filters=16, #16 8x8 filters
			filter_size=(8, 8),
			stride=(4, 4),
			nonlinearity=lasagne.nonlinearities.rectify, #convolution layer
			# W is the weights and b is bias term
			W=lasagne.init.Normal(.01),
			b=lasagne.init.Constant(.1),
			dimshuffle=True
		)

		l_conv2 = cuda_convnet.Conv2DCCLayer(
			l_conv1, #previous layer
			num_filters=32, #32 4x4 filters
			filter_size=(4, 4),
			stride=(2, 2),
			nonlinearity=lasagne.nonlinearities.rectify, #convolution layer
			W=lasagne.init.Normal(.01),
			b=lasagne.init.Constant(.1),
			dimshuffle=True
		)

		l_hidden1 = lasagne.layers.DenseLayer(
			l_conv2, #previous layer
			num_units=256, #256 hidden units
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.Normal(.01),
			b=lasagne.init.Constant(.1)
		)

		l_out = lasagne.layers.DenseLayer(
			l_hidden1,
			num_units=output_dim,
			nonlinearity=None,
			W=lasagne.init.Normal(.01),
			b=lasagne.init.Constant(.1)
		)

		return l_out


	def build_nips_network_dnn(self, input_width, input_height, output_dim,
							   num_frames, batch_size):
		"""
		Build a network based on google atari deep learning paper
		"""

		from lasagne.layers import dnn

		l_in = lasagne.layers.InputLayer(
			shape=(batch_size, num_frames, input_width, input_height)
		)


		l_conv1 = dnn.Conv2DDNNLayer(
			l_in,
			num_filters=16,
			filter_size=(8, 8),
			stride=(4, 4),
			nonlinearity=lasagne.nonlinearities.rectify,
			#W=lasagne.init.HeUniform(),
			W=lasagne.init.Normal(.01),
			b=lasagne.init.Constant(.1)
		)

		l_conv2 = dnn.Conv2DDNNLayer(
			l_conv1,
			num_filters=32,
			filter_size=(4, 4),
			stride=(2, 2),
			nonlinearity=lasagne.nonlinearities.rectify,
			#W=lasagne.init.HeUniform(),
			W=lasagne.init.Normal(.01),
			b=lasagne.init.Constant(.1)
		)

		l_hidden1 = lasagne.layers.DenseLayer(
			l_conv2,
			num_units=256,
			nonlinearity=lasagne.nonlinearities.rectify,
			#W=lasagne.init.HeUniform(),
			W=lasagne.init.Normal(.01),
			b=lasagne.init.Constant(.1)
		)

		l_out = lasagne.layers.DenseLayer(
			l_hidden1,
			num_units=output_dim,
			nonlinearity=None,
			#W=lasagne.init.HeUniform(),
			W=lasagne.init.Normal(.01),
			b=lasagne.init.Constant(.1)
		)

		return l_out


	def build_linear_network(self, input_width, input_height, output_dim,
							 num_frames, batch_size):
		"""
		Build a simple linear learner.  Useful for creating
		tests that sanity-check the weight update code.
		"""

		l_in = lasagne.layers.InputLayer(
			shape=(batch_size, num_frames, input_width, input_height)
		)

		l_out = lasagne.layers.DenseLayer(
			l_in,
			num_units=output_dim,
			nonlinearity=None,
			W=lasagne.init.Constant(0.0),
			b=None
		)

		return l_out

	def train(self, states, actions, rewards, next_states, terminals):
		
		self.states_shared.set_value(states)
		self.next_states_shared.set_value(next_states)
		self.actions_shared.set_value(np.matrix(actions).T)
		self.rewards_shared.set_value(np.matrix(rewards,dtype=theano.config.floatX).T)
		self.terminals_shared.set_value(np.matrix(terminals).T)
		loss, _ = self._train()
		if self.callback:
			self.callback.on_train(loss)
		#print loss,_
		self.update_counter += 1
		#print "finished train step.sqrt of loss: ",np.sqrt(loss)
		return np.sqrt(loss)

	def q_vals(self, state):
		states = np.zeros((self.batch_size, self.num_frames, self.input_height,
						   self.input_width), dtype=theano.config.floatX)
		states[0, ...] = state
		self.states_shared.set_value(states)
		return self._q_vals()[0]

	def choose_action(self, state, epsilon):
		#print "predicting"
		q_vals = self.q_vals(state)
		#print "what to choose? ",q_vals
		return np.argmax(q_vals)
		
	def predict(self, state):
		q_vals = self.q_vals(state)
		return q_vals

	def reset_q_hat(self):
		all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
		lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)
