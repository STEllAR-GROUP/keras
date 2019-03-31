"""Utilities for backend functionality checks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from phylanx import Phylanx, PhylanxSession, execution_tree
from .common import floatx, set_floatx
from .common import epsilon, set_epsilon
from .common import normalize_data_format

PhylanxSession.init(1)


def variable(value, dtype=None, name=None, constraint=None):
	if constraint is not None:
		raise TypeError("Constraint is the projection function to be "
						"applied to the variable after an optimizer update")
	return execution_tree.variable(np.array(value, dtype=dtype),
								dtype=dtype, name=name)


def eval(x):
	return x.eval()

_LEARNING_PHASE = True

def learning_phase():
	return _LEARNING_PHASE


def set_learning_phase(value):
	global _LEARNING_PHASE
	_LEARNING_PHASE = value


# not tested in the backend, should work on both variables and placeholders
@Phylanx
def ndim_eager(x):
	return ndim(x)

def ndim(x):
	return ndim_eager.lazy(x)




@Phylanx
def eye_eager(size, dtype, name):
	return np.eye(size)

def eye(size, dtype=None, name=None):
	return eye_eager.lazy(size)


# 4d
@Phylanx
def ones_eager(shape, dtype, name):
	return np.ones(shape)

def ones(shape, dtype=floatx(), name=None):
	return ones_eager.lazy(shape)


# 4d
@Phylanx
def zeros_eager(shape, dtype, name):
	return np.zeros(shape)

def zeros(shape, dtype=floatx(), name=None):
	return zeros_eager.lazy(shape)


# 4d
@Phylanx
def ones_like_eager(x, dtype, name):
	return np.ones_like(x)

def ones_like(x, dtype=floatx(), name=None):
	return ones_like_eager.lazy(x)


# 4d
@Phylanx
def zeros_like_eager(x, dtype, name):
	return np.zeros_like(x)

def zeros_like(x, dtype=floatx(), name=None):
	return zeros_like_eager.lazy(x)


@Phylanx
def dot_eager(x, y):
	return np.dot(x, y)

def dot(x, y):
	return dot_eager.lazy(x, y)


@Phylanx
def batch_dot_eager(x, y, axes):
	return batch_dot(x, y, axes)

def batch_dot(x, y, axes=None):
	return batch_dot_eager.lazy(x, y, axes)


@Phylanx
def transpose_eager(x):
	return np.transpose(x)

def transpose(x):
	return transpose_eager.lazy(x)


@Phylanx
def reverse_eager(x, axes):
	return np.flip(x, axes)

def reverse(x, axes):
	return reverse_eager.lazy(x, axes)


@Phylanx
def phylanx_random_uniform_variable(shape, low, high):
	return random(shape, ["uniform", low, high])

def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
	return execution_tree.variable(phylanx_random_uniform_variable(shape, low, high))


@Phylanx
def phylanx_random_normal_variable(shape, mean, scale):
	return random(shape, ["normal", mean, scale])

def random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None):
	return execution_tree.variable(phylanx_random_normal_variable(shape, mean, scale))


@Phylanx
def concatenate_eager(tensors, axis):
	return np.concatenate(tensors, axis)

def concatenate(tensors, axis=-1):
	return concatenate_eager.lazy(tensors, axis)


@Phylanx
def reshape_eager(x, shape):
	return np.reshape(x, shape)

def reshape(x, shape):
	return reshape_eager.lazy(x, shape)


@Phylanx
def permute_dimensions_eager(x, pattern):
	return np.transpose(x, pattern)

def permute_dimensions(x, pattern):
	return permute_dimensions_eager.lazy(x, pattern)


@Phylanx
def repeat_eager(x, n):
	y = np.expand_dims(x, 1)
	return np.repeat(y, n, 1)

def repeat(x, n):
	return repeat_eager.lazy(x, n)


@Phylanx
def flatten_eager(x):
	return flatten(x)

def flatten(x):
	return flatten_eager.lazy(x)


@Phylanx
def batch_flatten_eager(x):
	return np.reshape(x, [shape(x)[0], -1])

def batch_flatten(x):
	return batch_flatten_eager.lazy(x)


# needs 4d
@Phylanx
def expand_dims_eager(x, axis):
	return np.expand_dims(x, axis)

def expand_dims(x, axis=-1):
	return expand_dims_eager.lazy(x, axis)


@Phylanx
def squeeze_eager(x, axis):
	return np.squeeze(x, axis)

def squeeze(x, axis):
	return squeeze_eager.lazy(x, axis)


#placeholders
@Phylanx
def repeat_elements_eager(x, rep, axis):
	return np.repeat(x, rep, axis)

def repeat_elements(x, rep, axis):
	return repeat_elements_eager.lazy(x, rep, axis)


#placeholders
@Phylanx
def tile_eager(x, n):
	return np.tile(x, n)

def tile(x, n):
	return tile_eager.lazy(x, n)


# float.shape problem
@Phylanx
def max_eager(x, axis, keepdims):
	return np.amax(x, axis, keepdims)

def max(x, axis=None, keepdims=False):
	return max_eager.lazy(x, axis, keepdims)


@Phylanx
def min_eager(x, axis, keepdims):
	return np.amin(x, axis, keepdims)

def min(x, axis=None, keepdims=False):
	return min_eager.lazy(x, axis, keepdims)


@Phylanx
def mean_eager(x, axis, keepdims):
	return np.mean(x, axis, keepdims)

def mean(x, axis=None, keepdims=False):
	return mean_eager.lazy(x, axis, keepdims)


@Phylanx
def var_eager(x, axis, keepdims):
	return np.var(x, axis, keepdims)

def var(x, axis=None, keepdims=False):
	return var_eager.lazy(x, axis, keepdims)


@Phylanx
def std_eager(x, axis, keepdims):
	return np.std(x, axis, keepdims)

def std(x, axis=None, keepdims=False):
	return std_eager.lazy(x, axis, keepdims)


@Phylanx
def logsumexp_eager(x, axis, keepdims):
	return logsumexp(x, axis, keepdims)

def logsumexp(x, axis=None, keepdims=False):
	return logsumexp_eager.lazy(x, axis, keepdims)


@Phylanx
def prod_eager(x, axis, keepdims):
	return np.prod(x, axis, keepdims)

def prod(x, axis=None, keepdims=False):
	return prod_eager.lazy(x, axis, keepdims)


@Phylanx
def any_eager(x, axis, keepdims):
	return np.any(x, axis, keepdims)

def any(x, axis=None, keepdims=False):
	return any_eager.lazy(x, axis, keepdims)


@Phylanx
def all_eager(x, axis, keepdims):
	return np.all(x, axis, keepdims)

def all(x, axis=None, keepdims=False):
	return all_eager.lazy(x, axis, keepdims)


@Phylanx
def argmax_eager(x, axis):
	return np.argmax(x, axis)

def argmax(x, axis=-1):
	return argmax_eager.lazy(x, axis)


@Phylanx
def argmin_eager(x, axis):
	return np.argmin(x, axis)

def argmin(x, axis=-1):
	return argmin_eager.lazy(x, axis)


@Phylanx
def square_eager(x):
	return np.square(x)

def square(x):
	return square_eager.lazy(x)


@Phylanx
def abs_eager(x):
	return absolute(x)

def abs(x):
	return abs_eager.lazy(x)


#@Phylanx
#def sqrt_eager(x):
#	y = np.sqrt(x)
#	y[np.isnan(y)] = 0.
#	return y

#def sqrt(x):
#	return sqrt_eager.lazy(x)


@Phylanx
def exp_eager(x):
	return np.exp(x)

def exp(x):
	return exp_eager.lazy(x)


#passed although the data type should not be correct
@Phylanx
def round_eager(x):
	return rint(x)

def round(x):
	return round_eager.lazy(x)


@Phylanx
def sign_eager(x):
	return np.sign(x)

def sign(x):
	return sign_eager.lazy(x)


@Phylanx
def pow_eager(x, a):
	return np.power(x, a)

def pow(x, a=1.):
	return pow_eager.lazy(x, a)


@Phylanx
def clip_eager(x, min_value, max_value):
	return np.clip(x, min_value, max_value)

def clip(x, min_value, max_value):
	return clip_eager.lazy(x, min_value, max_value)


@Phylanx
def cos_eager(x):
	return np.cos(x)

def cos(x):
	return cos_eager.lazy(x)


@Phylanx
def sin_eager(x):
	return np.sin(x)

def sin(x):
	return sin_eager.lazy(x)


@Phylanx
def equal_eager(x, y):
	return x == y

def equal(x, y):
	return equal_eager.lazy(x, y)


@Phylanx
def not_equal_eager(x, y):
	return x != y

def not_equal(x, y):
	return not_equal_eager.lazy(x, y)


@Phylanx
def greater_eager(x, y):
	return x > y

def greater(x, y):
	return greater_eager.lazy(x, y)


@Phylanx
def greater_equal_eager(x, y):
	return x >= y

def greater_equal(x, y):
	return greater_equal_eager.lazy(x, y)


@Phylanx
def less_eager(x, y):
	return x < y

def less(x, y):
	return less_eager.lazy(x, y)


@Phylanx
def less_equal_eager(x, y):
	return x <= y

def less_equal(x, y):
	return less_equal_eager.lazy(x, y)


@Phylanx
def maximum_eager(x, y):
	return np.maximum(x, y)

def maximum(x, y):
	return maximum_eager.lazy(x, y)


@Phylanx
def minimum_eager(x, y):
	return np.minimum(x, y)

def minimum(x, y):
	return minimum_eager.lazy(x, y)


@Phylanx
def cumsum_eager(x, axis):
	return np.cumsum(x, axis)

def cumsum(x, axis=0):
	return cumsum_eager.lazy(x, axis)


@Phylanx
def cumprod_eager(x, axis):
	return np.cumprod(x, axis)

def cumprod(x, axis=0):
	return cumprod_eager.lazy(x, axis)


@Phylanx
def log_eager(x):
	return np.log(x)

def log(x):
	return log_eager.lazy(x)


@Phylanx
def _dropout(x, level, noise_shape, seed):
	if seed:
		set_seed(seed)
	noise = random(noise_shape if noise_shape else shape(x), ["bernoulli", 1 - level])
	return x * noise / (1 - level)

@Phylanx
def _no_dropout(x):
	return x

def dropout_eager(x, level, noise_shape, seed):
	if learning_phase():
		return _dropout.lazy(x, level, noise_shape, seed)

	# dropout is not considered outside the learning phase
	return _no_dropout.lazy(x)

def dropout(x, level, noise_shape=None, seed=None):
	return dropout_eager(x, level, noise_shape, seed)


@Phylanx
def relu_eager(x, alpha, max_value, threshold):
	return relu(x, alpha, max_value, threshold)

def relu(x, alpha=0.0, max_value=None, threshold=0.0):
	return relu_eager.lazy(x, alpha, max_value, threshold)


@Phylanx
def softsign_eager(x):
	return x / (1 + absolute(x))

def softsign(x):
	return softsign_eager.lazy(x)


@Phylanx
def softplus_eager(x):
	return softplus(x)

def softplus(x):
	return softplus_eager.lazy(x)


@Phylanx
def elu_eager(x, alpha):
	return elu(x, alpha)

def elu(x, alpha=1.):
	return elu_eager.lazy(x, alpha)


@Phylanx
def sigmoid_eager(x):
	return sigmoid(x)

def sigmoid(x):
	return sigmoid_eager.lazy(x)


@Phylanx
def hard_sigmoid_eager(x):
	return hard_sigmoid(x)

def hard_sigmoid(x):
	return hard_sigmoid_eager.lazy(x)


@Phylanx
def tanh_eager(x):
	return np.tanh(x)

def tanh(x):
	return tanh_eager.lazy(x)


# 4d
@Phylanx
def softmax_eager(x, axis):
	return softmax(x, axis)

def softmax(x, axis=-1):
	return softmax_eager.lazy(x, axis)


#@Phylanx
#def categorical_crossentropy_eager(target, output, from_logits, axis):
#	return categorical_crossentropy(target, output, from_logits)

#def categorical_crossentropy(target, output, from_logits=False, axis=-1):
#	return categorical_crossentropy_eager.lazy(target, output, from_logits, axis)


@Phylanx
def l2_normalize_eager(x, axis):
	return l2_normalize(x, axis)

def l2_normalize(x, axis=None):
	return l2_normalize_eager.lazy(x, axis)


@Phylanx
def random_normal_eager(shape, mean, stddev, dtype, seed):
	return random(shape, ["normal", mean, stddev])

def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
	return random_normal_eager.lazy(shape, mean, stddev)


@Phylanx
def random_uniform_eager(shape, minval, maxval, dtype=None, seed=None):
	return random(shape, ["uniform", minval, maxval])

def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
	return random_uniform_eager.lazy(shape, minval, maxval)


@Phylanx
def random_binomial_eager(shape, p, dtype=None, seed=None):
	return random(shape, ["bernoulli", p])

def random_binomial(shape, p=0.0, dtype=None, seed=None):
	return random_binomial_eager.lazy(shape, p)


@Phylanx
def truncated_normal_eager(shape, mean, stddev, dtype=None, seed=None):
	return random(shape, ["truncated_normal", mean, stddev])

def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
	return truncated_normal_eager.lazy(shape, mean, stddev)


#not representing the other interpolation, bilinear, 4d
@Phylanx
def resize_images_eager(x, height_factor, width_factor, data_format):
	if data_format == 'channels_first':
		x = np.repeat(x, height_factor, 2)
		x = np.repeat(x, width_factor, 3)
	elif data_format == 'channels_last':
		x = np.repeat(x, height_factor, 1)
		x = np.repeat(x, width_factor, 2)
	return x

def resize_images(x, height_factor, width_factor, data_format, interpolation='nearest'):
	return resize_images_eager.lazy(x, height_factor, width_factor, data_format)


# needs 5d
@Phylanx
def resize_volumes_eager(x, depth_factor, height_factor, width_factor, data_format):
	if data_format == 'channels_first':
		x = np.repeat(x, depth_factor, 2)
		x = np.repeat(x, height_factor, 3)
		x = np.repeat(x, width_factor, 4)
	elif data_format == 'channels_last':
		x = np.repeat(x, depth_factor, 1)
		x = np.repeat(x, height_factor, 2)
		x = np.repeat(x, width_factor, 3)
	return x

def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
	return resize_volumes_eager.lazy(x, depth_factor, height_factor, width_factor, data_format)


@Phylanx
def temporal_padding_eager(x, padding):
	return np.pad(x, [[0, 0], padding, [0, 0]], 'constant')

def temporal_padding(x, padding=(1, 1)):
	return temporal_padding_eager.lazy(x, padding)


#@Phylanx
#def slice_eager(x, indices):
#	if x.ndim == 1:
#		return slice(x, indices[0])
#	elif x.ndim == 2:
#		return slice(x, indices[0], indices[1])
#	elif x.ndim == 3:
#		return slice(x, indices[0], indices[1], indices[2])

#def slice(x, start, size):
#	indices = [[i, i+j] for i, j in zip(start, size)]
#	return slice_eager.lazy(x, indices)


@Phylanx
def one_hot_eager(indices, num_classes):
	return one_hot(indices, num_classes)

def one_hot(indices, num_classes):
	return one_hot_eager.lazy(indices, num_classes)


# tested in map_fn and gradient
@Phylanx
def sum_eager(x, axis, keepdims):
	return np.sum(x, axis, keepdims)

def sum(x, axis=None, keepdims=False):
	return sum_eager.lazy(x, axis, keepdims)


# 4d, 5d
@Phylanx
def stack_eager(x, axis):
	return np.stack(x, axis=axis)

def stack(x, axis=0):
	return stack_eager.lazy(x, axis)


@Phylanx
def constant_eager(value, dtype, shape):
	return constant(value, shape)

def constant(value, dtype=None, shape=None, name=None):
	return constant_eager.lazy(value, dtype, shape)


 #dtype problem
@Phylanx
def arange_eager(start, stop, step, dtype):
	return np.arange(start, stop, step)

def arange(start, stop=None, step=1, dtype='int32'):
	return arange_eager.lazy(start, stop, step, dtype)


#returns a list and asserted with a tuple
@Phylanx
def _int_shape(x):
	return np.shape(x)

def int_shape(x):
	return tuple(_int_shape(x))

def get_variable_shape(x):
	return int_shape(x)

def get_value(x):
	return eval(x)


@Phylanx
def count_params(x):
	return np.size(x)


def dtype(x):
	return execution_tree.variable(x).dtype


#@Phylanx
#def max_pool_eager(x, pool_size, strides, padding):
#	return max_pool(x, pool_size, padding, strides)

#def pool2d(x, pool_size, strides=(1, 1), padding='valid',
#		   data_format=None, pool_mode='max'):
#	#if data_format == 'channels_last':
#	#	if x.ndim == 4:
#	#		x = np.transpose(x, (0, 3, 1, 2))
#	#	else:
#	#		raise IndexError("Constraint is the projection function to be "
#	#					"applied to the variable after an optimizer update")

#	if pool_mode == "max":
#		z = []
#		for fourth in range(3):
#			y = []
#			for third in range(3):
#				y.append(max_pool_eager.lazy(x[fourth,third,:,:], pool_size, strides, padding))
#			y = np.stack(y, axis=0)
#			z.append(y)
#		z = np.stack(z,axis=0)
#		return z



