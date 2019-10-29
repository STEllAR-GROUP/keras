"""Utilities for backend functionality checks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from phylanx import Phylanx, PhylanxSession, execution_tree
from .common import floatx
from .common import epsilon
from .common import normalize_data_format
from collections import defaultdict

PhylanxSession.init(1)

from phylanx.plugins.keras import in_top_k


def variable(value, dtype=None, name=None, constraint=None):
	if dtype is None:
		dtype = floatx()
	if constraint is not None:
		raise TypeError("Constraint is the projection function to be "
						"applied to the variable after an optimizer update")
	if isinstance(value, execution_tree.variable):
		return value
	return execution_tree.variable(value, dtype=dtype, name=name)


def eval(x):
	return x.eval()

_LEARNING_PHASE = True

def learning_phase():
	return _LEARNING_PHASE


def set_learning_phase(value):
	global _LEARNING_PHASE
	_LEARNING_PHASE = value


@Phylanx
def eye_eager(n, m, dtype="float64", name=None):
	return np.eye(n, m, dtype=dtype)

def eye(size, dtype=None, name=None):
	if isinstance(size, (list, tuple)):
		n, m = size
	else:
		n, m = size, size
	return variable(eye_eager.lazy(n, m, dtype))


@Phylanx
def ones_eager(shape, dtype=None, name=None):
	return np.ones(shape, dtype=dtype)

def ones(shape, dtype=None, name=None):
	return ones_eager.lazy(shape, dtype)


@Phylanx
def zeros_eager(shape, dtype=None, name=None):
	return np.zeros(shape, dtype=dtype)

def zeros(shape, dtype=None, name=None):
	return zeros_eager.lazy(shape, dtype)


@Phylanx
def ones_like_eager(x, dtype=None, name=None):
	return np.ones_like(x, dtype=dtype)

def ones_like(x, dtype=None, name=None):
	return ones_like_eager.lazy(x)


@Phylanx
def zeros_like_eager(x, dtype=None, name=None):
	return np.zeros_like(x, dtype=dtype)

def zeros_like(x, dtype=None, name=None):
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
def switch_eager(condition, then_expression, else_expression):
	return switch(condition,then_expression, else_expression)

def switch(condition, then_expression, else_expression):
	return switch_eager.lazy(condition, then_expression, else_expression)


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
	if level < 0:
		raise ValueError("the level for dropout should be non-negative")
	return dropout_eager(x, level, noise_shape, seed)


@Phylanx
def relu_eager(x, alpha, max_value, threshold):
	return relu(x, alpha, max_value, threshold)

def relu(x, alpha=0.0, max_value=None, threshold=0.0):
	return relu_eager.lazy(x, alpha, max_value, threshold)


@Phylanx
def softsign_eager(x):
	return softsign(x)

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


@Phylanx
def categorical_crossentropy_eager(target, output, from_logits, axis):
	return categorical_crossentropy(target, output, from_logits, axis)[0]

def categorical_crossentropy(target, output, from_logits=False, axis=-1):
	return categorical_crossentropy_eager.lazy(target, output, from_logits, axis)


@Phylanx
def binary_crossentropy_eager(target, output, from_logits):
	return binary_crossentropy(target, output, from_logits)[0]

def binary_crossentropy(target, output, from_logits=False):
	return binary_crossentropy_eager.lazy(target, output, from_logits)


#@Phylanx
#def sparse_categorical_crossentropy_eager(target, output, from_logits, axis):
#	return sparse_categorical_crossentropy(target, output, from_logits, axis)[0]

#def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
#	return sparse_categorical_crossentropy_eager.lazy(target, output, from_logits, axis)


@Phylanx
def in_top_k_eager(predictions, targets, k):
	return in_top_k(predictions, targets, k)

def in_top_k(predictions, targets, k):
	return in_top_k_eager.lazy(predictions, targets, k)


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


@Phylanx
def resize_images_eager(x, height_factor, width_factor, interpolation):
	return resize_images(x, height_factor, width_factor, interpolation)

def resize_images(x, height_factor, width_factor, data_format, interpolation="nearest"):
	if data_format != "channels_last":
		raise ValueError("resize_images having a data format other than channels_last is not supported by Phylanx")
	return resize_images_eager.lazy(x, height_factor, width_factor, interpolation)


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


@Phylanx
def slice_eager(x, start, size):
	indices = [[i, i+j] for i, j in zip(start, size)]
	return tuple_slice(x, indices)

def slice(x, start, size):
	if len(start) != len(size):
		raise ValueError("start and size arguments should have the same shape")
	return slice_eager.lazy(x, start, size)


@Phylanx
def one_hot_eager(indices, num_classes):
	return one_hot(indices, num_classes)

def one_hot(indices, num_classes):
	return one_hot_eager.lazy(indices, num_classes)


# tested in map_fn and gradient
@Phylanx
def sum_eager(x, axis=None, keepdims=None):
	return np.sum(x, axis, keepdims)

sum = Phylanx.lazy(sum_eager)


# 4d, 5d
@Phylanx
def stack_eager(x, axis):
	return np.stack(x, axis=axis)

def stack(x, axis=0):
	return stack_eager.lazy(x, axis)


@Phylanx
def map_fn_eager(fn, elems):
	return fmap(fn, elems)

def map_fn(fn, elems, dtype=None):
	return variable(map_fn_eager(fn, elems), dtype=dtype)

@Phylanx
def foldl_eager(fn, elems, initializer, name):
	return fold_left(fn, initializer, elems)

def foldl(fn, elems, initializer=None, name=None):
	return foldl_eager.lazy(fn, elems, initializer, name)


@Phylanx
def foldr_eager(fn, elems, initializer, name):
	return fold_right(fn, initializer, elems)

def foldr(fn, elems, initializer=None, name=None):
	return foldr_eager.lazy(fn, elems, initializer, name)


@Phylanx
def constant_eager(value, dtype, shape):
	return constant(value, shape)

def constant(value, dtype=None, shape=None, name=None):
	return constant_eager.lazy(value, dtype, shape)


@Phylanx
def arange_eager(start, stop, step, dtype):
	return np.arange(start, stop=stop, step=step, dtype=dtype)

def arange(start, stop=None, step=1, dtype='int32'):
	return variable(arange_eager.lazy(start, stop, step, dtype),dtype=dtype)


#returns a list and asserted with a tuple
@Phylanx
def _int_shape(x):
	return np.shape(x)

def int_shape(x):
	return tuple(_int_shape(x))

def shape(x):
	return tuple(_int_shape(x))


def get_variable_shape(x):
	return int_shape(x)

def get_value(x):
	return eval(x)

def print_tensor(x, message=''):
	print(eval(x), message)
	return x

@Phylanx
def count_params(x):
	return np.size(x)


def dtype(x):
	if isinstance(x, execution_tree.variable):
		return x.dtype
	return execution_tree.variable(x, dtype).dtype


@Phylanx
def max_pool2d_eager(x, pool_size, strides, padding):
	return max_pool2d(x, pool_size, padding, strides)

@Phylanx
def avg_pool2d_eager(x, pool_size, strides, padding):
	return avg_pool2d(x, pool_size, padding, strides)

def pool2d(x, pool_size, strides=(1, 1), padding='valid', data_format=None, pool_mode='max'):
	if data_format != "channels_last":
		raise ValueError("pool2d having a data format other than channels_last is not supported by Phylanx")
	if pool_mode == "max":
		return max_pool2d_eager.lazy(x, pool_size, strides, padding)
	elif pool_mode == "avg":
		return avg_pool2d_eager.lazy(x, pool_size, strides, padding)
	raise ValueError("the `pool_mode` can be either `max` or `avg`")


def is_sparse(tensor):
	raise TypeError("sparse tensors are not supported by this version of Phylanx")


def to_dense(tensor):
	return tensor


@Phylanx
def in_train_phase_eager(x, alt, training=None):
	if training == 1 or training == True:
		return x()
	else:
		return alt()


def in_train_phase(x, alt, training=None):
	if training is None:
		training = learning_phase()
	return in_train_phase_eager.lazy(x, alt, training)

def in_test_phase(x, alt, training=None):
	return in_train_phase(alt, x, training=training)


@Phylanx
def ctc_decode_eager(y_pred, input_length, greedy=True, beam_width=100,
					 top_paths=1):
	return ctc_decode(y_pred, input_length, greedy, beam_width, top_paths)

def ctc_decode(y_pred, input_length, greedy=True, beam_width=100,
			   top_paths=1, merge_repeated=False):
	return ctc_decode_eager.lazy(y_pred, input_length, greedy, beam_width, top_paths)


def update_add(x, increment):
	x += increment
	return x


def update_sub(x, decrement):
	x -= decrement
	return x


def update(x, new_x):
	x = new_x
	return x


def moving_average_update(x, value, momentum):
	x.update_moving_average(value, momentum)
	return x


@Phylanx
def conv1d_eager(x, kernel, strides, padding, dilation_rate):
	return conv1d(x, kernel, padding, strides, dilation_rate)

def conv1d(x, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1):
	if data_format != "channels_last":
		raise ValueError("conv1d having a data format other than channels_last is not supported by Phylanx")
	return conv1d_eager.lazy(x, kernel, strides, padding, dilation_rate)


@Phylanx
def separable_conv1d_eager(x, depthwise_kernel, pointwise_kernel, strides, padding, dilation_rate):
	return separable_conv1d(x, depthwise_kernel, pointwise_kernel, padding, strides, dilation_rate)

def separable_conv1d(x, depthwise_kernel, pointwise_kernel, strides=1, padding='valid', data_format=None, dilation_rate=1):
	if data_format != "channels_last":
		raise ValueError("separable_conv1d having a data format other than channels_last is not supported by Phylanx")
	return separable_conv1d_eager.lazy(x, depthwise_kernel, pointwise_kernel, strides, padding, dilation_rate)


@Phylanx
def spatial_2d_padding_eager(x, padding):
	return spatial_2d_padding(x, padding)

def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
	if data_format != "channels_last":
		raise ValueError("spatial_2d_padding having a data format other than channels_last is not supported by Phylanx")
	return spatial_2d_padding_eager.lazy(x, padding)


@Phylanx
def bias_add_eager(x, bias):
	return bias_add(x, bias)

def bias_add(x, bias, data_format=None):
	if data_format != "channels_last":
		raise ValueError("bias_add having a data format other than channels_last is not supported by Phylanx")
	return bias_add_eager.lazy(x, bias)


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
	print("shape",shape)
	if shape:
		if not shape[0]:
			value = np.zeros((1,shape[1]))
	if dtype is None:
		dtype = floatx()
	return execution_tree.variable(value, dtype=dtype, name="placeholder_")

def is_placeholder(x):
	return x.name=="placeholder_"

def name_scope(name):
	return name

def is_tensor(x):
	return True

def is_keras_tensor(x):
	if not is_tensor(x):
		raise ValueError('Unexpectedly found an instance of type `' +
						 str(type(x)) + '`. '
						 'Expected a symbolic tensor instance.')
	return hasattr(x, '_keras_history')

# not tested in the backend, should work on both variables and placeholders
#@Phylanx
#def ndim_eager(x):
#	return ndim(x)

def ndim(x):
	return len(shape(x))

_UID_PREFIXES = defaultdict(int)
def get_uid(prefix=''):
	_UID_PREFIXES[prefix] += 1
	return _UID_PREFIXES[prefix]

@Phylanx
def cast_eager(x, dtype):
	return astype(x,dtype)

cast = Phylanx.lazy(cast_eager)
