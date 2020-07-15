import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from utils import DEFAULT_DTYPE, DEFAULT_DATA_FORMAT, NCHW_FORMAT, NHWC_FORMAT,\
    validate_data_format, HE_GAIN, DEFAULT_TRUNCATE_WEIGHTS, DEFAULT_USE_XLA


WEIGHTS_NAME = 'weights'
BIASES_NAME = 'biases'


biases_init = tf.zeros_initializer()


def weights_init(std):
    return tf.random_normal_initializer(stddev=std, seed=42)


def truncated_weights_init(std):
    return tf.initializers.TruncatedNormal(stddev=std, seed=42)


def select_initializer(truncate_weights, std):
    if truncate_weights:
        return truncated_weights_init(std)
    else:
        return weights_init(std)


class Scaled_Conv2d(Layer):

    def __init__(self, fmaps, kernel_size, stride=1, gain=HE_GAIN,
                 use_wscale=True, truncate_weights=DEFAULT_TRUNCATE_WEIGHTS,
                 dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, name=None):
        super(Scaled_Conv2d, self).__init__(dtype=dtype, name=name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.fmaps = fmaps
        self.kernel_size = kernel_size
        self.stride = stride
        self.gain = gain
        self.use_wscale = use_wscale
        self.truncate_weights = truncate_weights
        self.use_xla = use_xla

    def build(self, input_shape):
        if self.data_format == NCHW_FORMAT:
            self.channels_in = input_shape[1]
            self.strides = [1, 1, self.stride, self.stride]
        elif self.data_format == NHWC_FORMAT:
            self.channels_in = input_shape[-1]
            self.strides = [1, self.stride, self.stride, 1]

        self.wshape = [self.kernel_size, self.kernel_size, self.channels_in, self.fmaps]
        self.fan_in = np.prod(self.wshape[:-1])
        # He/LeCun init
        self.std = self.gain / np.sqrt(self.fan_in)
        if self.use_wscale:
            self.wscale = self.std
            initializer = select_initializer(self.truncate_weights, 1.)
        else:
            initializer = select_initializer(self.truncate_weights, self.std)

        self.w = self.add_weight(
            name=WEIGHTS_NAME,
            shape=self.wshape,
            initializer=initializer,
            trainable=True
        )

    @tf.function
    def call(self, x, *args, **kwargs):
        if self.use_wscale:
            weights = self.wscale * self.w
        else:
            weights = self.w

        return tf.nn.conv2d(
            x, weights, strides=self.strides, padding='SAME', data_format=self.data_format
        )


class Scaled_Linear(Layer):

    def __init__(self, units, gain=HE_GAIN,
                 use_wscale=True, truncate_weights=DEFAULT_TRUNCATE_WEIGHTS,
                 dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, name=None):
        super(Scaled_Linear, self).__init__(dtype=dtype, name=name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.units = units
        self.gain = gain
        self.use_wscale = use_wscale
        self.truncate_weights = truncate_weights
        self.use_xla = use_xla

    def build(self, input_shape):
        self.fan_in = np.prod(input_shape[1:])
        # He/LeCun init
        self.std = self.gain / np.sqrt(self.fan_in)
        if self.use_wscale:
            self.wscale = self.std
            initializer = select_initializer(self.truncate_weights, 1.)
        else:
            initializer = select_initializer(self.truncate_weights, self.std)

        self.w = self.add_weight(
            name=WEIGHTS_NAME,
            shape=[self.fan_in, self.units],
            initializer=initializer,
            trainable=True
        )

    @tf.function
    def call(self, x, *args, **kwargs):
        if len(x.shape) > 2:
            x = tf.reshape(x, [-1, self.fan_in])
        if self.use_wscale:
            weights = self.wscale * self.w
        else:
            weights = self.w

        return tf.linalg.matmul(x, weights)


class Bias(Layer):
    def __init__(self, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, name=None):
        super(Bias, self).__init__(dtype=dtype, name=name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.use_xla = use_xla

    def build(self, input_shape):
        self.is_linear_bias = len(input_shape) == 2

        if self.is_linear_bias:
            self.units = input_shape[1]
        else:
            if self.data_format == NCHW_FORMAT:
                self.bias_target_shape = [1, -1, 1, 1]
                self.units = input_shape[1]
            elif self.data_format == NHWC_FORMAT:
                self.bias_target_shape = [1, 1, 1, -1]
                self.units = input_shape[-1]

        self.b = self.add_weight(
            name=BIASES_NAME,
            shape=[self.units],
            initializer=biases_init,
            trainable=True
        )

    @tf.function
    def call(self, x, *args, **kwargs):
        if self.is_linear_bias:
            return x + self.b
        else:
            # Note: keep reshaping to allow easy weights decay between cpu and gpu models
            return x + tf.reshape(self.b, self.bias_target_shape)


class Upscale2d(Layer):

    def __init__(self, factor, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, name=None):
        super(Upscale2d, self).__init__(dtype=dtype, name=name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.factor = factor
        self.use_xla = use_xla

    @tf.function
    def call(self, x, *args, **kwargs):
        if self.factor == 1:
            return x

        if self.data_format == NCHW_FORMAT:
            _, c, h, w = x.shape
            shape1 = [-1, c, h, 1, w, 1]
            tile_shape = [1, 1, 1, self.factor, 1, self.factor]
            output_shape = [-1, c, h * self.factor, w * self.factor]
        elif self.data_format == NHWC_FORMAT:
            _, h, w, c = x.shape
            shape1 = [-1, h, 1, w, 1, c]
            tile_shape = [1, 1, self.factor, 1, self.factor, 1]
            output_shape = [-1, h * self.factor, w * self.factor, c]

        x = tf.reshape(x, shape1)
        x = tf.tile(x, tile_shape)
        x = tf.reshape(x, output_shape)
        return x


class Downscale2d(Layer):

    def __init__(self, factor, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, name=None):
        super(Downscale2d, self).__init__(dtype=dtype, name=name)
        validate_data_format(data_format)
        self.data_format = data_format
        if self.data_format == NCHW_FORMAT:
            self.ksize = [1, 1, factor, factor]
        elif self.data_format == NHWC_FORMAT:
            self.ksize = [1, factor, factor, 1]
        self.factor = factor
        self.use_xla = use_xla

    @tf.function
    def call(self, x, *args, **kwargs):
        if self.factor == 1:
            return x
        # Note: soft ops placements should be enabled
        return tf.nn.avg_pool2d(
            x, self.ksize, strides=self.ksize, padding='VALID', data_format=self.data_format
        )


class Pixel_Norm(Layer):

    def __init__(self, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, name=None):
        super(Pixel_Norm, self).__init__(dtype=dtype, name=name)
        validate_data_format(data_format)
        self.data_format = data_format
        if self.data_format == NCHW_FORMAT:
            self.channel_axis = 1
        elif self.data_format == NHWC_FORMAT:
            self.channel_axis = 3
        self.epsilon = 1e-8 if self._dtype_policy.compute_dtype == 'float32' else 1e-4
        self.use_xla = use_xla

    @tf.function
    def call(self, x, *args, **kwargs):
        return x * tf.math.rsqrt(
            tf.reduce_mean(tf.square(x), axis=self.channel_axis, keepdims=True) + self.epsilon
        )


class Minibatch_StdDev(Layer):

    def __init__(self, group_size, dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, name=None):
        super(Minibatch_StdDev, self).__init__(dtype=dtype, name=name)
        validate_data_format(data_format)
        self.data_format = data_format
        self.group_size = group_size
        self.use_xla = use_xla

    @tf.function
    def call(self, x, *args, **kwargs):
        if self.data_format == NCHW_FORMAT:
            _, c, h, w = x.shape
            n = tf.shape(x)[0]
            group_size = tf.math.minimum(self.group_size, n)      # Minibatch must be divisible or smaller than batch size
            y = tf.reshape(x, [group_size, -1, c, h, w])          # [GMCHW]
            y = tf.cast(y, tf.float32)                            # [GMCHW] Cast to fp32
            y -= tf.math.reduce_mean(y, axis=0, keepdims=True)    # [GMCHW]
            y = tf.reduce_mean(tf.square(y), axis=0)              # [MCHW] Variance over group
            y = tf.sqrt(y + 1e-8)                                 # [MCHW] Stddev over group
            y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)  # [M111] Average over fmaps `band pixels
            y = tf.cast(y, x.dtype)                               # [M111] Cast back to original dtype
            y = tf.tile(y, [group_size, 1, h, w])                 # [N1HW] Replicate over group and pixels
            return tf.concat([x, y], axis=1)
        elif self.data_format == NHWC_FORMAT:
            _, h, w, c = x.shape
            n = tf.shape(x)[0]
            group_size = tf.math.minimum(self.group_size, n)      # Minibatch must be divisible or smaller than batch size
            y = tf.reshape(x, [group_size, -1, h, w, c])          # [GMHWC]
            y = tf.cast(y, tf.float32)                            # [GMHWC] Cast to fp32
            y -= tf.math.reduce_mean(y, axis=0, keepdims=True)    # [GMHWC]
            y = tf.reduce_mean(tf.square(y), axis=0)              # [MHWC] Variance over group
            y = tf.sqrt(y + 1e-8)                                 # [MHWC] Stddev over group
            y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)  # [M111] Average over fmaps `band pixels
            y = tf.cast(y, x.dtype)                               # [M111] Cast back to original dtype
            y = tf.tile(y, [group_size, h, w, 1])                 # [NHW1] Replicate over group and pixels
            return tf.concat([x, y], axis=3)


class Weighted_sum(Layer):
    def __init__(self, dtype=DEFAULT_DTYPE, name=None):
        super(Weighted_sum, self).__init__(dtype=dtype, name=name)
        # Note: for mixed precision training constants can have float16 dtype
        self.alpha =self.add_weight(
            name='alpha',
            initializer=tf.constant_initializer(0.),
            trainable=False,
            dtype=self._dtype_policy.compute_dtype,
            experimental_autocast=False
        )
        self.one = tf.constant(1., dtype=self._dtype_policy.compute_dtype, name='One')

    # Avoid using tf.function or alpha will be compiled (if it no set as non trainable weight)
    def call(self, inputs, *args, **kwargs):
        return (self.one - self.alpha) * inputs[0] + self.alpha * inputs[1]


# Fused upscale2d + conv2d.
# Faster and uses less memory than performing the operations separately.
class Fused_Upscale2d_Scaled_Conv2d(Layer):

    def __init__(self, fmaps, kernel_size, gain=HE_GAIN,
                 use_wscale=True, truncate_weights=DEFAULT_TRUNCATE_WEIGHTS,
                 dtype=DEFAULT_DTYPE, use_xla=DEFAULT_USE_XLA,
                 data_format=DEFAULT_DATA_FORMAT, name=None):
        super(Fused_Upscale2d_Scaled_Conv2d, self).__init__(dtype=dtype, name=name)
        assert kernel_size >= 1 and kernel_size % 2 == 1
        validate_data_format(data_format)
        self.data_format = data_format
        self.fmaps = fmaps
        self.kernel_size = kernel_size
        self.gain = gain
        self.use_wscale = use_wscale
        self.truncate_weights = truncate_weights
        self.use_xla = use_xla

    def build(self, input_shape):
        if self.data_format == NCHW_FORMAT:
            self.channels_in = input_shape[1]
            self.strides = [1, 1, 2, 2]
            self.os_tail = [self.fmaps, input_shape[2] * 2, input_shape[3] * 2]
        elif self.data_format == NHWC_FORMAT:
            self.channels_in = input_shape[-1]
            self.strides = [1, 2, 2, 1]
            self.os_tail = [input_shape[1] * 2, input_shape[2] * 2, self.fmaps]

        # Wshape is different from Scaled_Conv2d
        self.wshape = [self.kernel_size, self.kernel_size, self.fmaps, self.channels_in]
        self.fan_in = (self.kernel_size ** 2) * self.channels_in
        # He/LeCun init
        self.std = self.gain / np.sqrt(self.fan_in)
        if self.use_wscale:
            self.wscale = self.std
            initializer = select_initializer(self.truncate_weights, 1.)
        else:
            initializer = select_initializer(self.truncate_weights, self.std)

        self.w = self.add_weight(
            name=WEIGHTS_NAME,
            shape=self.wshape,
            initializer=initializer,
            trainable=True
        )

    @tf.function
    def call(self, x, *args, **kwargs):
        if self.use_wscale:
            w = self.wscale * self.w
        else:
            w = self.w

        w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])

        os = [tf.shape(x)[0]] + self.os_tail

        return tf.nn.conv2d_transpose(
            x, w, os, strides=self.strides, padding='SAME', data_format=self.data_format
        )


# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.
class Fused_Scaled_Conv2d_Downscale2d(Layer):

    def __init__(self, fmaps, kernel_size, gain=HE_GAIN,
                 use_wscale=True, truncate_weights=DEFAULT_TRUNCATE_WEIGHTS,
                 use_xla=DEFAULT_USE_XLA, dtype=DEFAULT_DTYPE,
                 data_format=DEFAULT_DATA_FORMAT, name=None):
        super(Fused_Scaled_Conv2d_Downscale2d, self).__init__(dtype=dtype, name=name)
        assert kernel_size >= 1 and kernel_size % 2 == 1
        validate_data_format(data_format)
        self.data_format = data_format
        self.fmaps = fmaps
        self.kernel_size = kernel_size
        self.gain = gain
        self.use_wscale = use_wscale
        self.truncate_weights = truncate_weights
        self.use_xla = use_xla
        #self.call = tf.function(self.call, experimental_compile=self.use_xla)

    def build(self, input_shape):
        if self.data_format == NCHW_FORMAT:
            self.channels_in = input_shape[1]
            self.strides = [1, 1, 2, 2]
        elif self.data_format == NHWC_FORMAT:
            self.channels_in = input_shape[-1]
            self.strides = [1, 2, 2, 1]

        self.wshape = [self.kernel_size, self.kernel_size, self.channels_in, self.fmaps]
        self.fan_in = np.prod(self.wshape[:-1])
        # He/LeCun init
        self.std = self.gain / np.sqrt(self.fan_in)
        if self.use_wscale:
            self.wscale = self.std
            initializer = select_initializer(self.truncate_weights, 1.)
        else:
            initializer = select_initializer(self.truncate_weights, self.std)

        self.w = self.add_weight(
            name=WEIGHTS_NAME,
            shape=self.wshape,
            initializer=initializer,
            trainable=True
        )

    @tf.function
    def call(self, x, *args, **kwargs):
        if self.use_wscale:
            w = self.wscale * self.w
        else:
            w = self.w

        w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25

        return tf.nn.conv2d(
            x, w, strides=self.strides, padding='SAME', data_format=self.data_format
        )
