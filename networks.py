import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Input, Reshape
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from special_layers import Scaled_Conv2d, Scaled_Linear, Bias,\
    Downscale2d, Upscale2d,\
    Fused_Upscale2d_Scaled_Conv2d, Fused_Scaled_Conv2d_Downscale2d,\
    Minibatch_StdDev, Pixel_Norm, Weighted_sum
from utils import FADE_IN_MODE, STABILIZATION_MODE, WSUM_NAME,\
    GAIN_INIT_MODE_DICT, GAIN_ACTIVATION_FUNS_DICT, ACTIVATION_FUNS_DICT, FP32_ACTIVATIONS,\
    level_of_details, validate_data_format, extract_res_str, is_resolution_block_name

from utils import TARGET_RESOLUTION, START_RESOLUTION, LATENT_SIZE, NORMALIZE_LATENTS,\
    DATA_FORMAT, USE_MIXED_PRECISION, USE_BIAS, USE_WSCALE,\
    USE_PIXELNORM, TRUNCATE_WEIGHTS,\
    G_FUSED_SCALE, G_WEIGHTS_INIT_MODE, G_ACTIVATION, G_KERNEL_SIZE,\
    D_FUSED_SCALE, D_WEIGHTS_INIT_MODE, D_ACTIVATION, D_KERNEL_SIZE,\
    MBSTD_GROUP_SIZE, OVERRIDE_G_PROJECTING_GAIN, D_PROJECTING_NF,\
    G_FMAP_BASE, G_FMAP_DECAY, G_FMAP_MAX,\
    D_FMAP_BASE, D_FMAP_DECAY, D_FMAP_MAX,\
    LOD_NAME, RGB_NAME, BATCH_SIZES
from utils import NCHW_FORMAT, NHWC_FORMAT, DEFAULT_DATA_FORMAT,\
    DEFAULT_USE_MIXED_PRECISION, DEFAULT_START_RESOLUTION,\
    DEFAULT_OVERRIDE_G_PROJECTING_GAIN, DEFAULT_NORMALIZE_LATENTS,\
    DEFAULT_G_ACTIVATION, DEFAULT_D_ACTIVATION,\
    DEFAULT_G_FUSED_SCALE, DEFAULT_D_FUSED_SCALE,\
    DEFAULT_G_KERNEL_SIZE, DEFAULT_D_KERNEL_SIZE, DEFAULT_USE_BIAS,\
    DEFAULT_USE_PIXELNORM, DEFAULT_TRUNCATE_WEIGHTS, DEFAULT_USE_WSCALE,\
    DEFAULT_FMAP_BASE, DEFAULT_FMAP_DECAY, DEFAULT_FMAP_MAX


def n_filters(stage, fmap_base, fmap_decay, fmap_max):
    """
    fmap_base  Overall multiplier for the number of feature maps.
    fmap_decay log2 feature map reduction when doubling the resolution.
    fmap_max   Maximum number of feature maps in any layer.
    """
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)


class Generator():

    def __init__(self, config):

        self.target_resolution = config[TARGET_RESOLUTION]
        self.resolution_log2 = int(np.log2(self.target_resolution))
        assert self.target_resolution == 2 ** self.resolution_log2 and self.target_resolution >= 4

        self.start_resolution = config.get(START_RESOLUTION, DEFAULT_START_RESOLUTION)
        self.start_resolution_log2 = int(np.log2(self.start_resolution))
        assert self.start_resolution == 2 ** self.start_resolution_log2 and self.start_resolution >= 4

        self.data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
        validate_data_format(self.data_format)

        self.latent_size = config[LATENT_SIZE]
        self.normalize_latents = config.get(NORMALIZE_LATENTS, DEFAULT_NORMALIZE_LATENTS)
        self.use_bias = config.get(USE_BIAS, DEFAULT_USE_BIAS)
        self.use_wscale = config.get(USE_WSCALE, DEFAULT_USE_WSCALE)
        self.use_pixelnorm = config.get(USE_PIXELNORM, DEFAULT_USE_PIXELNORM)
        self.override_projecting_gain = config.get(
            OVERRIDE_G_PROJECTING_GAIN, DEFAULT_OVERRIDE_G_PROJECTING_GAIN
        )
        self.projecting_gain_correction = 4. if self.override_projecting_gain else 1.
        self.truncate_weights = config.get(TRUNCATE_WEIGHTS, DEFAULT_TRUNCATE_WEIGHTS)
        self.G_fused_scale = config.get(G_FUSED_SCALE, DEFAULT_G_FUSED_SCALE)
        self.G_kernel_size = config.get(G_KERNEL_SIZE, DEFAULT_G_KERNEL_SIZE)
        self.G_fmap_base = config.get(G_FMAP_BASE, DEFAULT_FMAP_BASE)
        self.G_fmap_decay = config.get(G_FMAP_DECAY, DEFAULT_FMAP_DECAY)
        self.G_fmap_max = config.get(G_FMAP_MAX, DEFAULT_FMAP_MAX)
        self.batch_sizes = config[BATCH_SIZES]

        self.G_act_name = config.get(G_ACTIVATION, DEFAULT_G_ACTIVATION)
        if self.G_act_name in ACTIVATION_FUNS_DICT.keys():
            self.G_act = ACTIVATION_FUNS_DICT[self.G_act_name]
        else:
            assert False, f"Generator activation '{self.G_act_name}' not supported. See ACTIVATION_FUNS_DICT"

        self.use_mixed_precision = config.get(USE_MIXED_PRECISION, DEFAULT_USE_MIXED_PRECISION)
        if self.use_mixed_precision:
            self.policy = mixed_precision.Policy('mixed_float16')
            self.act_dtype = 'float32' if self.G_act_name in FP32_ACTIVATIONS else self.policy
            self.compute_dtype = self.policy.compute_dtype
        else:
            self.policy = 'float32'
            self.act_dtype = 'float32'
            self.compute_dtype = 'float32'

        self.weights_init_mode = config.get(G_WEIGHTS_INIT_MODE, None)
        if self.weights_init_mode is None:
            self.gain = GAIN_ACTIVATION_FUNS_DICT[self.G_act_name]
        else:
            self.gain = GAIN_INIT_MODE_DICT[self.weights_init_mode]

        # This constant is taken from the original implementation
        self.projecting_mult = 4
        if self.data_format == NCHW_FORMAT:
            self.z_dim = (self.latent_size, 1, 1)
            self.projecting_target_shape = (
                self.G_n_filters(2 - 1), self.projecting_mult, self.projecting_mult
            )
        elif self.data_format == NHWC_FORMAT:
            self.z_dim = (1, 1, self.latent_size)
            self.projecting_target_shape = (
                self.projecting_mult, self.projecting_mult, self.G_n_filters(2 - 1)
            )

        self.min_lod = level_of_details(self.resolution_log2, self.resolution_log2)
        self.max_lod = level_of_details(self.start_resolution_log2, self.resolution_log2)

        self.create_model_layers()

    def G_n_filters(self, stage):
        return n_filters(stage, self.G_fmap_base, self.G_fmap_decay, self.G_fmap_max)

    def create_model_layers(self):
        self.up_layers = {
            res: Upscale2d(
                factor=2, dtype=self.policy, data_format=self.data_format,
                name='Upscale2D_%dx%d' % (2 ** res, 2 ** res)
            )
            for res in range(self.start_resolution_log2, self.resolution_log2 + 1)
        }
        self.G_wsum_layers = {
            lod: Weighted_sum(dtype=self.policy, name=f'G_{WSUM_NAME}_{LOD_NAME}{lod}')
            for lod in range(self.min_lod, self.max_lod + 1)
        }
        self.G_input_layer = Input(shape=self.z_dim, name='Latent_vector', dtype=self.compute_dtype)
        self.G_latents_normalizer = Pixel_Norm(
            dtype=self.policy, data_format=self.data_format, name='Latents_normalizer'
        )
        self.G_blocks = {
            res: self.G_block(res) for res in range(self.start_resolution_log2, self.resolution_log2 + 1)
        }
        self.toRGB_layers = {
            level_of_details(res, self.resolution_log2): self.to_rgb(res)
            for res in range(self.start_resolution_log2, self.resolution_log2 + 1)
        }

    def to_rgb(self, res):
        lod = level_of_details(res, self.resolution_log2)
        block_name = f'To{RGB_NAME}_{LOD_NAME}{lod}'
        conv_layers = [
            self.conv2d(fmaps=3, kernel_size=1, gain=1.)
        ]
        if self.use_bias: conv_layers = self.apply_bias(conv_layers)
        return tf.keras.Sequential(conv_layers, name=block_name)

    def dense(self, units, gain=None):
        if gain is None: gain = self.gain
        return Scaled_Linear(
            units=units, gain=gain,
            use_wscale=self.use_wscale, truncate_weights=self.truncate_weights,
            dtype=self.policy, data_format=self.data_format
        )

    def conv2d(self, fmaps, kernel_size=None, gain=None, fused_up=False):
        if kernel_size is None: kernel_size = self.G_kernel_size
        if gain is None: gain = self.gain
        if not fused_up:
            return Scaled_Conv2d(
                fmaps=fmaps, kernel_size=kernel_size, gain=gain,
                use_wscale=self.use_wscale, truncate_weights=self.truncate_weights,
                dtype=self.policy, data_format=self.data_format
            )
        else:
            return Fused_Upscale2d_Scaled_Conv2d(
                fmaps=fmaps, kernel_size=kernel_size, gain=gain,
                use_wscale=self.use_wscale, truncate_weights=self.truncate_weights,
                dtype=self.policy, data_format=self.data_format
            )

    def act(self):
        return Activation(self.G_act, dtype=self.act_dtype)

    def apply_bias(self, layers):
        bias_layer = Bias(dtype=self.policy, data_format=self.data_format)
        if len(layers) > 1:
            return layers[:-1] + [bias_layer, layers[-1]]
        else:
            return [layers[-1], bias_layer]

    def PN(self, layers):
        return layers + [Pixel_Norm(dtype=self.policy, data_format=self.data_format)]

    def G_block(self, res):
        block_name = '%dx%d' % (2 ** res, 2 ** res)
        # res = 2 ... resolution_log2
        if res == 2:  # 4x4
            # Linear block
            # Gain is overridden to match the original implementation
            # sqrt(2) / 4 was used with He init
            projecting_layer = self.dense(
                units=np.prod(self.projecting_target_shape),
                gain=self.gain / self.projecting_gain_correction
            )
            linear_layers = [
                projecting_layer,
                Reshape(target_shape=self.projecting_target_shape, dtype=self.policy),
                self.act()
            ]
            if self.use_bias: linear_layers = self.apply_bias(linear_layers)
            if self.use_pixelnorm: linear_layers = self.PN(linear_layers)
            linear_block = tf.keras.Sequential(linear_layers, name='Projecting')

            # Conv block
            conv_layers = [
                self.conv2d(fmaps=self.G_n_filters(res - 1)),
                self.act()
            ]
            if self.use_bias: conv_layers = self.apply_bias(conv_layers)
            if self.use_pixelnorm: conv_layers = self.PN(conv_layers)
            conv_block = tf.keras.Sequential(conv_layers, name='Conv')

            # Full block
            block_model = tf.keras.Sequential([
                linear_block,
                conv_block
            ], name=block_name)
        else:  # 8x8 and up
            # 1st conv block
            if self.G_fused_scale:
                conv0_layers = [
                    self.conv2d(fmaps=self.G_n_filters(res - 1), fused_up=True),
                    self.act()
                ]
                if self.use_bias: conv0_layers = self.apply_bias(conv0_layers)
                if self.use_pixelnorm: conv0_layers = self.PN(conv0_layers)
                conv0_block = tf.keras.Sequential(conv0_layers, name='Conv0_up')

                block_layers = [conv0_block]
            else:
                conv0_layers = [
                    self.conv2d(fmaps=self.G_n_filters(res - 1)),
                    self.act()
                ]
                if self.use_bias: conv0_layers = self.apply_bias(conv0_layers)
                if self.use_pixelnorm: conv0_layers = self.PN(conv0_layers)
                conv0_block = tf.keras.Sequential(conv0_layers, name='Conv0')

                block_layers = [self.up_layers[res], conv0_block]

            # 2nd conv block
            conv1_layers = [
                self.conv2d(fmaps=self.G_n_filters(res - 1)),
                self.act()
            ]
            if self.use_bias: conv1_layers = self.apply_bias(conv1_layers)
            if self.use_pixelnorm: conv1_layers = self.PN(conv1_layers)
            conv1_block = tf.keras.Sequential(conv1_layers, name='Conv1')

            # Full block
            block_layers += [conv1_block]
            block_model = tf.keras.Sequential(block_layers, name=block_name)

        return block_model

    def create_G_model(self, res, mode=STABILIZATION_MODE):
        assert mode in [FADE_IN_MODE, STABILIZATION_MODE], 'Mode ' + mode + ' is not supported'

        lod = level_of_details(res, self.resolution_log2)

        details_layers = [self.G_blocks[i] for i in range(2, res + 1)]
        if self.normalize_latents:
            details_layers = [self.G_latents_normalizer] + details_layers

        if res == 2:  # 4x4
            toRGB_layer = self.toRGB_layers[lod]
            images_layers = details_layers + [toRGB_layer]

            images_out = self.G_input_layer
            for layer in images_layers:
                images_out = layer(images_out)
        else:  # 8x8 and up
            if mode == FADE_IN_MODE:
                up_layer_name = 'Upscale2D_%dx%d_%s' % (2 ** res, 2 ** res, FADE_IN_MODE)
                up_layer = Upscale2d(
                    factor=2, dtype=self.policy, data_format=self.data_format, name=up_layer_name
                )
                toRGB_layer1 = self.toRGB_layers[lod + 1]
                toRGB_layer2 = self.toRGB_layers[lod]

                images = self.G_input_layer
                for layer in details_layers[:-1]:
                    images = layer(images)

                images1_layers = [toRGB_layer1, up_layer]
                images1 = images
                for layer in images1_layers:
                    images1 = layer(images1)

                images2_layers = [details_layers[-1], toRGB_layer2]
                images2 = images
                for layer in images2_layers:
                    images2 = layer(images2)

                images_out = self.G_wsum_layers[lod]([images1, images2])

            elif mode == STABILIZATION_MODE:
                toRGB_layer = self.toRGB_layers[lod]
                images_layers = details_layers + [toRGB_layer]
                images_out = self.G_input_layer
                for layer in images_layers:
                    images_out = layer(images_out)

        G_model = tf.keras.Model(
            inputs=self.G_input_layer, outputs=tf.identity(images_out, name='Images_out'),
            name=f'G_model_{LOD_NAME}{lod}'
        )
        return G_model

    def initialize_G_model(self, summary_model=False, model_res=None, mode=None):
        if model_res is not None:
            batch_size = self.batch_sizes[str(model_res)]
            latents = tf.random.normal(
                shape=(batch_size,) + self.z_dim, stddev=0.05, dtype=self.compute_dtype
            )
            G_model = self.create_G_model(model_res, mode=mode)

            _ = G_model(latents)
        else:
            for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
                batch_size = self.batch_sizes[str(res)]
                latents = tf.random.normal(
                    shape=(batch_size,) + self.z_dim, stddev=0.05, dtype=self.compute_dtype
                )
                G_model = self.create_G_model(res, mode=FADE_IN_MODE)

                _ = G_model(latents)
                if res == self.resolution_log2 and summary_model:
                    logging.info('\nThe biggest Generator:\n')
                    G_model.summary(print_fn=logging.info)

        if not summary_model:
            print('G model built')

    def update_G_weights(self, G_model):
        logging.info('\nUpdating G weights...')
        for layer in G_model.layers:
            layer_name = layer.name
            if RGB_NAME in layer_name:
                lod = int(layer_name.split(LOD_NAME)[1])
                w = layer.get_weights()
                self.toRGB_layers[lod].set_weights(w)
                logging.info(f'src layer name: {layer_name}')
                logging.info(f'target layer name: {self.toRGB_layers[lod].name}')
            elif is_resolution_block_name(layer_name, self.resolution_log2):
                res = int(np.log2(int(extract_res_str(layer_name))))
                w = layer.get_weights()
                self.G_blocks[res].set_weights(w)
                logging.info(f'src layer name: {layer_name}')
                logging.info(f'target layer name: {self.G_blocks[res].name}')

    def trace_G_graphs(self, summary_writers, writers_dirs):
        G_prefix = 'Generator_'
        trace_G_input = tf.random.normal(
            shape=(1,) + self.z_dim, dtype=self.compute_dtype
        )
        for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
            writer = summary_writers[res]
            profiler_dir = writers_dirs[res]
            if res == self.start_resolution_log2:
                trace_G_model = tf.function(
                    self.create_G_model(res, mode=STABILIZATION_MODE)
                )
                with writer.as_default():
                    tf.summary.trace_on(graph=True, profiler=True)
                    trace_G_model(trace_G_input)

                    tf.summary.trace_export(
                        name=G_prefix + STABILIZATION_MODE,
                        step=0,
                        profiler_outdir=profiler_dir
                    )
                    tf.summary.trace_off()

                    writer.flush()
            else:
                trace_G_model1 = tf.function(
                    self.create_G_model(res, mode=FADE_IN_MODE)
                )
                trace_G_model2 = tf.function(
                    self.create_G_model(res, mode=STABILIZATION_MODE)
                )
                with writer.as_default():
                    # Fade-in model
                    tf.summary.trace_on(graph=True, profiler=True)
                    trace_G_model1(trace_G_input)

                    tf.summary.trace_export(
                        name=G_prefix + FADE_IN_MODE,
                        step=0,
                        profiler_outdir=profiler_dir
                    )
                    tf.summary.trace_off()

                    # Stabilization model
                    tf.summary.trace_on(graph=True, profiler=True)
                    trace_G_model2(trace_G_input)

                    tf.summary.trace_export(
                        name=G_prefix + STABILIZATION_MODE,
                        step=0,
                        profiler_outdir=profiler_dir
                    )
                    tf.summary.trace_off()

                    writer.flush()

        print('All Generator models traced!')


class Discriminator():

    def __init__(self, config):

        self.target_resolution = config[TARGET_RESOLUTION]
        self.resolution_log2 = int(np.log2(self.target_resolution))
        assert self.target_resolution == 2 ** self.resolution_log2 and self.target_resolution >= 4

        self.start_resolution = config.get(START_RESOLUTION, DEFAULT_START_RESOLUTION)
        self.start_resolution_log2 = int(np.log2(self.start_resolution))
        assert self.start_resolution == 2 ** self.start_resolution_log2 and self.start_resolution >= 4

        self.data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
        validate_data_format(self.data_format)

        self.use_bias = config.get(USE_BIAS, DEFAULT_USE_BIAS)
        self.use_wscale = config.get(USE_WSCALE, DEFAULT_USE_WSCALE)
        self.truncate_weights = config.get(TRUNCATE_WEIGHTS, DEFAULT_TRUNCATE_WEIGHTS)
        self.mbstd_group_size = config[MBSTD_GROUP_SIZE]
        self.D_fused_scale = config.get(D_FUSED_SCALE, DEFAULT_D_FUSED_SCALE)
        self.D_kernel_size = config.get(D_KERNEL_SIZE, DEFAULT_D_KERNEL_SIZE)
        self.D_fmap_base = config.get(D_FMAP_BASE, DEFAULT_FMAP_BASE)
        self.D_fmap_decay = config.get(D_FMAP_DECAY, DEFAULT_FMAP_DECAY)
        self.D_fmap_max = config.get(D_FMAP_MAX, DEFAULT_FMAP_MAX)
        self.batch_sizes = config[BATCH_SIZES]

        self.D_act_name = config.get(D_ACTIVATION, DEFAULT_D_ACTIVATION)
        if self.D_act_name in ACTIVATION_FUNS_DICT.keys():
            self.D_act = ACTIVATION_FUNS_DICT[self.D_act_name]
        else:
            assert False, f"Discriminator activation '{self.D_act_name}' not supported. See ACTIVATION_FUNS_DICT"

        self.use_mixed_precision = config.get(USE_MIXED_PRECISION, DEFAULT_USE_MIXED_PRECISION)
        if self.use_mixed_precision:
            self.policy = mixed_precision.Policy('mixed_float16')
            self.act_dtype = 'float32' if self.D_act_name in FP32_ACTIVATIONS else self.policy
            self.compute_dtype = self.policy.compute_dtype
        else:
            self.policy = 'float32'
            self.act_dtype = 'float32'
            self.compute_dtype = 'float32'

        self.weights_init_mode = config.get(D_WEIGHTS_INIT_MODE, None)
        if self.weights_init_mode is None:
            self.gain = GAIN_ACTIVATION_FUNS_DICT[self.D_act_name]
        else:
            self.gain = GAIN_INIT_MODE_DICT[self.weights_init_mode]

        # Might be useful to override number of units in projecting layer
        # in case latent size is not 512 to make models have almost the same number
        # of trainable params
        self.projecting_nf = config.get(D_PROJECTING_NF, self.D_n_filters(2 - 2))

        self.min_lod = level_of_details(self.resolution_log2, self.resolution_log2)
        self.max_lod = level_of_details(self.start_resolution_log2, self.resolution_log2)

        self.create_model_layers()

    def D_n_filters(self, stage):
        return n_filters(stage, self.D_fmap_base, self.D_fmap_decay, self.D_fmap_max)

    def D_input_shape(self, res):
        if self.data_format == NCHW_FORMAT:
            return (3, 2 ** res, 2 ** res)
        elif self.data_format == NHWC_FORMAT:
            return (2 ** res, 2 ** res, 3)

    def create_model_layers(self):
        self.down_layers = {
            res: Downscale2d(
                factor=2, dtype=self.policy, data_format=self.data_format,
                name='Downscale2D_%dx%d' % (2 ** res, 2 ** res)
            ) for res in range(self.start_resolution_log2, self.resolution_log2 + 1)
        }
        self.D_wsum_layers = {
            lod: Weighted_sum(dtype=self.policy, name=f'D_{WSUM_NAME}_{LOD_NAME}{lod}')
            for lod in range(self.min_lod, self.max_lod + 1)
        }
        self.D_input_layers = {
            res: Input(
                shape=self.D_input_shape(res), dtype=self.compute_dtype,
                name='Images_%dx%d' % (2 ** res, 2 ** res)
            ) for res in range(self.start_resolution_log2, self.resolution_log2 + 1)
        }
        self.D_blocks = {
            res: self.D_block(res) for res in range(self.start_resolution_log2, self.resolution_log2 + 1)
        }
        self.fromRGB_layers = {
            level_of_details(res, self.resolution_log2): self.from_rgb(res)
            for res in range(self.start_resolution_log2, self.resolution_log2 + 1)
        }

    def from_rgb(self, res):
        lod = level_of_details(res, self.resolution_log2)
        block_name = f'From{RGB_NAME}_{LOD_NAME}{lod}'
        conv_layers = [
            self.conv2d(fmaps=self.D_n_filters(res - 1), kernel_size=1),
            self.act()
        ]
        if self.use_bias: conv_layers = self.apply_bias(conv_layers)
        return tf.keras.Sequential(conv_layers, name=block_name)

    def dense(self, units, gain=None):
        if gain is None: gain=self.gain
        return Scaled_Linear(
            units=units, gain=gain,
            use_wscale=self.use_wscale, truncate_weights=self.truncate_weights,
            dtype=self.policy, data_format=self.data_format
        )

    def conv2d(self, fmaps, kernel_size=None, gain=None, fused_down=False):
        if kernel_size is None: kernel_size = self.D_kernel_size
        if gain is None: gain = self.gain
        if not fused_down:
            return Scaled_Conv2d(
                fmaps=fmaps, kernel_size=kernel_size, gain=gain,
                use_wscale=self.use_wscale, truncate_weights=self.truncate_weights,
                dtype=self.policy, data_format=self.data_format
        )
        else:
            return Fused_Scaled_Conv2d_Downscale2d(
                fmaps=fmaps, kernel_size=kernel_size, gain=gain,
                use_wscale=self.use_wscale, truncate_weights=self.truncate_weights,
                dtype=self.policy, data_format=self.data_format
            )

    def act(self):
        return Activation(self.D_act, dtype=self.act_dtype)

    def apply_bias(self, layers):
        bias_layer = Bias(dtype=self.policy, data_format=self.data_format)
        if len(layers) > 1:
            return layers[:-1] + [bias_layer, layers[-1]]
        else:
            return [layers[-1], bias_layer]

    def D_block(self, res):
        # res = 2 ... resolution_log2
        block_name = '%dx%d' % (2 ** res, 2 ** res)
        if res >= 3:  # 8x8 and up
            conv0_layers = [
                self.conv2d(fmaps=self.D_n_filters(res - 1)),
                self.act()
            ]
            if self.use_bias: conv0_layers = self.apply_bias(conv0_layers)
            conv0_block = tf.keras.Sequential(conv0_layers, name='Conv0')

            if self.D_fused_scale:
                conv1_layers = [
                    self.conv2d(fmaps=self.D_n_filters(res - 2), fused_down=True),
                    self.act()
                ]
                if self.use_bias: conv1_layers = self.apply_bias(conv1_layers)
                conv1_block = tf.keras.Sequential(conv1_layers, name='Conv1_down')

                block_layers = [conv0_block, conv1_block]
            else:
                conv1_layers = [
                    self.conv2d(fmaps=self.D_n_filters(res - 2)),
                    self.act()
                ]
                if self.use_bias: conv1_layers = self.apply_bias(conv1_layers)
                conv1_block = tf.keras.Sequential(conv1_layers, name='Conv1')

                down_layer = self.down_layers[res]
                block_layers = [conv0_block, conv1_block, down_layer]

            block_model = tf.keras.Sequential(block_layers, name=block_name)
        else:  # 4x4
            conv_layers = [
                self.conv2d(fmaps=self.D_n_filters(res - 1)),
                self.act()
            ]
            if self.use_bias: conv_layers = self.apply_bias(conv_layers)
            conv_block = tf.keras.Sequential(conv_layers, name='Conv')

            dense0_layers = [
                self.dense(units=self.projecting_nf),
                self.act()
            ]
            if self.use_bias: dense0_layers = self.apply_bias(dense0_layers)
            dense0_block = tf.keras.Sequential(dense0_layers, name='Projecting')

            dense1_layers = [
                self.dense(units=1, gain=1.)
            ]
            if self.use_bias: dense1_layers = self.apply_bias(dense1_layers)
            # Will make graph look better
            dense1_block = tf.keras.Sequential(dense1_layers, name='Dense1')

            block_layers = [conv_block, dense0_block, dense1_block]
            if self.mbstd_group_size > 1:
                print("Using mbstd layer...")
                mbstd_layer = Minibatch_StdDev(
                    group_size=self.mbstd_group_size, dtype=self.policy,
                    data_format=self.data_format, name='Minibatch_stddev'
                )
                block_layers = [mbstd_layer] + block_layers

            block_model = tf.keras.Sequential(block_layers, name=block_name)

        return block_model

    def create_D_model(self, res, mode=STABILIZATION_MODE):
        assert mode in [FADE_IN_MODE, STABILIZATION_MODE], 'Mode ' + mode + ' is not supported'

        lod = level_of_details(res, self.resolution_log2)

        details_layers = [self.D_blocks[i] for i in range(res, 2 - 1, -1)]
        D_input_layer = self.D_input_layers[res]
        if res == 2:  # 4x4
            fromRGB_layer = self.fromRGB_layers[lod]
            model_layers = [fromRGB_layer] + details_layers

            x = D_input_layer
            for layer in model_layers:
                x = layer(x)
        else:  # 8x8 and up
            if mode == FADE_IN_MODE:
                fromRGB_layer1 = self.fromRGB_layers[lod + 1]
                down_layer_name = 'Downscale2D_%dx%d_%s' % (2 ** res, 2 ** res, FADE_IN_MODE)
                down_layer = Downscale2d(
                    factor=2, dtype=self.policy, data_format=self.data_format, name=down_layer_name
                )
                branch1_layers = [down_layer, fromRGB_layer1]
                x1 = D_input_layer
                for layer in branch1_layers:
                    x1 = layer(x1)

                fromRGB_layer2 = self.fromRGB_layers[lod]
                branch2_layers = [fromRGB_layer2, details_layers[0]]
                x2 = D_input_layer
                for layer in branch2_layers:
                    x2 = layer(x2)

                x = self.D_wsum_layers[lod]([x1, x2])
                for layer in details_layers[1:]:
                    x = layer(x)

            elif mode == STABILIZATION_MODE:
                fromRGB_layer = self.fromRGB_layers[lod]
                model_layers = [fromRGB_layer] + details_layers

                x = D_input_layer
                for layer in model_layers:
                    x = layer(x)

        D_model = tf.keras.Model(
            inputs=D_input_layer, outputs=tf.identity(x, name='Scores'), name=f'D_model_{LOD_NAME}{lod}'
        )
        return D_model

    def initialize_D_model(self, summary_model=False, model_res=None, mode=None):
        if model_res is not None:
            batch_size = self.batch_sizes[str(model_res)]
            images = tf.random.normal(
                shape=(batch_size,) + self.D_input_shape(model_res), stddev=0.05, dtype=self.compute_dtype
            )

            D_model = self.create_D_model(model_res, mode=mode)
            _ = D_model(images)
        else:
            for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
                batch_size = self.batch_sizes[str(res)]
                images = tf.random.normal(
                    shape=(batch_size,) + self.D_input_shape(res), stddev=0.05, dtype=self.compute_dtype
                )

                D_model = self.create_D_model(res, mode=FADE_IN_MODE)
                _ = D_model(images)

                if res == self.resolution_log2 and summary_model:
                    logging.info('\nThe biggest Discriminator:\n')
                    D_model.summary(print_fn=logging.info)

        if not summary_model:
            print('D model built')

    def update_D_weights(self, D_model):
        logging.info('\nUpdating D weights...')
        for layer in D_model.layers:
            layer_name = layer.name
            if RGB_NAME in layer_name:
                lod = int(layer_name.split(LOD_NAME)[1])
                w = layer.get_weights()
                self.fromRGB_layers[lod].set_weights(w)
                logging.info(f'src layer name: {layer_name}')
                logging.info(f'target layer name: {self.fromRGB_layers[lod].name}')
            elif is_resolution_block_name(layer_name, self.resolution_log2):
                res = int(np.log2(int(extract_res_str(layer_name))))
                w = layer.get_weights()
                self.D_blocks[res].set_weights(w)
                logging.info(f'src layer name: {layer_name}')
                logging.info(f'target layer name: {self.D_blocks[res].name}')

    def trace_D_graphs(self, summary_writers, writers_dirs):
        D_prefix = 'Discriminator_'
        for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
            writer = summary_writers[res]
            # Change it later!
            profiler_dir = writers_dirs[res]
            trace_D_input = tf.random.normal(
                (1,) + self.D_input_shape(res), dtype=self.compute_dtype
            )

            if res == self.start_resolution_log2:
                trace_D_model = tf.function(
                    self.create_D_model(res, mode=STABILIZATION_MODE)
                )
                with writer.as_default():
                    # Fade-in model
                    tf.summary.trace_on(graph=True, profiler=True)
                    trace_D_model(trace_D_input)

                    tf.summary.trace_export(
                        name=D_prefix + STABILIZATION_MODE,
                        step=0,
                        profiler_outdir=profiler_dir
                    )
                    tf.summary.trace_off()

                    writer.flush()
            else:
                trace_D_model1 = tf.function(
                    self.create_D_model(res, mode=FADE_IN_MODE)
                )
                trace_D_model2 = tf.function(
                    self.create_D_model(res, mode=STABILIZATION_MODE)
                )
                with writer.as_default():
                    # Fade-in model
                    tf.summary.trace_on(graph=True, profiler=True)
                    trace_D_model1(trace_D_input)

                    tf.summary.trace_export(
                        name=D_prefix + FADE_IN_MODE,
                        step=0,
                        profiler_outdir=profiler_dir
                    )
                    tf.summary.trace_off()

                    # Stabilization model
                    tf.summary.trace_on(graph=True, profiler=True)
                    trace_D_model2(trace_D_input)

                    tf.summary.trace_export(
                        name=D_prefix + STABILIZATION_MODE,
                        step=0,
                        profiler_outdir=profiler_dir
                    )
                    tf.summary.trace_off()

                    writer.flush()

        print('All Discriminator models traced!')
