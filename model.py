import os
import logging
import time
import random
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from losses import G_loss_fn, D_loss_fn
from utils import level_of_details, compute_alpha, update_wsum_alpha,\
    save_model,load_model,\
    FADE_IN_MODE, STABILIZATION_MODE, SMOOTH_POSTFIX,\
    create_images_dir_name, create_images_grid_title, remove_old_models,\
    getsize, validate_data_format, trace_vars, trace_message, mult_by_zero,\
    fp32, scale_loss, unscale_grads
from utils import TARGET_RESOLUTION, START_RESOLUTION, LATENT_SIZE, USE_MIXED_PRECISION,\
    DATA_FORMAT, MODEL_NAME, MAX_MODELS_TO_KEEP, \
    SUMMARY_EVERY, SAVE_MODEL_EVERY, SAVE_IMAGES_EVERY,\
    G_LEARNING_RATE, D_LEARNING_RATE, \
    G_LEARNING_RATE_DICT, D_LEARNING_RATE_DICT,\
    ADAM_BETA1, ADAM_BETA2, RESET_OPT_STATE_FOR_NEW_LOD,\
    USE_G_SMOOTHING, G_SMOOTHING_BETA, USE_GPU_FOR_GS,\
    GENERATOR_NAME, DISCRIMINATOR_NAME,\
    NCHW_FORMAT, NHWC_FORMAT,\
    TF_LOGS_DIR, STORAGE_PATH, DATASET_CACHE_FOLDER,\
    BATCH_SIZES, DATASET_MAX_CACHE_RES, IMAGES_PATHS_FILENAME,\
    FADE_IN_IMAGES, STABILIZATION_IMAGES,\
    DATASET_N_PARALLEL_CALLS, DATASET_N_PREFETCHED_BATCHES, DATASET_N_MAX_IMAGES,\
    SHUFFLE_DATASET, MIRROR_AUGMENT,\
    TRAIN_MODE, INFERENCE_MODE, DEFAULT_MODE
from utils import DEFAULT_STORAGE_PATH, DEFAULT_MAX_MODELS_TO_KEEP,\
    DEFAULT_SUMMARY_EVERY, DEFAULT_SAVE_MODEL_EVERY, DEFAULT_SAVE_IMAGES_EVERY,\
    DEFAULT_G_LEARNING_RATE, DEFAULT_D_LEARNING_RATE,\
    DEFAULT_G_LEARNING_RATE_DICT, DEFAULT_D_LEARNING_RATE_DICT,\
    DEFAULT_ADAM_BETA1, DEFAULT_ADAM_BETA2, DEFAULT_RESET_OPT_STATE_FOR_NEW_LOD,\
    DEFAULT_DATASET_MAX_CACHE_RES,\
    DEFAULT_START_RESOLUTION, DEFAULT_USE_MIXED_PRECISION,\
    DEFAULT_DATASET_N_PARALLEL_CALLS,\
    DEFAULT_DATASET_N_PREFETCHED_BATCHES, DEFAULT_DATASET_N_MAX_IMAGES,\
    DEFAULT_SHUFFLE_DATASET, DEFAULT_MIRROR_AUGMENT,\
    DEFAULT_USE_G_SMOOTHING, DEFAULT_G_SMOOTHING_BETA, DEFAULT_USE_GPU_FOR_GS,\
    DEFAULT_DATA_FORMAT
from dataset_utils import create_training_dataset, convert_outputs_to_images
from image_utils import fast_save_grid
from networks import Generator, Discriminator


class ProGAN():

    def __init__(self, config, mode=DEFAULT_MODE, images_paths=None, res=None, stage=None,
                 single_process_training=False):

        self.target_resolution = config[TARGET_RESOLUTION]
        self.resolution_log2 = int(np.log2(self.target_resolution))
        assert self.target_resolution == 2 ** self.resolution_log2 and self.target_resolution >= 4

        self.start_resolution = config.get(START_RESOLUTION, DEFAULT_START_RESOLUTION)
        self.start_resolution_log2 = int(np.log2(self.start_resolution))
        assert self.start_resolution == 2 ** self.start_resolution_log2 and self.start_resolution >= 4

        self.min_lod = level_of_details(self.resolution_log2, self.resolution_log2)
        self.max_lod = level_of_details(self.start_resolution_log2, self.resolution_log2)

        self.data_format = config.get(DATA_FORMAT, DEFAULT_DATA_FORMAT)
        validate_data_format(self.data_format)

        self.latent_size = config[LATENT_SIZE]
        if self.data_format == NCHW_FORMAT:
            self.z_dim = (self.latent_size, 1, 1)
        elif self.data_format == NHWC_FORMAT:
            self.z_dim = (1, 1, self.latent_size)

        self.model_name = config[MODEL_NAME]
        self.storage_path = config.get(STORAGE_PATH, DEFAULT_STORAGE_PATH)
        self.max_models_to_keep = config.get(MAX_MODELS_TO_KEEP, DEFAULT_MAX_MODELS_TO_KEEP)
        self.summary_every = config.get(SUMMARY_EVERY, DEFAULT_SUMMARY_EVERY)
        self.save_model_every = config.get(SAVE_MODEL_EVERY, DEFAULT_SAVE_MODEL_EVERY)
        self.save_images_every = config.get(SAVE_IMAGES_EVERY, DEFAULT_SAVE_IMAGES_EVERY)
        self.batch_sizes = config[BATCH_SIZES]
        self.fade_in_images = config[FADE_IN_IMAGES]
        self.stabilization_images = config[STABILIZATION_IMAGES]
        self.use_mixed_precision = config.get(USE_MIXED_PRECISION, DEFAULT_USE_MIXED_PRECISION)
        self.compute_dtype = 'float16' if self.use_mixed_precision else 'float32'
        self.use_Gs = config.get(USE_G_SMOOTHING, DEFAULT_USE_G_SMOOTHING)
        # Even in mixed precision training weights are stored in fp32
        self.smoothed_beta = tf.constant(
            config.get(G_SMOOTHING_BETA, DEFAULT_G_SMOOTHING_BETA), dtype='float32'
        )
        self.use_gpu_for_Gs = config.get(USE_GPU_FOR_GS, DEFAULT_USE_GPU_FOR_GS)
        self.shuffle_dataset = config.get(
            SHUFFLE_DATASET, DEFAULT_SHUFFLE_DATASET
        )
        self.dataset_n_parallel_calls = config.get(
            DATASET_N_PARALLEL_CALLS, DEFAULT_DATASET_N_PARALLEL_CALLS
        )
        self.dataset_n_prefetched_batches = config.get(
            DATASET_N_PREFETCHED_BATCHES, DEFAULT_DATASET_N_PREFETCHED_BATCHES
        )
        self.dataset_n_max_images = config.get(
            DATASET_N_MAX_IMAGES, DEFAULT_DATASET_N_MAX_IMAGES
        )
        self.dataset_max_cache_res = config.get(DATASET_MAX_CACHE_RES, DEFAULT_DATASET_MAX_CACHE_RES)
        self.mirror_augment = config.get(MIRROR_AUGMENT, DEFAULT_MIRROR_AUGMENT)

        self.G_learning_rate = config.get(G_LEARNING_RATE, DEFAULT_G_LEARNING_RATE)
        self.D_learning_rate = config.get(D_LEARNING_RATE, DEFAULT_D_LEARNING_RATE)
        self.G_learning_rate_dict = config.get(G_LEARNING_RATE_DICT, DEFAULT_G_LEARNING_RATE_DICT)
        self.D_learning_rate_dict = config.get(D_LEARNING_RATE_DICT, DEFAULT_D_LEARNING_RATE_DICT)
        self.beta1 = config.get(ADAM_BETA1, DEFAULT_ADAM_BETA1)
        self.beta2 = config.get(ADAM_BETA2, DEFAULT_ADAM_BETA2)
        self.reset_opt_state_for_new_lod = config.get(
            RESET_OPT_STATE_FOR_NEW_LOD, DEFAULT_RESET_OPT_STATE_FOR_NEW_LOD
        )

        self.valid_grid_nrows = 10
        self.valid_grid_ncols = 10
        self.valid_grid_padding = 2
        self.min_target_single_image_size = 2 ** 7
        self.max_png_res = 5
        self.valid_latents = self.generate_latents(self.valid_grid_nrows * self.valid_grid_ncols)

        self.logs_path = os.path.join(TF_LOGS_DIR, self.model_name)

        self.writers_dirs = {
            res: os.path.join(self.logs_path, '%dx%d' % (2 ** res, 2 ** res))
            for res in range(self.start_resolution_log2, self.resolution_log2 + 1)
        }

        self.summary_writers = {
            res: tf.summary.create_file_writer(self.writers_dirs[res])
            for res in range(self.start_resolution_log2, self.resolution_log2 + 1)
        }

        self.validate_config()

        self.clear_session_for_new_model = True

        self.G_object = Generator(config)
        self.D_object = Discriminator(config)

        if self.use_Gs:
            Gs_config = config
            self.Gs_valid_latents = self.valid_latents
            self.Gs_device = '/GPU:0' if self.use_gpu_for_Gs else '/CPU:0'

            if not self.use_gpu_for_Gs:
                Gs_config[DATA_FORMAT] = NHWC_FORMAT
                # NCHW -> NHWC
                self.toNHWC_axis = [0, 2, 3, 1]
                # NCHW -> NHWC -> NCHW
                self.toNCHW_axis = [0, 3, 1, 2]
                self.Gs_valid_latents = tf.transpose(self.valid_latents, self.toNHWC_axis)

            self.Gs_object = Generator(Gs_config)

        if mode == INFERENCE_MODE:
            self.initialize_models()
        elif mode == TRAIN_MODE:
            if single_process_training:
                self.initialize_models()
                self.create_images_generators(config)
                self.initialize_optimizers(create_all_variables=True)
            else:
                self.initialize_models(res, stage)
                self.create_images_generator(config, res, images_paths)
                self.initialize_optimizers(create_all_variables=False)

    def initialize_models(self, model_res=None, stage=None):
        self.G_object.initialize_G_model(model_res=model_res, mode=stage)
        self.D_object.initialize_D_model(model_res=model_res, mode=stage)
        if self.use_Gs:
            with tf.device(self.Gs_device):
                self.initialize_Gs_model(model_res=model_res, mode=stage)

    def initialize_Gs_model(self, model_res=None, mode=None):
        start_time = time.time()
        print('Initializing smoothed generator...')

        if model_res is not None:
            self.Gs_object.initialize_G_model(model_res=model_res, mode=mode)

            for res in range(2, model_res + 1):
                self.Gs_object.G_blocks[res].set_weights(
                    self.G_object.G_blocks[res].get_weights()
                )

            min_lod = level_of_details(model_res, self.resolution_log2)
            max_lod = min_lod + (1 if mode == FADE_IN_MODE else 0)

            for lod in range(min_lod, max_lod + 1):
                self.Gs_object.toRGB_layers[lod].set_weights(
                    self.G_object.toRGB_layers[lod].get_weights()
                )
        else:
            self.Gs_object.initialize_G_model()

            for res in range(2, self.resolution_log2 + 1):
                self.Gs_object.G_blocks[res].set_weights(
                    self.G_object.G_blocks[res].get_weights()
                )

            for lod in range(self.min_lod, self.max_lod + 1):
                self.Gs_object.toRGB_layers[lod].set_weights(
                    self.G_object.toRGB_layers[lod].get_weights()
                )

        total_time = time.time() - start_time
        logging.info(f'Smoothed generator initialized in {total_time:.3f} seconds!')

    def trace_graphs(self):
        self.G_object.trace_G_graphs(self.summary_writers, self.writers_dirs)
        self.D_object.trace_D_graphs(self.summary_writers, self.writers_dirs)

        self.G_object.initialize_G_model(model_res=self.resolution_log2, mode=FADE_IN_MODE)
        G_model = self.G_object.create_G_model(res=self.resolution_log2, mode=FADE_IN_MODE)
        logging.info('\nThe biggest Generator:\n')
        G_model.summary(print_fn=logging.info)

        self.D_object.initialize_D_model(model_res=self.resolution_log2, mode=FADE_IN_MODE)
        D_model = self.D_object.create_D_model(res=self.resolution_log2, mode=FADE_IN_MODE)
        logging.info('\nThe biggest Discriminator:\n')
        D_model.summary(print_fn=logging.info)

    def validate_config(self):
        for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
            if str(res) not in self.fade_in_images.keys():
                assert False, f'Missing fade-in images for res={res}'
            if str(res) not in self.stabilization_images.keys():
                assert False, f'Missing stabilization images for res={res}'
            if str(res) not in self.batch_sizes.keys():
                assert False, f'Missing batch size for res={res}'

    def initialize_G_optimizer(self, create_all_variables):
        self.G_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.G_learning_rate,
            beta_1=self.beta1, beta_2=self.beta2,
            epsilon=1e-8,
            name='G_Adam'
        )
        if self.use_mixed_precision:
            self.G_optimizer = mixed_precision.LossScaleOptimizer(self.G_optimizer, 'dynamic')

        if not create_all_variables:
            return

        step = tf.Variable(0, trainable=False, dtype=tf.int64)
        write_summary = tf.Variable(False, trainable=False, dtype=tf.bool)

        res = self.resolution_log2
        batch_size = self.batch_sizes[str(res)]

        G_model = self.G_object.create_G_model(res, mode=FADE_IN_MODE)
        D_model = self.D_object.create_D_model(res, mode=FADE_IN_MODE)

        latents = self.generate_latents(batch_size)

        G_vars = G_model.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as G_tape:
            G_tape.watch(G_vars)

            fake_images = G_model(latents)
            fake_scores = fp32(D_model(fake_images))

            G_loss = G_loss_fn(fake_scores, write_summary=write_summary, step=step)
            G_loss = scale_loss(G_loss, self.G_optimizer, self.use_mixed_precision)
            print('G loss computed')

        # No need to update weights!
        G_grads = mult_by_zero(G_tape.gradient(G_loss, G_vars))
        G_grads = unscale_grads(G_grads, self.G_optimizer, self.use_mixed_precision)
        print('G gradients obtained')

        self.G_optimizer.apply_gradients(zip(G_grads, G_vars))
        print('G gradients applied')

        print('Creating slots for intermediate output layers...')
        for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
            lod = level_of_details(res, self.resolution_log2)

            # toRGB layers
            to_layer = self.G_object.toRGB_layers[lod]
            to_input_shape = (batch_size,) + to_layer.input_shape[1:]
            to_inputs = tf.random.normal(shape=to_input_shape, mean=0., stddev=0.05, dtype=self.compute_dtype)

            with tf.GradientTape() as tape:
                to_outputs = to_layer(to_inputs)
                loss = tf.reduce_mean(tf.square(to_outputs))
                loss = scale_loss(loss, self.G_optimizer, self.use_mixed_precision)

            G_vars = to_layer.trainable_variables
            # No need to update weights!
            G_grads = mult_by_zero(tape.gradient(loss, G_vars))
            G_grads = unscale_grads(G_grads, self.G_optimizer, self.use_mixed_precision)
            self.G_optimizer.apply_gradients(zip(G_grads, G_vars))

        print('G optimizer slots created!')

    def initialize_D_optimizer(self, create_all_variables):
        self.D_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.D_learning_rate,
            beta_1=self.beta1, beta_2=self.beta2,
            epsilon=1e-8,
            name='D_Adam'
        )
        if self.use_mixed_precision:
            self.D_optimizer = mixed_precision.LossScaleOptimizer(self.D_optimizer, 'dynamic')

        if not create_all_variables:
            return

        # First step: create optimizer states for all internal and final output layers
        step = tf.Variable(0, trainable=False, dtype=tf.int64)
        write_summary = tf.Variable(False, trainable=False, dtype=tf.bool)

        res = self.resolution_log2
        batch_size = self.batch_sizes[str(res)]

        G_model = self.G_object.create_G_model(res, mode=FADE_IN_MODE)
        D_model = self.D_object.create_D_model(res, mode=FADE_IN_MODE)

        latents = self.generate_latents(batch_size)
        D_input_shape = self.D_object.D_input_shape(res)
        images = tf.random.normal(
            shape=(batch_size,) + D_input_shape, mean=0., stddev=0.05, dtype=self.compute_dtype
        )

        D_vars = D_model.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as D_tape:
            D_tape.watch(D_vars)

            fake_images = G_model(latents)
            fake_scores = fp32(D_model(fake_images))
            real_scores = fp32(D_model(images))

            D_loss = D_loss_fn(
                D_model,
                optimizer=self.D_optimizer, mixed_precision=self.use_mixed_precision,
                real_scores=real_scores, real_images=images,
                fake_scores=fake_scores, fake_images=fake_images,
                write_summary=write_summary, step=step
            )
            D_loss = scale_loss(D_loss, self.D_optimizer, self.use_mixed_precision)
            print('D loss computed')

        # No need to update weights!
        D_grads = mult_by_zero(D_tape.gradient(D_loss, D_vars))
        D_grads = unscale_grads(D_grads, self.D_optimizer, self.use_mixed_precision)
        print('D gradients obtained')

        self.D_optimizer.apply_gradients(zip(D_grads, D_vars))
        print('D gradients applied')

        print('Creating slots for intermediate output layers...')
        for res in range(self.start_resolution_log2, self.resolution_log2 + 1):
            lod = level_of_details(res, self.resolution_log2)

            # fromRGB layers
            from_layer = self.D_object.fromRGB_layers[lod]
            from_input_shape = (batch_size,) + from_layer.input_shape[1:]
            from_inputs = tf.random.normal(shape=from_input_shape, mean=0., stddev=0.05, dtype=self.compute_dtype)

            with tf.GradientTape() as tape:
                from_outputs = from_layer(from_inputs)
                loss = tf.reduce_mean(tf.square(from_outputs))
                loss = scale_loss(loss, self.D_optimizer, self.use_mixed_precision)

            D_vars = from_layer.trainable_variables
            # No need to update weights!
            D_grads = mult_by_zero(tape.gradient(loss, D_vars))
            D_grads = unscale_grads(D_grads, self.D_optimizer, self.use_mixed_precision)
            self.D_optimizer.apply_gradients(zip(D_grads, D_vars))

        print('D optimizer slots created!')

    def reset_optimizers_state(self):
        G_optimizer_new_weights = mult_by_zero(self.G_optimizer.get_weights())
        self.G_optimizer.set_weights(G_optimizer_new_weights)

        D_optimizer_new_weights = mult_by_zero(self.D_optimizer.get_weights())
        self.D_optimizer.set_weights(D_optimizer_new_weights)

        logging.info('Reset optimizer states')

    def restore_optimizers_state(self):
        logging.error("Restoring optimizer state not implemented")
        assert False, "Restoring optimizer state not implemented"

    def initialize_optimizers(self, create_all_variables):
        start_time = time.time()
        print('Initializing optimizers...')

        # 1: create optimizer states for all internal and final output layers
        # 2: create optimizer states for all intermediate output layers
        # 3: set slots of optimizer to zero
        self.initialize_G_optimizer(create_all_variables)
        self.initialize_D_optimizer(create_all_variables)

        self.reset_optimizers_state()

        total_time = time.time() - start_time
        logging.info(f'Optimizers initialized in {total_time:.3f} seconds!')

    def adjust_optimizers_learning_rate(self, res):
        G_lr = self.G_learning_rate_dict.get(str(res), self.G_learning_rate)
        self.G_optimizer.learning_rate.assign(G_lr)

        D_lr = self.D_learning_rate_dict.get(str(res), self.D_learning_rate)
        self.D_optimizer.learning_rate.assign(D_lr)

        # print(f'G_lr={self.G_optimizer.learning_rate.numpy()}, D_lr={self.D_optimizer.learning_rate.numpy()}')

    @tf.function
    def smooth_net_weights(self, Gs_model, G_model, beta):
        trace_message('...Tracing smoothing weights...')
        smoothed_net_vars = Gs_model.trainable_variables
        source_net_vars = G_model.trainable_variables
        trace_vars(smoothed_net_vars, 'Smoothed vars:')
        trace_vars(source_net_vars, 'Source vars:')

        with tf.device(self.Gs_device):
            for a, b in zip(smoothed_net_vars, source_net_vars):
                a.assign(b + (a - b) * beta)

    def create_images_generators(self, config):
        # This method is used in function train() which was used when training
        # all models used the same process
        start_time = time.time()
        print('Initializing images generators...')

        images_paths_filename = config[IMAGES_PATHS_FILENAME]
        with open(images_paths_filename, 'r') as f:
            file_lines = f.readlines()
        images_paths = [x.strip() for x in file_lines]

        if self.dataset_n_max_images > 0:
            print(f'Dataset n max images: {self.dataset_n_max_images}')
            if len(images_paths) > self.dataset_n_max_images:
                images_paths = images_paths[:self.dataset_n_max_images]

        logging.info(f'Total number of images: {len(images_paths)}')

        self.images_generators = {}

        for res in tqdm(range(self.start_resolution_log2, self.resolution_log2 + 1), desc='Generator res'):
            # No caching by default
            cache = False
            if res <= self.dataset_max_cache_res:
                cache = os.path.join(DATASET_CACHE_FOLDER, self.model_name, str(res))
                if STORAGE_PATH is not None:
                    cache = os.path.join(STORAGE_PATH, cache)
                if not os.path.exists(cache):
                    os.makedirs(cache)

            batch_size = self.batch_sizes[str(res)]
            # Shuffle data set at least once if it is not shuffling is disabled in config
            # No need to use now as dataset was optimized
            # random.shuffle(images_paths)
            self.images_generators[res] = create_training_dataset(
                images_paths, res, cache, batch_size,
                mirror_augment=self.mirror_augment,
                shuffle_dataset=self.shuffle_dataset,
                dtype=self.compute_dtype, data_format=self.data_format,
                n_parallel_calls=self.dataset_n_parallel_calls,
                n_prefetched_batches=self.dataset_n_prefetched_batches
            )

        total_time = time.time() - start_time
        logging.info(f'Images generators initialized in {total_time:.3f} seconds!')

    def create_images_generator(self, config, res, images_paths):
        # This method is used for training each model in a separate process
        start_time = time.time()
        print(f'Initializing images generator for res={res}...')

        # No caching by default
        cache = False
        if res <= self.dataset_max_cache_res:
            cache = os.path.join(DATASET_CACHE_FOLDER, self.model_name, str(res))
            if STORAGE_PATH is not None:
                cache = os.path.join(STORAGE_PATH, cache)
            if not os.path.exists(cache):
                os.makedirs(cache)

        batch_size = self.batch_sizes[str(res)]
        self.images_generator = create_training_dataset(
            images_paths, res, cache, batch_size,
            mirror_augment=self.mirror_augment,
            shuffle_dataset=self.shuffle_dataset,
            dtype=self.compute_dtype, data_format=self.data_format,
            n_parallel_calls=self.dataset_n_parallel_calls,
            n_prefetched_batches=self.dataset_n_prefetched_batches
        )

        total_time = time.time() - start_time
        logging.info(f'Image generator initialized in {total_time:.3f} seconds!')

    def create_models(self, res, mode):
        D_model = self.D_object.create_D_model(res, mode=mode)
        G_model = self.G_object.create_G_model(res, mode=mode)
        if self.use_Gs:
            Gs_model = self.Gs_object.create_G_model(res, mode=mode)
        else:
            Gs_model = None
        return D_model, G_model, Gs_model

    def update_models_weights(self):
        self.D_object.update_D_weights(self.D_model)
        self.G_object.update_G_weights(self.G_model)
        if self.use_Gs:
            self.Gs_object.update_G_weights(self.Gs_model)

    def save_model_wrapper(self, model_type, res, stage, step):
        Gs_name = 'G' + SMOOTH_POSTFIX
        assert model_type in ['G', 'D', Gs_name]
        MODEL_ARG = 'model'
        MODEL_TYPE_ARG = 'model_type'

        if model_type == 'D':
            kwargs = {MODEL_ARG: self.D_model, MODEL_TYPE_ARG: DISCRIMINATOR_NAME}
        elif model_type == 'G':
            kwargs = {MODEL_ARG: self.G_model, MODEL_TYPE_ARG: GENERATOR_NAME}
        elif model_type == Gs_name:
            kwargs = {
                MODEL_ARG: self.Gs_model,
                MODEL_TYPE_ARG: GENERATOR_NAME + SMOOTH_POSTFIX
            }

        shared_kwargs = {
            'model_name': self.model_name,
            'res': res,
            'resolution_log2': self.resolution_log2,
            'stage': stage,
            'step': step,
            'storage_path': self.storage_path
        }

        kwargs = {**kwargs, **shared_kwargs}
        save_model(**kwargs)

    def save_models(self, res, mode, step):
        self.save_model_wrapper(model_type='D', res=res, stage=mode, step=step)
        self.save_model_wrapper(model_type='G', res=res, stage=mode, step=step)
        if self.use_Gs:
            self.save_model_wrapper(model_type='G' + SMOOTH_POSTFIX, res=res, stage=mode, step=step)

    def save_valid_images(self, res, step, stage, smoothed=False):
        dir_stage = stage
        if smoothed:
            dir_stage += SMOOTH_POSTFIX

        digits_in_number = 6
        fname = ('%0' + str(digits_in_number) + 'd') % step

        valid_images_dir = create_images_dir_name(self.model_name, res, dir_stage)
        use_grid_title = False
        if use_grid_title:
            valid_images_grid_title = create_images_grid_title(res, dir_stage, step)
        else:
            valid_images_grid_title = None

        save_in_jpg = res > self.max_png_res

        if smoothed:
            valid_images = self.Gs_model(self.Gs_valid_latents)
            if not self.use_gpu_for_Gs:
                valid_images = tf.transpose(valid_images, self.toNCHW_axis)
        else:
            valid_images = self.G_model(self.valid_latents)

        valid_images = convert_outputs_to_images(
            valid_images,
            max(2 ** res, self.min_target_single_image_size),
            data_format=self.data_format
        ).numpy()

        fast_save_grid(
            out_dir=valid_images_dir,
            fname=fname,
            images=valid_images,
            title=valid_images_grid_title,
            nrows=self.valid_grid_nrows,
            ncols=self.valid_grid_ncols,
            padding=self.valid_grid_padding,
            save_in_jpg=save_in_jpg
        )

    @tf.function
    def generate_latents(self, batch_size):
        latents = tf.random.normal(
            shape=(batch_size,) + self.z_dim, mean=0., stddev=1., dtype=self.compute_dtype
        )
        return latents

    @tf.function
    def G_train_step(self, G_model, D_model, latents, write_summary, step):
        G_vars = G_model.trainable_variables
        # Trace which variables are used to make sure that net is training
        trace_vars(G_vars, 'Generator variables:')

        with tf.GradientTape(watch_accessed_variables=False) as G_tape:
            G_tape.watch(G_vars)

            fake_images = G_model(latents)
            fake_scores = fp32(D_model(fake_images))

            G_loss = G_loss_fn(fake_scores, write_summary=write_summary, step=step)
            G_loss = scale_loss(G_loss, self.G_optimizer, self.use_mixed_precision)

        G_grads = G_tape.gradient(G_loss, G_vars)
        G_grads = unscale_grads(G_grads, self.G_optimizer, self.use_mixed_precision)
        self.G_optimizer.apply_gradients(zip(G_grads, G_vars))

        if write_summary:
            # Write gradients
            with tf.name_scope('G-grads'):
                for grad, var in zip(G_grads, G_vars):
                    tf.summary.histogram(var.name, grad, step=step)
            # Write weights
            with tf.name_scope('G-weights'):
                for var in G_vars:
                    tf.summary.histogram(var.name, var, step=step)

    @tf.function
    def D_train_step(self, G_model, D_model, latents, images, write_summary, step):
        D_vars = D_model.trainable_variables
        # Trace which variables are used to make sure that net is training
        trace_vars(D_vars, 'Discriminator variables:')

        with tf.GradientTape(watch_accessed_variables=False) as D_tape:
            D_tape.watch(D_vars)

            # Note: fake images should not be casted to fp32 (see loss for details)
            fake_images = G_model(latents)
            fake_scores = fp32(D_model(fake_images))
            real_scores = fp32(D_model(images))

            D_loss = D_loss_fn(
                D_model,
                optimizer=self.D_optimizer, mixed_precision=self.use_mixed_precision,
                real_scores=real_scores, real_images=images,
                fake_scores=fake_scores, fake_images=fake_images,
                write_summary=write_summary, step=step
            )
            D_loss = scale_loss(D_loss, self.D_optimizer, self.use_mixed_precision)

        D_grads = D_tape.gradient(D_loss, D_vars)
        D_grads = unscale_grads(D_grads, self.D_optimizer, self.use_mixed_precision)
        self.D_optimizer.apply_gradients(zip(D_grads, D_vars))

        if write_summary:
            # Write gradients
            with tf.name_scope('D-grads'):
                for grad, var in zip(D_grads, D_vars):
                    tf.summary.histogram(var.name, grad, step=step)
            # Write weights
            with tf.name_scope('D-weights'):
                for var in D_vars:
                    tf.summary.histogram(var.name, var, step=step)

    @tf.function
    def train_step(self, G_model, D_model, G_latents, D_latents,
                   images, write_summary, step):
        trace_message(' ...Modified train step tracing... ')
        # Note: explicit use of G and D models allows one to make sure that
        # tf.function doesn't compile models. Additionally tracing is used
        # (previously for res=3 and mode=fade-in G model used variables only from res=2)
        self.D_train_step(G_model, D_model, D_latents, images, write_summary, step)
        self.G_train_step(G_model, D_model, G_latents, write_summary, step)

    def train(self):
        tStep = tf.Variable(0, trainable=False, dtype=tf.int64)
        tWriteSummary = tf.Variable(True, trainable=False, dtype=tf.bool)
        train_start_time = time.time()

        for res in tqdm(range(self.start_resolution_log2, self.resolution_log2 + 1), desc='Training res'):
            logging.info('Training %dx%d model...' % (2 ** res, 2 ** res))
            res_start_time = time.time()

            if self.reset_opt_state_for_new_lod:
                self.reset_optimizers_state()
            self.adjust_optimizers_learning_rate(res)

            images_generator = iter(self.images_generators[res])
            logging.info('Images generator size: {:.2f} Mbs'.format(
                getsize(images_generator, convert_to='Mb'))
            )
            batch_size = self.batch_sizes[str(res)]
            summary_writer = self.summary_writers[res]

            fade_in_steps = self.get_n_steps(res, FADE_IN_MODE)
            stabilization_steps = self.get_n_steps(res, STABILIZATION_MODE)

            with summary_writer.as_default():
                # The first resolution doesn't use alpha parameter,
                # but has usual number of steps for stabilization phase
                if res > self.start_resolution_log2:
                    # Fading in stage
                    fade_in_stage_start_time = time.time()

                    if self.clear_session_for_new_model:
                        logging.info('Clearing session...')
                        tf.keras.backend.clear_session()

                    self.D_model, self.G_model, self.Gs_model = self.create_models(res, mode=FADE_IN_MODE)

                    tStep.assign(0)
                    tWriteSummary.assign(True)

                    desc = f'Res {res}, fade-in steps'
                    for step in tqdm(range(fade_in_steps), desc=desc):
                        last_step_cond = step == (fade_in_steps - 1)

                        alpha = compute_alpha(step, fade_in_steps)
                        self.G_model = update_wsum_alpha(self.G_model, alpha)
                        self.D_model = update_wsum_alpha(self.D_model, alpha)
                        if self.use_Gs:
                            self.Gs_model = update_wsum_alpha(self.Gs_model, alpha)

                        write_summary = step % self.summary_every == 0 or last_step_cond
                        tWriteSummary.assign(write_summary)
                        if write_summary:
                            tStep.assign(step)
                            tf.summary.scalar('alpha', alpha, step=step)

                        G_latents = self.generate_latents(batch_size)
                        D_latents = self.generate_latents(batch_size)
                        batch_images = next(images_generator)

                        self.train_step(
                            G_model=self.G_model, D_model=self.D_model,
                            G_latents=G_latents, D_latents=D_latents, images=batch_images,
                            write_summary=tWriteSummary, step=tStep
                        )

                        if self.use_Gs:
                            self.smooth_net_weights(
                                Gs_model=self.Gs_model, G_model=self.G_model, beta=self.smoothed_beta
                            )

                        if write_summary:
                            summary_writer.flush()

                        if step % self.save_model_every == 0 or last_step_cond:
                            self.save_models(res=res, mode=FADE_IN_MODE, step=step)

                        if step % self.save_images_every == 0 or last_step_cond:
                            self.save_valid_images(res, step, stage=FADE_IN_MODE)
                            if self.use_Gs:
                                self.save_valid_images(res, step, stage=FADE_IN_MODE, smoothed=True)

                    self.update_models_weights()

                    remove_old_models(
                        self.model_name, res, stage=FADE_IN_MODE,
                        max_models_to_keep=self.max_models_to_keep, storage_path=self.storage_path
                    )

                    fade_in_stage_total_time = time.time() - fade_in_stage_start_time
                    logging.info(f'Fade-in stage took {fade_in_stage_total_time:.3f} seconds')

                # Stabilization stage
                stabilization_stage_start_time = time.time()

                if self.clear_session_for_new_model:
                    logging.info('Clearing session...')
                    tf.keras.backend.clear_session()

                self.D_model, self.G_model, self.Gs_model =\
                    self.create_models(res, mode=STABILIZATION_MODE)

                tStep.assign(fade_in_steps)
                tWriteSummary.assign(True)

                desc = f'Res {res}, stabilization steps'
                for step in tqdm(range(stabilization_steps), desc=desc):
                    last_step_cond = step == (stabilization_steps - 1)

                    write_summary = step % self.summary_every == 0 or last_step_cond
                    tWriteSummary.assign(write_summary)
                    if write_summary:
                        tStep.assign(step + fade_in_steps)

                    G_latents = self.generate_latents(batch_size)
                    D_latents = self.generate_latents(batch_size)
                    batch_images = next(images_generator)

                    self.train_step(
                        G_model=self.G_model, D_model=self.D_model,
                        G_latents=G_latents, D_latents=D_latents, images=batch_images,
                        write_summary=tWriteSummary, step=tStep
                    )

                    if self.use_Gs:
                        self.smooth_net_weights(
                            Gs_model=self.Gs_model, G_model=self.G_model, beta=self.smoothed_beta
                        )

                    if write_summary:
                        summary_writer.flush()

                    if step % self.save_model_every == 0 or last_step_cond:
                        self.save_models(res=res, mode=STABILIZATION_MODE, step=step)

                    if step % self.save_images_every == 0 or last_step_cond:
                        self.save_valid_images(res, step, STABILIZATION_MODE)
                        if self.use_Gs:
                            self.save_valid_images(res, step, STABILIZATION_MODE, smoothed=True)

                self.update_models_weights()

                remove_old_models(
                    self.model_name, res, stage=STABILIZATION_MODE,
                    max_models_to_keep=self.max_models_to_keep, storage_path=self.storage_path
                )

                stabilization_stage_total_time = time.time() - stabilization_stage_start_time
                logging.info(f'Stabilization stage took {stabilization_stage_total_time:.3f} seconds')

                res_total_time = time.time() - res_start_time
                logging.info(f'Training model of resolution {res} took {res_total_time:.3f} seconds\n\n')

        train_total_time = time.time() - train_start_time
        logging.info(f'Training finished in {train_total_time:.3f} seconds!')

    def get_n_steps(self, res, mode):
        batch_size = self.batch_sizes[str(res)]
        if mode == STABILIZATION_MODE:
            return int(np.ceil(self.stabilization_images[str(res)] / batch_size))
        elif mode == FADE_IN_MODE:
            return int(np.ceil(self.fade_in_images[str(res)] / batch_size))

    def load_trained_models(self, res, mode):
        self.D_model, self.G_model, self.Gs_model = self.create_models(res, mode=mode)

        step = self.get_n_steps(res, mode) - 1

        self.D_model = load_model(
            self.D_model, self.model_name, DISCRIMINATOR_NAME,
            res=res, resolution_log2=self.resolution_log2,
            stage=mode, step=step
        )
        self.G_model = load_model(
            self.G_model, self.model_name, GENERATOR_NAME,
            res=res, resolution_log2=self.resolution_log2,
            stage=mode, step=step
        )
        if self.use_Gs:
            self.Gs_model = load_model(
                self.Gs_model, self.model_name, GENERATOR_NAME + SMOOTH_POSTFIX,
                res=res, resolution_log2=self.resolution_log2,
                stage=mode, step=step
            )

        self.update_models_weights()
        logging.info('Weights loaded')

    def run_fade_in_stage(self, res):
        fade_in_stage_start_time = time.time()

        if not self.reset_opt_state_for_new_lod:
            self.restore_optimizers_state()
        self.adjust_optimizers_learning_rate(res)

        # Load weights from previous stage: res - 1 and stabilization mode
        logging.info(f'Loading models for res={res} and mode={FADE_IN_MODE}...')
        self.load_trained_models(res - 1, STABILIZATION_MODE)

        self.D_model, self.G_model, self.Gs_model = self.create_models(res, mode=FADE_IN_MODE)

        tStep = tf.Variable(0, trainable=False, dtype=tf.int64)
        tWriteSummary = tf.Variable(True, trainable=False, dtype=tf.bool)

        images_generator = iter(self.images_generator)
        batch_size = self.batch_sizes[str(res)]
        fade_in_steps = self.get_n_steps(res, FADE_IN_MODE)
        summary_writer = self.summary_writers[res]

        with summary_writer.as_default():

            desc = f'Res {res}, fade-in steps'
            for step in tqdm(range(fade_in_steps), desc=desc):
                last_step_cond = step == (fade_in_steps - 1)

                alpha = compute_alpha(step, fade_in_steps)
                self.G_model = update_wsum_alpha(self.G_model, alpha)
                self.D_model = update_wsum_alpha(self.D_model, alpha)
                if self.use_Gs:
                    self.Gs_model = update_wsum_alpha(self.Gs_model, alpha)

                write_summary = step % self.summary_every == 0 or last_step_cond
                tWriteSummary.assign(write_summary)
                if write_summary:
                    tStep.assign(step)
                    tf.summary.scalar('alpha', alpha, step=step)

                G_latents = self.generate_latents(batch_size)
                D_latents = self.generate_latents(batch_size)
                batch_images = next(images_generator)

                self.train_step(
                    G_model=self.G_model, D_model=self.D_model,
                    G_latents=G_latents, D_latents=D_latents, images=batch_images,
                    write_summary=tWriteSummary, step=tStep
                )

                if self.use_Gs:
                    self.smooth_net_weights(
                        Gs_model=self.Gs_model, G_model=self.G_model, beta=self.smoothed_beta
                    )

                if write_summary:
                    summary_writer.flush()

                if step % self.save_model_every == 0 or last_step_cond:
                    self.save_models(res=res, mode=FADE_IN_MODE, step=step)

                if step % self.save_images_every == 0 or last_step_cond:
                    self.save_valid_images(res, step, stage=FADE_IN_MODE)
                    if self.use_Gs:
                        self.save_valid_images(res, step, stage=FADE_IN_MODE, smoothed=True)

        self.update_models_weights()

        remove_old_models(
            self.model_name, res, stage=FADE_IN_MODE,
            max_models_to_keep=self.max_models_to_keep, storage_path=self.storage_path
        )

        fade_in_stage_total_time = time.time() - fade_in_stage_start_time
        logging.info(f'Fade-in stage took {fade_in_stage_total_time:.3f} seconds')

    def run_stabilization_stage(self, res):
        stabilization_stage_start_time = time.time()

        # Note: optimizers state can optionally be changed when running fade-in stage
        self.adjust_optimizers_learning_rate(res)

        if res > self.start_resolution_log2:
            logging.info(f'Loading models for res={res} and mode={STABILIZATION_MODE}...')
            # Load weights from previous stage: res and fade-in mode
            self.load_trained_models(res, FADE_IN_MODE)

        self.D_model, self.G_model, self.Gs_model = self.create_models(res, mode=STABILIZATION_MODE)

        images_generator = iter(self.images_generator)
        batch_size = self.batch_sizes[str(res)]
        summary_writer = self.summary_writers[res]

        fade_in_steps = self.get_n_steps(res, FADE_IN_MODE)
        stabilization_steps = self.get_n_steps(res, STABILIZATION_MODE)

        tStep = tf.Variable(fade_in_steps, trainable=False, dtype=tf.int64)
        tWriteSummary = tf.Variable(True, trainable=False, dtype=tf.bool)

        with summary_writer.as_default():

            desc = f'Res {res}, stabilization steps'
            for step in tqdm(range(stabilization_steps), desc=desc):
                last_step_cond = step == (stabilization_steps - 1)

                write_summary = step % self.summary_every == 0 or last_step_cond
                tWriteSummary.assign(write_summary)
                if write_summary:
                    tStep.assign(step + fade_in_steps)

                G_latents = self.generate_latents(batch_size)
                D_latents = self.generate_latents(batch_size)
                batch_images = next(images_generator)

                self.train_step(
                    G_model=self.G_model, D_model=self.D_model,
                    G_latents=G_latents, D_latents=D_latents, images=batch_images,
                    write_summary=tWriteSummary, step=tStep
                )

                if self.use_Gs:
                    self.smooth_net_weights(
                        Gs_model=self.Gs_model, G_model=self.G_model, beta=self.smoothed_beta
                    )

                if write_summary:
                    summary_writer.flush()

                if step % self.save_model_every == 0 or last_step_cond:
                    self.save_models(res=res, mode=STABILIZATION_MODE, step=step)

                if step % self.save_images_every == 0 or last_step_cond:
                    self.save_valid_images(res, step, STABILIZATION_MODE)
                    if self.use_Gs:
                        self.save_valid_images(res, step, STABILIZATION_MODE, smoothed=True)

        self.update_models_weights()

        remove_old_models(
            self.model_name, res, stage=STABILIZATION_MODE,
            max_models_to_keep=self.max_models_to_keep, storage_path=self.storage_path
        )

        stabilization_stage_total_time = time.time() - stabilization_stage_start_time
        logging.info(f'Stabilization stage took {stabilization_stage_total_time:.3f} seconds')

    def train_stage(self, res, mode):
        assert self.start_resolution_log2 <= res <= self.resolution_log2
        if mode == STABILIZATION_MODE:
            self.run_stabilization_stage(res)
        elif mode == FADE_IN_MODE:
            if res > self.start_resolution_log2:
                self.run_fade_in_stage(res)
