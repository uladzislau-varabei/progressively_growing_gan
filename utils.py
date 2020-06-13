import os
import sys
import logging
from glob import glob
import shutil
from types import ModuleType, FunctionType
import platform
from gc import get_referents

import numpy as np
import tensorflow as tf

# Recommended for Tensorflow
NCHW_FORMAT = 'NCHW'
# Recommended by Nvidia
NHWC_FORMAT = 'NHWC'

DEFAULT_DATA_FORMAT = NCHW_FORMAT

FADE_IN_MODE = 'fade-in'
STABILIZATION_MODE = 'stabilization'
SMOOTH_POSTFIX = '_smoothed'
RGB_NAME = 'RGB'
LOD_NAME = 'lod'
WSUM_NAME = 'WSum'
WEIGHTS_FOLDER = 'weights'
GENERATOR_NAME = 'G_model'
DISCRIMINATOR_NAME = 'D_model'
LOGS_DIR = 'logs'
TF_LOGS_DIR = 'tf_logs'
IMAGES_DIR = 'images'
DATASET_CACHE_FOLDER = 'tf_ds_cache'
OS_LINUX = 'Linux'
OS_WIN = 'Windows'
HE_INIT = 'He'
LECUN_INIT = 'LeCun'

TRAIN_MODE = 'training'
INFERENCE_MODE = 'inference'
DEFAULT_MODE = INFERENCE_MODE

# ---------- Config options ----------
# Weights data format, NHWC and NCHW are supported
DATA_FORMAT = 'data_format'
# Model name (used in folders for logs and progress images)
MODEL_NAME = 'model_name'
# Additional prefix for storage
# Note: it was meant to be used if model was to be saved not in script directory,
# only consider using it if model is trained on server, otherwise just skip it
STORAGE_PATH = 'storage_path'
# How often to write summaries
SUMMARY_EVERY = 'summary_every'
# How often to save models weights
SAVE_MODEL_EVERY = 'save_model_every'
# How often to save progress images
SAVE_IMAGES_EVERY = 'save_images_every'
# Number of images for fade-in stage for each resolution (in a form of dict)
FADE_IN_IMAGES = 'fade_in_images'
# Number of images for stabilization stage for each resolution (in a form of dict)
STABILIZATION_IMAGES = 'stabilization_images'
# Base learning rate for generator
G_LEARNING_RATE = 'G_learning_rate'
# Base learning rate for discriminator
D_LEARNING_RATE = 'D_learning_rate'
# Adam beta 1 for generator and discriminator
ADAM_BETA1 = 'adam_beta1'
# Adam beta 2 for generator and discriminator
ADAM_BETA2 = 'adam_beta2'
# Reset optimizers states when new layers are introduced?
RESET_OPT_STATE_FOR_NEW_LOD = 'reset_opt_state_for_new_lod'
# Batch size for each resolution (in a form of dict)
BATCH_SIZES = 'batch_sizes'
# Max resolution of images to cache dataset (int, 2...max_resolution_log2)
# Note: only use this option of dataset is not very big and there is lots of available memory
MAX_CACHE_RES = 'max_cache_res'
# Path to a file with images paths
IMAGES_PATHS_FILENAME = 'images_paths_filename'
# Target resolution (should be a power of 2, e.g. 128, 256, etc.)
TARGET_RESOLUTION = 'target_resolution'
# Size of latent vector
LATENT_SIZE = 'latent_size'
# Apply pixel normalization to latent vector?
NORMALIZE_LATENTS = 'normalize_latents'
# Data type
# Note: it is not recommended to change it to float16, just use float32
DTYPE = 'dtype'
# Use bias layers?
USE_BIAS = 'use_bias'
# Use equalized learning rates?
USE_WSCALE = 'use_wscale'
# Use pixel normalization in generator?
USE_PIXELNORM = 'use_pixelnorm'
# Weights initialization technique: one of He, LeCun (should only be used with Selu)
WEIGHTS_INIT_MODE = 'weights_init_mode'
# Truncate weights?
TRUNCATE_WEIGHTS = 'truncate_weights'
# Override gain in projecting layer of generator to match the original paper implementation?
OVERRIDE_G_PROJECTING_GAIN = 'override_G_projecting_gain'
# Use fused layers in generator?
G_FUSED_SCALE = 'G_fused_scale'
# Use fused layers in discriminator?
D_FUSED_SCALE = 'D_fused_scale'
# Activation function in generator
G_ACTIVATION = 'G_activation'
# Activation function on discriminator
D_ACTIVATION = 'D_activation'
# Kernel size of convolutional layers in generator
# Note: only change default value if there is enough video memory,
# though values higher than 3 will lead to increasing training time
G_KERNEL_SIZE = 'G_kernel_size'
# Kernel size of convolutional layers in generator
# Note: only change default value if there is enough video memory,
# though values higher than 3 will lead to increasing training time
D_KERNEL_SIZE = 'D_kernel_size'
# Overall multiplier for the number of feature maps of generator
G_FMAP_BASE = 'G_fmap_base'
# log2 feature map reduction when doubling the resolution of generator
G_FMAP_DECAY = 'G_fmap_decay'
# Maximum number of feature maps in any layer of generator
G_FMAP_MAX = 'G_fmap_max'
# Overall multiplier for the number of feature maps of discriminator
D_FMAP_BASE = 'D_fmap_base'
# log2 feature map reduction when doubling the resolution of discriminator
D_FMAP_DECAY = 'D_fmap_decay'
# Maximum number of feature maps in any layer of discriminator
D_FMAP_MAX = 'D_fmap_max'
# Group size for minibatch standard deviation layer
MBSTD_GROUP_SIZE = 'mbstd_group_size'
# Number of filters in a projecting layer of discriminator
# Note: it should only be used if latent size is different from 512,
# it was meant to keep number of parameters in generator and discriminator at roughly
# the same layer
D_PROJECTING_NF = 'D_projecting_nf'
# Use smoothing og generator weights?
USE_G_SMOOTHING = 'use_G_smoothing'
# Beta for smoothing weights of generator
G_SMOOTHING_BETA = 'G_smoothed_beta'
# Number of parallel calls to dataset
DATASET_N_PARALLEL_CALLS = 'dataset_n_parallel_calls'
# Number of prefetched batches for dataset
DATASET_N_PREFETCHED_BATCHES = 'dataset_n_prefetched_batches'
# Maximum number of images to be used for training
DATASET_N_MAX_IMAGES = 'dataset_max_images'
# Shuffle dataset every time it is finished?
SHUFFLE_DATASET = 'shuffle_dataset'

DEFAULT_STORAGE_PATH = None
DEFAULT_MAX_CACHE_RES = -1
DEFAULT_USE_BIAS = True
DEFAULT_USE_WSCALE = True
DEFAULT_USE_PIXELNORM = True
DEFAULT_WEIGHTS_INIT_MODE = HE_INIT
DEFAULT_TRUNCATE_WEIGHTS = False
DEFAULT_OVERRIDE_G_PROJECTING_GAIN = True
DEFAULT_G_FUSED_SCALE = True
DEFAULT_D_FUSED_SCALE = True
DEFAULT_DTYPE = 'float32'
DEFAULT_G_KERNEL_SIZE = 3
DEFAULT_D_KERNEL_SIZE = 3
DEFAULT_G_LEARNING_RATE = 0.001
DEFAULT_D_LEARNING_RATE = 0.001
DEFAULT_ADAM_BETA1 = 0.
DEFAULT_ADAM_BETA2 = 0.99
DEFAULT_RESET_OPT_STATE_FOR_NEW_LOD = False
DEFAULT_USE_G_SMOOTHING = True
DEFAULT_G_SMOOTHING_BETA = 0.999
DEFAULT_DATASET_N_PARALLEL_CALLS = 4
DEFAULT_DATASET_N_PREFETCHED_BATCHES = 4
DEFAULT_DATASET_N_MAX_IMAGES = -1
DEFAULT_SHUFFLE_DATASET = False

# Note: by default Generator and Discriminator
# use the same values for these constants
# Note: for the light version described in the appendix set fmap_base to 2048
DEFAULT_FMAP_BASE = 8192
DEFAULT_FMAP_DECAY = 1.0
DEFAULT_FMAP_MAX = 512

HE_GAIN = np.sqrt(2.)
GAIN = {
    LECUN_INIT: 1.,
    HE_INIT: HE_GAIN
}

activation_funs_dict = {
    'relu': tf.nn.relu,
    'leaky_relu': tf.nn.leaky_relu,
    'selu': tf.nn.selu,
    'swish': tf.nn.swish
}

# For jupyter notebooks
EXAMPLE_IMGS_DIR = 'example_images'


def validate_data_format(data_format):
    assert data_format in [NCHW_FORMAT, NHWC_FORMAT]


def create_images_dir_name(model_name, res, mode):
    return os.path.join(IMAGES_DIR, model_name, '%dx%d' % (2 ** res, 2 ** res), mode)


def create_images_grid_title(res, mode, step):
    return f'{2 ** res}x{2 ** res}, mode={mode}, step={step}'


def level_of_details(res, resolution_log2):
    return resolution_log2 - res + 1


def compute_alpha(step, total_steps):
    return step / total_steps


def update_wsum_alpha(model, alpha):
    for layer in model.layers:
        if WSUM_NAME in layer.name:
            layer.set_weights([np.array(alpha)])
    return model


def trace_vars(vars, title):
    # Ugly way to trace variables:
    # Python side-effects will only happen once, when func is traced
    os_name = platform.system()
    if OS_LINUX == os_name:
        logging.info('\n' + title)
        [logging.info(var.name) for var in vars]
    else:
        # Calling logging inside tf.function on Windows can cause error
        print('\n' + title)
        [print(var.name) for var in vars]
        print()


def trace_message(message):
    os_name = platform.system()
    if os_name == OS_LINUX:
        logging.info(message)
    else:
        # Calling logging inside tf.function on Windows can cause errors
        print(message)


def mult_by_zero(weights):
    return [0. * w for w in weights]


def prepare_gpu():
    os_name = platform.system()
    os_message = f'\nScript is running on {os_name}, '

    # Use for Ubuntu
    set_memory_growth = False
    # Use for Windows
    set_memory_limit = False
    if os_name == OS_LINUX:
        print(os_message + 'memory growth option is used\n')
        set_memory_growth = True
    elif os_name == OS_WIN:
        print(os_message + 'memory limit option is used\n')
        set_memory_limit = True
        memory_limit = 7800
    else:
        print(
            os_message + f'GPU can only be configured for {OS_LINUX}|{OS_WIN}, '
            f'memory growth option is used\n'
        )
        set_memory_growth = True

    if set_memory_limit:
        physical_gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_gpus) > 0:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    physical_gpus[0],
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=memory_limit
                        )
                    ]
                )
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(
                    f"\nPhysical GPUs: {len(physical_gpus)}, "
                    f"logical GPUs: {len(logical_gpus)}"
                )
                print(f'Set memory limit to {memory_limit} Mbs\n')
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
        else:
            print('\nGPU is not available\n')

    if set_memory_growth:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print('\nGPU found. Set memory growth\n')
        else:
            print('\nGPU is not available\n')


def extract_res_str(s):
    # Note: if str doesn't start with a digit then empty string is returned
    digits = []
    for ch in s:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return ''.join(digits)


def is_resolution_block_name(s, resolution_log2):
    eps = 1e-8
    number_str = extract_res_str(s)
    res = int(number_str) if len(number_str) > 0 else None
    if res is not None:
        return 2 - eps <= int(np.log2(res)) <= resolution_log2 + eps
    else:
        return False


def create_model_folder_path(model_name, res, stage, step, model_type,
                             storage_path=DEFAULT_STORAGE_PATH):
    """
    model_name - name of configuration model
    model_type - one of [GENERATOR_NAME, DISCRIMINATOR_NAME],
                 used as a separate dir level,
    res - current resolution
    stage - one of [FADE_IN_MODE, STABILIZATION_MODE]
    step - number of step for given resolution and stage
    storage_path - optional prefix path
    """
    res_folder = 'res' + str(res)
    stage_folder = stage
    step_folder = 'step' + str(step)
    model_folder_path = os.path.join(
        WEIGHTS_FOLDER, model_name, res_folder, stage_folder, step_folder, model_type
    )
    if storage_path is not None:
        model_folder_path = os.path.join(storage_path, model_folder_path)
    return model_folder_path


def save_model(model, model_name, model_type, res, resolution_log2,
               stage, step, storage_path=DEFAULT_STORAGE_PATH):
    """
    model - a model to be saved
    model_name - name of configuration model
    model_type - one of [GENERATOR_NAME, DISCRIMINATOR_NAME],
                 used as a separate dir level
    res - current log2 resolution
    resolution_log2 - max log2 resolution
    stage - one of [FADE_IN_MODE, STABILIZATION_MODE]
    step - number of step for given resolution and stage
    storage_path - optional prefix path
    Note: should probably change this fun to standard way of saving model
    """
    model_folder_path = create_model_folder_path(
        model_name=model_name,
        res=res,
        stage=stage,
        step=step,
        model_type=model_type,
        storage_path=storage_path
    )
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    for layer in model.layers:
        layer_name = layer.name
        res_cond = is_resolution_block_name(layer_name, resolution_log2)
        if RGB_NAME in layer_name or res_cond:
            layer_weights = layer.get_weights()
            for idx, w in enumerate(layer_weights):
                fname = layer_name + '_' + str(idx) + '.npy'
                fname = os.path.join(model_folder_path, fname)
                np.save(fname, w, allow_pickle=False)


def load_model(model, model_name, model_type, res, resolution_log2,
               stage, step, storage_path=DEFAULT_STORAGE_PATH):
    """
    model - a model to be loaded
    model_name - name of configuration model
    model_type - one of [GENERATOR_NAME, DISCRIMINATOR_NAME],
                 used as a separate dir level
    res - current log2 resolution
    resolution_log2 - max log2 resolution
    stage - one of [FADE_IN_MODE, STABILIZATION_MODE]
    step - number of step for given resolution and stage
    storage_path - optional prefix path
    Note: should probably change this fun to standard way of loading model
    """
    model_folder_path = create_model_folder_path(
        model_name=model_name,
        res=res,
        stage=stage,
        step=step,
        model_type=model_type,
        storage_path=storage_path
    )
    assert os.path.exists(model_folder_path),\
        f"Can't load weights: folder {model_folder_path} does not exist"

    for layer in model.layers:
        layer_name = layer.name
        res_cond = is_resolution_block_name(layer_name, resolution_log2)
        if RGB_NAME in layer_name or res_cond:
            layer_weights = []
            for i in range(len(layer.get_weights())):
                fname = layer_name + '_' + str(i) + '.npy'
                fname = os.path.join(model_folder_path, fname)
                layer_weights.append(
                    np.load(fname, allow_pickle=False)
                )
            layer.set_weights(layer_weights)
            # print('Set weights for layer', layer_name)
    return model


def remove_old_models(model_name, res, stage, storage_path=DEFAULT_STORAGE_PATH):
    """
    model_name - name of configuration model
    model_type - one of [GENERATOR_NAME, DISCRIMINATOR_NAME],
                 used as a separate dir level
    res - current resolution
    storage_path - optional prefix path
    """
    # step and model_type are not used, so jut use valid values
    logging.info('\nRemoving weights...')
    weights_path = create_model_folder_path(
        model_name=model_name,
        res=res,
        stage=stage,
        step=1,
        model_type=GENERATOR_NAME,
        storage_path=storage_path
    )
    res_stage_path = os.path.split(
        os.path.split(weights_path)[0]
    )[0]
    sorted_steps_paths = sorted(
        glob(res_stage_path + os.sep + '*'),
        key=lambda x: int(x.split('step')[1])
    )
    # Remove weights for all steps except the last one
    for p in sorted_steps_paths[:-1]:
        shutil.rmtree(p)
        logging.info(f'Removed weights for path={p}')


def old_smooth_net_weights(smoothed_net, source_net, beta, resolution_log2):
    """
    Gs = G.clone('Gs')
    Gs_update_op = Gs.setup_as_moving_average_of(G, beta=G_smoothing)

    def: Setup_as_moving_average_of(self, src_net, beta=0.99)
        new_value = lerp(src_net.vars[name], var, beta)

    def lerp(a, b, t): return a + (b - a) * t == a - a * t + b * t = a(1 -t) + b * t
    """
    assert len(smoothed_net.layers) == len(source_net.layers)

    for idx in range(len(smoothed_net.layers)):
        layer_name = smoothed_net.layers[idx].name
        res_cond = is_resolution_block_name(layer_name, resolution_log2)
        if RGB_NAME in layer_name or res_cond:
            smoothed_net_layer_weights = []
            smoothed_weights = smoothed_net.layers[idx].get_weights()
            src_weights = source_net.layers[idx].get_weights()
            for smoothed_w, src_w in zip(smoothed_weights, src_weights):
                smoothed_net_layer_weights.append(
                    (1. - beta) * src_w + beta * smoothed_w
                )
            smoothed_net.layers[idx].set_weights(smoothed_net_layer_weights)
    return smoothed_net


def getsize(obj, convert_to=None):
    # Thanks to https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python
    # answer by Aaron Hall
    """sum size of object & members."""

    # Custom objects know their class.
    # Function objects seem to know way too much, including modules.
    # Exclude modules as well.
    BLACKLIST = (type, ModuleType, FunctionType)

    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: ' + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)

    if convert_to is not None:
        if convert_to == 'Mb':
            size = size / (1024 ** 2)
        elif convert_to == 'Gb':
            size = size / (1024 ** 3)
        elif convert_to == 'Kb':
            size = size / (1024 ** 1)
        else:
            print(f'Format {convert_to} is not supported. Size is given in bytes')

    return size
