import os

import tensorflow as tf

from utils import DEFAULT_DTYPE, DEFAULT_DATA_FORMAT, NCHW_FORMAT,\
    DEFAULT_DATASET_N_PARALLEL_CALLS, DEFAULT_DATASET_N_PREFETCHED_BATCHES,\
    DEFAULT_SHUFFLE_DATASET, validate_data_format

MAX_CACHE_RESOLUTION = 7


def load_image(file_path, dtype=DEFAULT_DTYPE):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image


def preprocess_image(image, res):
    image = tf.image.resize(
        image,
        size=(2 ** res, 2 ** res),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    # Augmentations
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    return image


def normalize_images(images):
    # Scaling is performed with the function convert_image_dtype
    return 2 * images - 1.


def restore_images(images):
    return tf.cast((images + 1.) * 127.5, dtype=tf.uint8)


def convert_outputs_to_images(net_outputs, target_single_image_size,
                              data_format=DEFAULT_DATA_FORMAT):
    # Note: should work for linear and tanh activation
    validate_data_format(data_format)
    x = tf.clip_by_value(net_outputs, -1., 1.)
    x = restore_images(x)
    if data_format == NCHW_FORMAT:
        x = tf.transpose(x, [0, 2, 3, 1])
    x = tf.image.resize(
        x,
        size=(target_single_image_size, target_single_image_size),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return x


def load_and_preprocess_image(file_path, res, dtype=DEFAULT_DTYPE,
                              data_format=DEFAULT_DATA_FORMAT):
    validate_data_format(data_format)
    image = load_image(file_path, dtype)
    image = preprocess_image(image, res)
    image = normalize_images(image)
    if data_format == NCHW_FORMAT:
        image = tf.transpose(image, [2, 0, 1])
    return image


def create_training_dataset(fpaths, res, cache, batch_size,
                            shuffle_dataset=DEFAULT_SHUFFLE_DATASET,
                            dtype=DEFAULT_DTYPE,
                            data_format=DEFAULT_DATA_FORMAT,
                            n_parallel_calls=DEFAULT_DATASET_N_PARALLEL_CALLS,
                            n_prefetched_batches=DEFAULT_DATASET_N_PREFETCHED_BATCHES):
    ds = tf.data.Dataset.from_tensor_slices(fpaths)

    if shuffle_dataset:
        shuffle_buffer_size = len(fpaths)
        print('Shuffling dataset...')
        ds = ds.shuffle(
            buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True
        )

    ds = ds.map(
        lambda x: load_and_preprocess_image(
            x, res=res, dtype=dtype, data_format=data_format
        ),
        num_parallel_calls=n_parallel_calls
    )

    # cache can be a path to folder where files should be created
    # Note: when working with high resolutions there is no need to cache ds
    # as it consumes too much space on data storage (up to several GBs)
    if res <= MAX_CACHE_RESOLUTION:
        if isinstance(cache, str):
            ds = ds.cache(os.path.join(cache, str(res)))

    # Repeat forever
    ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=True)

    # Fetch batches in the background while model is training
    # If applied after ds.batch() then buffer_size is given in batches,
    # so total number of prefetched elements is batch_size * buffer_size
    ds = ds.prefetch(buffer_size=n_prefetched_batches)
    return ds
