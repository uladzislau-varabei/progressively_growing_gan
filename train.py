import os
import sys
import argparse
import json
import logging
import time
from multiprocessing import Process

import numpy as np
# Note: do not import tensorflow here or you won't be able to train each stage
# in a new process

from utils import LOGS_DIR, TRAIN_MODE, INFERENCE_MODE,\
    FADE_IN_MODE, STABILIZATION_MODE,\
    DATASET_N_MAX_IMAGES, DEFAULT_DATASET_N_MAX_IMAGES,\
    IMAGES_PATHS_FILENAME, TARGET_RESOLUTION, START_RESOLUTION, DEFAULT_START_RESOLUTION
from utils import prepare_gpu
from model import ProGAN


def parse_args():
    parser = argparse.ArgumentParser(description='Script to train Progressive GAN model')
    parser.add_argument(
        '--config_path',
        help='Path to a config of a model to train (json format)',
        required=True
    )
    args = parser.parse_args()
    return args


def prepare_logger(config_path):
    if not os.path.exists(LOGS_DIR):
        os.mkdir(LOGS_DIR)

    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    filename = os.path.join(
        LOGS_DIR, 'logs_' + os.path.split(config_path)[1].split('.')[0] + '.txt'
    )
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(sh)
    root_logger.addHandler(fh)

    print('Logging initialized!')


def load_config(config_path):
    with open(config_path, 'r') as fp:
        config = json.load(fp)

    logging.info('Training with the following config:')
    logging.info(config)

    return config


def load_images_paths(config):
    images_paths_filename = config[IMAGES_PATHS_FILENAME]
    with open(images_paths_filename, 'r') as f:
        file_lines = f.readlines()
    images_paths = [x.strip() for x in file_lines]

    dataset_n_max_images = config.get(DATASET_N_MAX_IMAGES, DEFAULT_DATASET_N_MAX_IMAGES)
    if dataset_n_max_images > 0:
        print(f'Dataset n max images: {dataset_n_max_images}')
        if len(images_paths) > dataset_n_max_images:
            images_paths = images_paths[:dataset_n_max_images]

    logging.info(f'Total number of images: {len(images_paths)}')
    return images_paths


def run_process(target, args):
    p = Process(target=target, args=args)
    p.start()
    p.join()


def trace_graphs(config):
    pid = os.getpid()
    logging.info(f'Tracing graphs uses PID={pid}')
    prepare_gpu()
    ProGAN_model = ProGAN(config, mode=INFERENCE_MODE)
    ProGAN_model.trace_graphs()


def run_train_mode(config, images_paths, res, mode):
    pid = os.getpid()
    logging.info(f'Training for res={res} and mode={mode} uses PID={pid}')
    prepare_gpu()
    ProGAN_model = ProGAN(
        config, mode=TRAIN_MODE, images_paths=images_paths, res=res, stage=mode
    )
    ProGAN_model.train_stage(res=res, mode=mode)


def train_model(config):
    target_resolution = config[TARGET_RESOLUTION]
    resolution_log2 = int(np.log2(target_resolution))
    assert target_resolution == 2 ** resolution_log2 and target_resolution >= 4

    start_resolution = config.get(START_RESOLUTION, DEFAULT_START_RESOLUTION)
    start_resolution_log2 = int(np.log2(start_resolution))
    assert start_resolution == 2 ** start_resolution_log2 and start_resolution >= 4

    images_paths = load_images_paths(config)

    run_process(target=trace_graphs, args=(config,))
    sleep(3)

    train_start_time = time.time()

    for res in range(start_resolution_log2, resolution_log2 + 1):
        logging.info('Training %dx%d model...' % (2 ** res, 2 ** res))
        res_start_time = time.time()

        if res > start_resolution_log2:
            # The first resolution doesn't use alpha parameter,
            # but has usual number of steps for stabilization phase

            # Fade-in stage
            run_process(
                target=run_train_mode,
                args=(config, images_paths, res, FADE_IN_MODE)
            )
            sleep(1)

        # Stabilization stage
        run_process(
            target=run_train_mode,
            args=(config, images_paths, res, STABILIZATION_MODE)
        )
        sleep(1)

        res_total_time = time.time() - res_start_time
        logging.info(f'Training model of resolution {res} took {res_total_time:.3f} seconds\n\n')

    train_total_time = time.time() - train_start_time
    logging.info(f'Training finished in {train_total_time:.3f} seconds!')


def sleep(s):
    print(f"Sleeping {s}s...")
    time.sleep(s)
    print("Sleeping finished")


if __name__ == '__main__':
    args = parse_args()

    prepare_logger(args.config_path)
    config = load_config(args.config_path)

    # Training model of each stage in a separate process can be much faster
    # as all GPU resources are released after process is finished
    # Note: restoring/saving optimizer state is not implemented, so
    # if optimizers state should not be reset a model mush be trained in a single process
    single_process_training = False

    if single_process_training:
        prepare_gpu()
        ProGAN_model = ProGAN(config, mode=TRAIN_MODE, single_process_training=True)
        ProGAN_model.train()
    else:
        train_model(config)
