import os
import time
import gc
import subprocess
import shutil
import numpy as np
import tensorflow as tf

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from time import sleep
from random import shuffle
from threading import Thread

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.config import Config
from cchess_alphazero.lib.data_helper import get_game_data_filenames, read_game_data_from_file
from cchess_alphazero.lib.model_helper import load_best_model_weight, save_as_best_model
from cchess_alphazero.lib.model_helper import need_to_reload_best_model_weight, save_as_next_generation_model, save_as_best_model
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_policy, flip_move
from cchess_alphazero.lib.tf_util import set_session_config
from cchess_alphazero.lib.web_helper import http_request

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import backend as K

logger = getLogger(__name__)

def start(config: Config):
    # 如果device_list为空，强制使用CPU
    if config.opts.device_list == "":
        import os
        logger.info("强制使用CPU进行优化训练...")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
        # 设置CPU会话配置
        set_session_config(per_process_gpu_memory_fraction=None, allow_growth=None, device_list='')
    else:
        # 使用GPU配置
        set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)

    return OptimizeWorker(config).start()

class OptimizeWorker:
    def __init__(self, config:Config):
        self.config = config
        self.model = None
        self.loaded_filenames = set()
        self.loaded_data = deque(maxlen=self.config.trainer.dataset_size)
        self.dataset = deque(), deque(), deque()
        self.executor = ProcessPoolExecutor(max_workers=config.trainer.cleaning_processes)
        self.filenames = []
        self.opt = None
        self.count = 0
        self.eva = False

    def start(self):
        self.model = self.load_model()
        self.training()

    def training(self):
        self.compile_model()
        total_steps = self.config.trainer.start_total_steps
        bef_files = []
        last_file = None

        while True:
            files = get_game_data_filenames(self.config.resource)
            offset = self.config.trainer.min_games_to_begin_learn
            if (len(files) < self.config.trainer.min_games_to_begin_learn \
              or ((last_file is not None and last_file in files) and files.index(last_file) + 1 + offset > len(files))):
                # if last_file is not None:
                #     logger.info('Waiting for enough data 300s, ' + str((len(files) - files.index(last_file)) * self.config.play_data.nb_game_in_file) \
                #             +' vs '+ str(self.config.trainer.min_games_to_begin_learn)+' games')
                # else:
                #     logger.info('Waiting for enough data 300s, ' + str(len(files) * self.config.play_data.nb_game_in_file) \
                #             +' vs '+ str(self.config.trainer.min_games_to_begin_learn)+' games')
                # time.sleep(300)
                if last_file is not None:
                    self.save_current_model(send=True)
                break
            else:
                if last_file is not None and last_file in files:
                    idx = files.index(last_file) + 1
                    if len(files) - idx > self.config.trainer.load_step:
                        files = files[idx:idx + self.config.trainer.load_step]
                    else:
                        files = files[idx:]
                elif len(files) > self.config.trainer.load_step:
                    files = files[0:self.config.trainer.load_step]
                last_file = files[-1]
                logger.info(f"Last file = {last_file}")
                logger.debug(f"files = {files[0:-1:2000]}")
                self.filenames = deque(files)
                logger.debug(f"Start training {len(self.filenames)} files")
                shuffle(self.filenames)
                self.fill_queue()
                self.update_learning_rate(total_steps)
                if len(self.dataset[0]) > self.config.trainer.batch_size:
                    steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
                    total_steps += steps
                    self.save_current_model(send=False)
                    self.update_learning_rate(total_steps)
                    self.count += 1
                    a, b, c = self.dataset
                    a.clear()
                    b.clear()
                    c.clear()
                    del self.dataset, a, b, c
                    gc.collect()
                    self.dataset = deque(), deque(), deque()
                    self.backup_play_data(files)

    def train_epoch(self, epochs):
        tc = self.config.trainer
        state_ary, policy_ary, value_ary = self.collect_all_loaded_data()

        # 确保在模型的图和会话中训练模型
        with self.model.graph.as_default():
            with self.model.session.as_default():
                if self.config.opts.use_multiple_gpus:
                    self.mg_model.fit(state_ary, [policy_ary, value_ary],
                                     batch_size=tc.batch_size,
                                     epochs=epochs,
                                     shuffle=True,
                                     validation_split=0.02)
                else:
                    self.model.model.fit(state_ary, [policy_ary, value_ary],
                                     batch_size=tc.batch_size,
                                     epochs=epochs,
                                     shuffle=True,
                                     validation_split=0.02)
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def compile_model(self):
        # 使用 tf.keras 而不是独立的 keras
        from tensorflow.keras.optimizers import SGD
        # 确保在模型的图和会话中编译模型
        with self.model.graph.as_default():
            with self.model.session.as_default():
                self.opt = SGD(lr=0.02, momentum=self.config.trainer.momentum)
                losses = ['categorical_crossentropy', 'mean_squared_error']
                if self.config.opts.use_multiple_gpus:
                    self.mg_model = multi_gpu_model(self.model.model, gpus=self.config.opts.gpu_num)
                    self.mg_model.compile(optimizer=self.opt, loss=losses, loss_weights=self.config.trainer.loss_weights)
                else:
                    self.model.model.compile(optimizer=self.opt, loss=losses, loss_weights=self.config.trainer.loss_weights)

    def update_learning_rate(self, total_steps):
        # The deepmind paper says
        # ~400k: 1e-2
        # 400k~600k: 1e-3
        # 600k~: 1e-4

        lr = self.decide_learning_rate(total_steps)
        if lr:
            with self.model.graph.as_default():
                with self.model.session.as_default():
                    K.set_value(self.opt.lr, lr)
                    logger.debug(f"total step={total_steps}, set learning rate to {lr}")

    def fill_queue(self):
        futures = deque()
        n = len(self.filenames)
        with ProcessPoolExecutor(max_workers=self.config.trainer.cleaning_processes) as executor:
            for _ in range(self.config.trainer.cleaning_processes):
                if len(self.filenames) == 0:
                    break
                filename = self.filenames.pop()
                # logger.debug("loading data from %s" % (filename))
                futures.append(executor.submit(load_data_from_file, filename, self.config.opts.has_history))
            while futures and len(self.dataset[0]) < self.config.trainer.dataset_size: #fill tuples
                _tuple = futures.popleft().result()
                if _tuple is not None:
                    for x, y in zip(self.dataset, _tuple):
                        x.extend(y)
                m = len(self.filenames)
                if m > 0:
                    if (n - m) % 1000 == 0:
                        logger.info(f"Reading {n - m} files")
                    filename = self.filenames.pop()
                    # logger.debug("loading data from %s" % (filename))
                    futures.append(executor.submit(load_data_from_file, filename, self.config.opts.has_history))

    def collect_all_loaded_data(self):
        state_ary, policy_ary, value_ary = self.dataset

        state_ary1 = np.asarray(state_ary, dtype=np.float32)
        policy_ary1 = np.asarray(policy_ary, dtype=np.float32)
        value_ary1 = np.asarray(value_ary, dtype=np.float32)

        # Check if we need to convert from channels_first to channels_last
        if len(state_ary1.shape) == 4 and state_ary1.shape[1] == 14:
            # Data is in channels_first format (batch, channels, height, width)
            # Convert to channels_last format (batch, height, width, channels)
            logger.debug(f"Converting data from channels_first {state_ary1.shape} to channels_last")
            state_ary1 = np.transpose(state_ary1, (0, 2, 3, 1))
            logger.debug(f"Converted data shape: {state_ary1.shape}")

        return state_ary1, policy_ary1, value_ary1

    def load_model(self):
        model = CChessModel(self.config)

        # Check if GPU is available
        gpu_available = len(tf.config.experimental.list_physical_devices('GPU')) > 0

        # For CPU, we need to rebuild the model with channels_last format
        if not gpu_available:
            # Don't try to load the model, just build a new one
            model.build()
            save_as_best_model(model)
        else:
            # For GPU, we can load the existing model
            if self.config.opts.new or not load_best_model_weight(model):
                model.build()
                save_as_best_model(model)

        return model

    def save_current_model(self, send=False):
        logger.info("Save as ng model")
        if not send:
            save_as_best_model(self.model)
        else:
            save_as_next_generation_model(self.model)

    def decide_learning_rate(self, total_steps):
        ret = None

        for step, lr in self.config.trainer.lr_schedules:
            if total_steps >= step:
                ret = lr
        return ret

    def try_reload_model(self):
        logger.debug("check model")
        if need_to_reload_best_model_weight(self.model):
            with self.model.graph.as_default():
                load_best_model_weight(self.model)
            return True
        return False

    def backup_play_data(self, files):
        backup_folder = os.path.join(self.config.resource.data_dir, 'trained')
        cnt = 0
        if not os.path.exists(backup_folder):
            os.makedirs(backup_folder)
        for i in range(len(files)):
            try:
                shutil.move(files[i], backup_folder)
            except Exception as e:
                # logger.error(f"Backup error : {e}")
                cnt = cnt + 1
        logger.info(f"backup {len(files)} files, {cnt} empty files")

def load_data_from_file(filename, use_history=False):
    try:
        data = read_game_data_from_file(filename)
    except Exception as e:
        logger.error(f"Error when loading data {e}")
        os.remove(filename)
        return None
    if data is None:
        return None
    return expanding_data(data, use_history)

def expanding_data(data, use_history=False):
    state = data[0]
    real_data = []
    action = None
    policy = None
    value = None
    if use_history:
        history = [state]
    else:
        history = None
    for item in data[1:]:
        action = item[0]
        value = item[1]
        try:
            policy = build_policy(action, flip=False)
        except Exception as e:
            logger.error(f"Expand data error {e}, item = {item}, data = {data}, state = {state}")
            return None
        real_data.append([state, policy, value])
        state = senv.step(state, action)
        if use_history:
            history.append(action)
            history.append(state)

    return convert_to_trainging_data(real_data, history)


def convert_to_trainging_data(data, history):
    state_list = []
    policy_list = []
    value_list = []
    i = 0

    # Force channels_last format to match existing models
    data_format = "channels_last"

    for state, policy, value in data:
        if history is None:
            state_planes = senv.state_to_planes(state, data_format)
        else:
            state_planes = senv.state_history_to_planes(state, history[0:i * 2 + 1], data_format)
        sl_value = value

        state_list.append(state_planes)
        policy_list.append(policy)
        value_list.append(sl_value)
        i += 1

    state_array = np.asarray(state_list, dtype=np.float32)
    policy_array = np.asarray(policy_list, dtype=np.float32)
    value_array = np.asarray(value_list, dtype=np.float32)

    # Check if we need to convert from channels_first to channels_last
    if len(state_array.shape) == 4 and state_array.shape[1] == 14:
        # Data is in channels_first format (batch, channels, height, width)
        # Convert to channels_last format (batch, height, width, channels)
        logger.debug(f"Converting data from channels_first {state_array.shape} to channels_last")
        state_array = np.transpose(state_array, (0, 2, 3, 1))
        logger.debug(f"Converted data shape: {state_array.shape}")

    return state_array, policy_array, value_array

def build_policy(action, flip):
    labels_n = len(ActionLabelsRed)
    move_lookup = {move: i for move, i in zip(ActionLabelsRed, range(labels_n))}
    policy = np.zeros(labels_n)

    policy[move_lookup[action]] = 1

    if flip:
        policy = flip_policy(policy)
    return list(policy)



