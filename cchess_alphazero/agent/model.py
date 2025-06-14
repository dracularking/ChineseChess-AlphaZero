import hashlib
import json
import os
from logging import getLogger

import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from keras.regularizers import l2

from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, ActionLabelsBlack

logger = getLogger(__name__)

class CChessModel:

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    per_process_gpu_memory_fraction=None,
                    allow_growth=True,
                    visible_device_list=self.config.opts.device_list
                )
            )
            self.session = tf.Session(config=config)
            K.set_session(self.session)
        self.digest = None
        self.n_labels = len(ActionLabelsRed)
        self.api = None

    def build(self):
        with self.graph.as_default():
            with self.session.as_default():
                mc = self.config.model

                # Force channels_last format to match existing models
                data_format = "channels_last"
                input_shape = (10, 9, 14)  # (batch, height, width, channels)
                bn_axis = -1

                # Store the data format for later use
                self.data_format = data_format
                in_x = x = Input(input_shape)

                x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_first_filter_size, padding="same",
                          data_format=data_format, use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                          name="input_conv-"+str(mc.cnn_first_filter_size)+"-"+str(mc.cnn_filter_num))(x)
                x = BatchNormalization(axis=bn_axis, name="input_batchnorm")(x)
                x = Activation("relu", name="input_relu")(x)

                for i in range(mc.res_layer_num):
                    x = self._build_residual_block(x, i + 1, data_format, bn_axis)

                res_out = x

                # for policy output
                x = Conv2D(filters=4, kernel_size=1, data_format=data_format, use_bias=False,
                            kernel_regularizer=l2(mc.l2_reg), name="policy_conv-1-2")(res_out)
                x = BatchNormalization(axis=bn_axis, name="policy_batchnorm")(x)
                x = Activation("relu", name="policy_relu")(x)
                x = Flatten(name="policy_flatten")(x)
                policy_out = Dense(self.n_labels, kernel_regularizer=l2(mc.l2_reg), activation="softmax", name="policy_out")(x)

                # for value output
                x = Conv2D(filters=2, kernel_size=1, data_format=data_format, use_bias=False,
                            kernel_regularizer=l2(mc.l2_reg), name="value_conv-1-4")(res_out)
                x = BatchNormalization(axis=bn_axis, name="value_batchnorm")(x)
                x = Activation("relu",name="value_relu")(x)
                x = Flatten(name="value_flatten")(x)
                x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu", name="value_dense")(x)
                value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)

                self.model = Model(in_x, [policy_out, value_out], name="cchess_model")

    def _build_residual_block(self, x, index, data_format="channels_first", bn_axis=1):
        mc = self.config.model
        in_x = x
        res_name = "res" + str(index)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format=data_format, use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name=res_name+"_conv1-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=bn_axis, name=res_name+"_batchnorm1")(x)
        x = Activation("relu",name=res_name+"_relu1")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format=data_format, use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name=res_name+"_conv2-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=bn_axis, name="res"+str(index)+"_batchnorm2")(x)
        x = Add(name=res_name+"_add")([in_x, x])
        x = Activation("relu", name=res_name+"_relu2")(x)
        return x

    @staticmethod
    def fetch_digest(weight_path):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()
        return None


    def load(self, config_path, weight_path):
        if os.path.exists(config_path) and os.path.exists(weight_path):
            logger.debug(f"loading model from {config_path}")
            with self.graph.as_default():
                with self.session.as_default():
                    with open(config_path, "rt") as f:
                        self.model = Model.from_config(json.load(f))
                    self.model.load_weights(weight_path)
                    # Compile the model after loading weights
                    self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer='adam')
                    # Initialize all variables
                    self.session.run(tf.global_variables_initializer())
            self.digest = self.fetch_digest(weight_path)
            logger.debug(f"loaded model digest = {self.digest}")
            return True
        else:
            logger.debug(f"model files does not exist at {config_path} and {weight_path}")
            return False

    def save(self, config_path, weight_path):
        logger.debug(f"save model to {config_path}")
        with open(config_path, "wt") as f:
            json.dump(self.model.get_config(), f)
            self.model.save_weights(weight_path)
        self.digest = self.fetch_digest(weight_path)
        logger.debug(f"saved model digest {self.digest}")

    def get_pipes(self, num=1, api=None, need_reload=True):
        if self.api is None:
            self.api = CChessModelAPI(self.config, self)
            self.api.start(need_reload)
        return self.api.get_pipe(need_reload)

    def close_pipes(self):
        if self.api is not None:
            self.api.close()
            self.api = None

