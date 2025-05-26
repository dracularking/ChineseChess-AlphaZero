#!/usr/bin/env python3
"""
强制使用CPU的启动脚本
"""

import os
import sys

# 强制使用CPU - 必须在导入TensorFlow之前设置
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import backend as K

# 设置全局数据格式为 channels_first
K.set_image_data_format('channels_first')

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

from cchess_alphazero.lib.logger import setup_logger
from cchess_alphazero.config import Config
import cchess_alphazero.manager as manager

def main():
    print("=== 强制CPU模式启动 ===")
    print("已设置环境变量强制使用CPU")
    
    # 确保使用 channels_first 格式
    K.set_image_data_format('channels_first')
    
    # 验证CPU配置
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"可用GPU数量: {len(tf.config.experimental.list_physical_devices('GPU'))}")
    
    manager.start()

if __name__ == "__main__":
    main()
