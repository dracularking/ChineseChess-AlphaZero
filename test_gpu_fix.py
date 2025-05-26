#!/usr/bin/env python3
"""
测试GPU修复的脚本
"""

import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gpu_detection():
    """测试GPU检测"""
    print("=== 测试GPU检测 ===")
    try:
        import tensorflow as tf
        gpu_available = len(tf.config.experimental.list_physical_devices('GPU')) > 0
        print(f"GPU可用: {gpu_available}")
        
        if gpu_available:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print(f"检测到的GPU: {gpus}")
        else:
            print("未检测到GPU，将使用CPU")
            
        return gpu_available
    except Exception as e:
        print(f"GPU检测失败: {e}")
        return False

def test_tf_session():
    """测试TensorFlow会话配置"""
    print("\n=== 测试TensorFlow会话配置 ===")
    try:
        from cchess_alphazero.lib.tf_util import set_session_config
        
        # 测试会话配置
        session = set_session_config(per_process_gpu_memory_fraction=0.3, allow_growth=True, device_list='0')
        print("会话配置成功")
        
        # 测试简单的TensorFlow操作
        import tensorflow as tf
        with session.as_default():
            a = tf.constant([1, 2, 3])
            b = tf.constant([4, 5, 6])
            c = tf.add(a, b)
            result = session.run(c)
            print(f"TensorFlow测试操作成功: {result}")
            
        return True
    except Exception as e:
        print(f"TensorFlow会话配置失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n=== 测试模型创建 ===")
    try:
        from cchess_alphazero.config import Config
        from cchess_alphazero.agent.model import CChessModel
        
        # 使用mini配置
        config = Config(config_type='mini')
        
        # 创建模型
        model = CChessModel(config)
        model.build()
        print("模型创建成功")
        
        # 测试模型输入形状
        print(f"模型输入形状: {model.model.input_shape}")
        print(f"模型输出形状: {[output.shape for output in model.model.outputs]}")
        
        return True
    except Exception as e:
        print(f"模型创建失败: {e}")
        return False

def test_import_fix():
    """测试move_to_action_idx导入修复"""
    print("\n=== 测试move_to_action_idx导入修复 ===")
    try:
        from cchess_alphazero.environment.static_env import move_to_action_idx
        
        # 测试几个动作转换
        test_moves = ['7374', '2311', '2535', '1123']
        for move in test_moves:
            idx = move_to_action_idx(move)
            print(f"动作 {move} -> 索引 {idx}")
            
        print("move_to_action_idx函数工作正常")
        return True
    except Exception as e:
        print(f"move_to_action_idx测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试GPU修复...")
    
    # 测试各个组件
    tests = [
        ("GPU检测", test_gpu_detection),
        ("TensorFlow会话", test_tf_session),
        ("导入修复", test_import_fix),
        ("模型创建", test_model_creation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"{test_name}测试出现异常: {e}")
            results[test_name] = False
    
    # 输出测试结果
    print("\n=== 测试结果汇总 ===")
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    # 总体结果
    all_passed = all(results.values())
    print(f"\n总体结果: {'所有测试通过' if all_passed else '部分测试失败'}")
    
    if all_passed:
        print("GPU修复成功，可以继续训练！")
    else:
        print("仍有问题需要解决，但基本功能应该可以工作。")

if __name__ == "__main__":
    main()
