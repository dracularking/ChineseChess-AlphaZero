#!/usr/bin/env python3
"""
测试CPU fallback功能
"""

import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_cpu_only():
    """强制使用CPU进行测试"""
    print("=== 强制CPU测试 ===")
    
    # 设置环境变量强制使用CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    try:
        import tensorflow as tf
        
        # 创建CPU配置
        config = tf.ConfigProto(
            device_count={'GPU': 0},
            allow_soft_placement=True,
            log_device_placement=False
        )
        
        session = tf.Session(config=config)
        
        # 测试简单操作
        with session.as_default():
            a = tf.constant([1, 2, 3])
            b = tf.constant([4, 5, 6])
            c = tf.add(a, b)
            result = session.run(c)
            print(f"CPU TensorFlow操作成功: {result}")
            
        session.close()
        return True
        
    except Exception as e:
        print(f"CPU测试失败: {e}")
        return False

def test_import_only():
    """只测试导入功能"""
    print("\n=== 测试导入功能 ===")
    try:
        from cchess_alphazero.environment.static_env import move_to_action_idx
        
        # 测试几个动作
        test_moves = ['7374', '2311']
        for move in test_moves:
            idx = move_to_action_idx(move)
            print(f"动作 {move} -> 索引 {idx}")
            
        print("导入功能正常")
        return True
    except Exception as e:
        print(f"导入测试失败: {e}")
        return False

def test_config():
    """测试配置加载"""
    print("\n=== 测试配置加载 ===")
    try:
        from cchess_alphazero.config import Config
        config = Config(config_type='mini')
        print(f"配置加载成功，批量大小: {config.trainer.batch_size}")
        return True
    except Exception as e:
        print(f"配置测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始简单测试...")
    
    tests = [
        ("配置加载", test_config),
        ("导入功能", test_import_only),
        ("CPU操作", test_cpu_only),
    ]
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            print(f"{test_name}: {'✓ 成功' if result else '✗ 失败'}")
        except Exception as e:
            print(f"{test_name}: ✗ 异常 - {e}")

if __name__ == "__main__":
    main()
