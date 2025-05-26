#!/usr/bin/env python3
"""
简单的评估功能测试脚本
"""
import os
import sys
import logging

# 添加项目路径
_PATH_ = os.path.dirname(os.path.abspath(__file__))
if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

def test_config_loading():
    """测试配置加载"""
    print("=== 测试配置加载 ===")
    try:
        from cchess_alphazero.config import Config
        config = Config('mini')
        print(f"✓ 配置加载成功")
        print(f"  - 评估游戏数量: {config.eval.game_num}")
        print(f"  - 每步模拟次数: {config.eval.simulation_num_per_move}")
        print(f"  - 最大进程数: {config.eval.max_processes}")
        print(f"  - 最大游戏长度: {config.eval.max_game_length}")
        return config
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return None

def test_model_files():
    """测试模型文件是否存在"""
    print("\n=== 测试模型文件 ===")
    try:
        from cchess_alphazero.config import Config
        config = Config('mini')
        config.resource.create_directories()
        
        # 检查基准模型
        best_config = config.resource.model_best_config_path
        best_weight = config.resource.model_best_weight_path
        print(f"基准模型配置: {best_config}")
        print(f"基准模型权重: {best_weight}")
        print(f"  - 配置文件存在: {os.path.exists(best_config)}")
        print(f"  - 权重文件存在: {os.path.exists(best_weight)}")
        
        # 检查待评估模型
        ng_config = config.resource.next_generation_config_path
        ng_weight = config.resource.next_generation_weight_path
        print(f"待评估模型配置: {ng_config}")
        print(f"待评估模型权重: {ng_weight}")
        print(f"  - 配置文件存在: {os.path.exists(ng_config)}")
        print(f"  - 权重文件存在: {os.path.exists(ng_weight)}")
        
        if all([os.path.exists(f) for f in [best_config, best_weight, ng_config, ng_weight]]):
            print("✓ 所有模型文件都存在")
            return True
        else:
            print("✗ 部分模型文件缺失")
            return False
    except Exception as e:
        print(f"✗ 模型文件检查失败: {e}")
        return False

def test_model_loading():
    """测试模型加载"""
    print("\n=== 测试模型加载 ===")
    try:
        from cchess_alphazero.config import Config
        from cchess_alphazero.worker.evaluator import load_model
        
        config = Config('mini')
        config.resource.create_directories()
        
        print("正在加载基准模型...")
        model_bt = load_model(config, config.resource.model_best_config_path, config.resource.model_best_weight_path)
        if model_bt:
            print("✓ 基准模型加载成功")
        else:
            print("✗ 基准模型加载失败")
            return False
            
        print("正在加载待评估模型...")
        model_ng = load_model(config, config.resource.next_generation_config_path, config.resource.next_generation_weight_path)
        if model_ng:
            print("✓ 待评估模型加载成功")
        else:
            print("✗ 待评估模型加载失败")
            return False
            
        # 清理资源
        if hasattr(model_bt, 'close_pipes'):
            model_bt.close_pipes()
        if hasattr(model_ng, 'close_pipes'):
            model_ng.close_pipes()
            
        print("✓ 模型加载测试完成")
        return True
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment():
    """测试环境"""
    print("\n=== 测试环境 ===")
    try:
        import cchess_alphazero.environment.static_env as senv
        from cchess_alphazero.environment.env import CChessEnv
        
        # 测试静态环境
        state = senv.INIT_STATE
        print(f"✓ 初始状态: {state[:20]}...")
        
        # 测试环境类
        env = CChessEnv()
        env.reset()
        print(f"✓ 环境重置成功")
        
        return True
    except Exception as e:
        print(f"✗ 环境测试失败: {e}")
        return False

def test_player():
    """测试玩家"""
    print("\n=== 测试玩家 ===")
    try:
        from cchess_alphazero.config import Config
        from cchess_alphazero.agent.player import CChessPlayer
        from cchess_alphazero.worker.evaluator import load_model
        from collections import defaultdict
        from cchess_alphazero.agent.player import VisitState
        
        config = Config('mini')
        config.resource.create_directories()
        
        # 加载模型
        model = load_model(config, config.resource.model_best_config_path, config.resource.model_best_weight_path)
        if not model:
            print("✗ 无法加载模型")
            return False
            
        # 创建玩家
        search_tree = defaultdict(VisitState)
        pipes = model.get_pipes(need_reload=False)
        player = CChessPlayer(config, search_tree=search_tree, pipes=pipes, debugging=False, enable_resign=False)
        
        print("✓ 玩家创建成功")
        
        # 清理资源
        player.close()
        model.close_pipes()
        
        return True
    except Exception as e:
        print(f"✗ 玩家测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始评估功能测试...")
    
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)
    
    tests = [
        ("配置加载", test_config_loading),
        ("模型文件", test_model_files),
        ("环境", test_environment),
        ("模型加载", test_model_loading),
        ("玩家", test_player),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    print("\n=== 测试结果汇总 ===")
    passed = 0
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 项测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！评估功能应该可以正常工作。")
        print("\n建议运行完整评估:")
        print("python cchess_alphazero/run.py eval --type mini --gpu 0")
    else:
        print("⚠️  部分测试失败，请检查相关问题。")

if __name__ == "__main__":
    main()
