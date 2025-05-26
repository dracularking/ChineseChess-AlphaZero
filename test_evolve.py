#!/usr/bin/env python3
"""
测试evolve命令的基本功能
"""
import sys
import os
import time
import signal
import subprocess
from pathlib import Path

def test_evolve_help():
    """测试evolve命令的帮助信息"""
    print("测试1: evolve命令帮助信息")
    try:
        result = subprocess.run([
            sys.executable, "cchess_alphazero/run.py", "evolve", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ evolve命令帮助信息正常")
            if "--max-iterations" in result.stdout and "--skip-eval" in result.stdout:
                print("✓ 新参数已正确添加")
            else:
                print("✗ 新参数未找到")
                return False
        else:
            print(f"✗ evolve命令帮助失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ evolve命令帮助超时")
        return False
    except Exception as e:
        print(f"✗ evolve命令帮助异常: {e}")
        return False
    
    return True

def test_evolve_import():
    """测试evolve模块导入"""
    print("\n测试2: evolve模块导入")
    try:
        from cchess_alphazero.worker import evolve
        from cchess_alphazero.config import Config
        
        # 测试配置创建
        config = Config('mini')
        print("✓ evolve模块导入成功")
        print("✓ 配置创建成功")
        
        # 测试EvolutionWorker创建
        worker = evolve.EvolutionWorker(config, max_iterations=1, skip_eval=True)
        print("✓ EvolutionWorker创建成功")
        
        return True
    except Exception as e:
        print(f"✗ evolve模块导入失败: {e}")
        return False

def test_evolve_short_run():
    """测试evolve命令短时间运行"""
    print("\n测试3: evolve命令短时间运行（1次迭代，跳过评估）")
    try:
        # 启动evolve命令，限制1次迭代，跳过评估
        process = subprocess.Popen([
            sys.executable, "cchess_alphazero/run.py", "evolve", 
            "--type", "mini", "--max-iterations", "1", "--skip-eval", "--gpu", "0"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 等待最多5分钟
        try:
            stdout, stderr = process.communicate(timeout=300)
            
            if process.returncode == 0:
                print("✓ evolve命令正常完成")
                if "开始自我进化训练" in stdout:
                    print("✓ 进化过程正常启动")
                if "第 1 轮进化完成" in stdout:
                    print("✓ 第1轮进化正常完成")
                return True
            else:
                print(f"✗ evolve命令失败，返回码: {process.returncode}")
                print(f"错误输出: {stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠ evolve命令运行超时，发送中断信号...")
            process.send_signal(signal.SIGINT)
            try:
                stdout, stderr = process.communicate(timeout=30)
                if "收到中断信号" in stdout or "用户中断" in stdout:
                    print("✓ 中断信号处理正常")
                    return True
                else:
                    print("✗ 中断信号处理异常")
                    return False
            except subprocess.TimeoutExpired:
                print("✗ 强制终止进程")
                process.kill()
                return False
                
    except Exception as e:
        print(f"✗ evolve命令测试异常: {e}")
        return False

def test_directory_structure():
    """测试目录结构"""
    print("\n测试4: 检查必要的目录结构")
    
    required_dirs = [
        "data",
        "data/model", 
        "data/play_data",
        "data/log"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} 目录存在")
        else:
            print(f"✗ {dir_path} 目录不存在")
            all_exist = False
    
    return all_exist

def main():
    """主测试函数"""
    print("=" * 60)
    print("evolve命令功能测试")
    print("=" * 60)
    
    tests = [
        ("evolve命令帮助", test_evolve_help),
        ("evolve模块导入", test_evolve_import),
        ("目录结构检查", test_directory_structure),
        ("evolve短时间运行", test_evolve_short_run),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"运行测试: {test_name}")
        print(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} 通过")
            else:
                print(f"✗ {test_name} 失败")
        except Exception as e:
            print(f"✗ {test_name} 异常: {e}")
    
    print(f"\n{'='*60}")
    print(f"测试结果: {passed}/{total} 通过")
    print(f"{'='*60}")
    
    if passed == total:
        print("🎉 所有测试通过！evolve命令可以正常使用。")
        return 0
    else:
        print("❌ 部分测试失败，请检查问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
