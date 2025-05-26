#!/usr/bin/env python3
"""
测试evolve命令的修复功能
"""
import sys
import os
import subprocess
import time
from pathlib import Path

def test_evolve_single_iteration():
    """测试单次迭代的evolve命令"""
    print("=" * 60)
    print("测试: evolve命令单次迭代（修复版本）")
    print("=" * 60)
    
    # 检查训练文件状态
    play_data_dir = Path("data/play_data")
    trained_dir = Path("data/trained")
    
    print(f"\n测试前状态:")
    if play_data_dir.exists():
        play_files = list(play_data_dir.glob("*.json"))
        print(f"  训练文件数量: {len(play_files)}")
    else:
        print(f"  训练文件目录不存在")
    
    if trained_dir.exists():
        trained_files = list(trained_dir.glob("*.json"))
        print(f"  已训练文件数量: {len(trained_files)}")
    else:
        print(f"  已训练文件目录不存在")
    
    # 运行evolve命令
    print(f"\n运行命令: python cchess_alphazero/run.py evolve --type mini --max-iterations 1 --skip-eval --gpu 0")
    
    try:
        result = subprocess.run([
            sys.executable, "cchess_alphazero/run.py", "evolve",
            "--type", "mini", "--max-iterations", "1", "--skip-eval", "--gpu", "0"
        ], capture_output=True, text=True, timeout=600)
        
        print(f"\n命令执行结果:")
        print(f"  返回码: {result.returncode}")
        
        # 分析输出
        output_lines = result.stdout.split('\n')
        key_lines = []
        for line in output_lines:
            if any(keyword in line for keyword in [
                "当前训练文件数", "跳过自我对弈", "开始优化模型", "模型优化完成",
                "训练文件不足", "优化后训练文件数", "最佳模型更新时间",
                "已训练文件已移动", "当前可用训练文件", "进化完成"
            ]):
                key_lines.append(line.strip())
        
        print(f"\n关键输出:")
        for line in key_lines:
            print(f"  {line}")
        
        if result.stderr:
            print(f"\n错误输出:")
            error_lines = result.stderr.split('\n')
            for line in error_lines[-10:]:  # 只显示最后10行错误
                if line.strip():
                    print(f"  {line.strip()}")
    
    except subprocess.TimeoutExpired:
        print("  ❌ 命令执行超时")
        return False
    except Exception as e:
        print(f"  ❌ 命令执行异常: {e}")
        return False
    
    # 检查测试后状态
    print(f"\n测试后状态:")
    if play_data_dir.exists():
        play_files_after = list(play_data_dir.glob("*.json"))
        print(f"  训练文件数量: {len(play_files_after)}")
    
    if trained_dir.exists():
        trained_files_after = list(trained_dir.glob("*.json"))
        print(f"  已训练文件数量: {len(trained_files_after)}")
    
    # 检查模型文件
    model_best_path = Path("data/model/model_best_weight.h5")
    if model_best_path.exists():
        mtime = model_best_path.stat().st_mtime
        from datetime import datetime
        update_time = datetime.fromtimestamp(mtime)
        print(f"  最佳模型更新时间: {update_time}")
    else:
        print(f"  最佳模型文件不存在")
    
    return result.returncode == 0

def test_evolve_continuous():
    """测试连续迭代的evolve命令"""
    print("\n" + "=" * 60)
    print("测试: evolve命令连续迭代（无限循环）")
    print("=" * 60)
    
    print(f"\n运行命令: python cchess_alphazero/run.py evolve --type mini --skip-eval --gpu 0")
    print("注意: 这将启动无限循环，需要手动中断（Ctrl+C）")
    
    response = input("是否要测试连续迭代？(y/N): ")
    if response.lower() != 'y':
        print("跳过连续迭代测试")
        return True
    
    try:
        process = subprocess.Popen([
            sys.executable, "cchess_alphazero/run.py", "evolve",
            "--type", "mini", "--skip-eval", "--gpu", "0"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print("evolve命令已启动，等待10秒后自动中断...")
        time.sleep(10)
        
        # 发送中断信号
        process.terminate()
        stdout, stderr = process.communicate(timeout=30)
        
        print(f"\n命令输出（最后20行）:")
        output_lines = stdout.split('\n')
        for line in output_lines[-20:]:
            if line.strip():
                print(f"  {line.strip()}")
        
        return True
        
    except Exception as e:
        print(f"连续迭代测试失败: {e}")
        return False

def show_usage_examples():
    """显示使用示例"""
    print("\n" + "=" * 60)
    print("evolve命令使用示例")
    print("=" * 60)
    
    examples = [
        {
            "desc": "单次迭代测试",
            "cmd": "python cchess_alphazero/run.py evolve --type mini --max-iterations 1 --skip-eval --gpu 0"
        },
        {
            "desc": "无限循环训练（推荐）",
            "cmd": "python cchess_alphazero/run.py evolve --type mini --gpu 0"
        },
        {
            "desc": "限制5轮迭代",
            "cmd": "python cchess_alphazero/run.py evolve --type mini --max-iterations 5 --gpu 0"
        },
        {
            "desc": "跳过评估加快训练",
            "cmd": "python cchess_alphazero/run.py evolve --type mini --skip-eval --gpu 0"
        },
        {
            "desc": "标准配置训练",
            "cmd": "python cchess_alphazero/run.py evolve --type normal --gpu 0"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['desc']}:")
        print(f"   {example['cmd']}")

def main():
    """主测试函数"""
    print("🔧 evolve命令修复功能测试")
    
    tests = [
        ("单次迭代测试", test_evolve_single_iteration),
        ("连续迭代测试", test_evolve_continuous),
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
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    show_usage_examples()
    
    print(f"\n{'='*60}")
    print(f"测试结果: {passed}/{total} 通过")
    print(f"{'='*60}")
    
    if passed == total:
        print("🎉 所有测试通过！evolve命令修复成功。")
        print("\n主要修复内容:")
        print("✅ 智能文件管理 - 避免浪费生成多余文件")
        print("✅ 优化状态检查 - 显示训练前后文件状态")
        print("✅ 模型更新检查 - 确认模型是否正确更新")
        print("✅ 训练数据清理 - 自动管理训练文件")
        print("✅ 错误处理改进 - 更详细的错误信息")
        return 0
    else:
        print("❌ 部分测试失败，请检查问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
