#!/usr/bin/env python3
"""
演示evolve命令的使用
"""
import sys
import os
import time
import subprocess
import signal

def demo_evolve_basic():
    """演示基本的evolve命令使用"""
    print("=" * 60)
    print("演示: evolve命令基本使用")
    print("=" * 60)
    
    print("\n1. 查看evolve命令帮助:")
    print("命令: python cchess_alphazero/run.py evolve --help")
    print("-" * 40)
    
    try:
        result = subprocess.run([
            sys.executable, "cchess_alphazero/run.py", "evolve", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # 只显示关键部分
            lines = result.stdout.split('\n')
            for line in lines:
                if 'evolve' in line or '--max-iterations' in line or '--skip-eval' in line:
                    print(line)
        else:
            print(f"错误: {result.stderr}")
    except Exception as e:
        print(f"异常: {e}")

def demo_evolve_commands():
    """演示不同的evolve命令用法"""
    print("\n" + "=" * 60)
    print("演示: evolve命令的不同用法")
    print("=" * 60)
    
    commands = [
        {
            "desc": "无限循环自动训练（推荐）",
            "cmd": "python cchess_alphazero/run.py evolve --type mini --gpu 0"
        },
        {
            "desc": "限制迭代次数（测试用）",
            "cmd": "python cchess_alphazero/run.py evolve --type mini --gpu 0 --max-iterations 5"
        },
        {
            "desc": "跳过评估步骤（加快训练）",
            "cmd": "python cchess_alphazero/run.py evolve --type mini --gpu 0 --skip-eval"
        },
        {
            "desc": "标准配置自动训练",
            "cmd": "python cchess_alphazero/run.py evolve --type normal --gpu 0"
        },
        {
            "desc": "分布式配置（高性能）",
            "cmd": "python cchess_alphazero/run.py evolve --type distribute --gpu 0"
        }
    ]
    
    for i, cmd_info in enumerate(commands, 1):
        print(f"\n{i}. {cmd_info['desc']}:")
        print(f"   {cmd_info['cmd']}")

def demo_evolve_features():
    """演示evolve命令的特性"""
    print("\n" + "=" * 60)
    print("演示: evolve命令的特性")
    print("=" * 60)
    
    features = [
        "🔄 自动循环执行：self-play → optimize → evaluate → self-play → ...",
        "⏹️  支持用户中断（Ctrl+C）优雅停止",
        "📊 详细的进度日志和时间统计",
        "🔢 可配置最大迭代次数",
        "⚡ 可选择跳过评估步骤",
        "⚙️  支持所有配置类型（mini/normal/distribute）",
        "🛡️  异常处理和恢复机制",
        "📝 完整的训练记录和模型保存"
    ]
    
    for feature in features:
        print(f"  {feature}")

def demo_evolve_workflow():
    """演示evolve命令的工作流程"""
    print("\n" + "=" * 60)
    print("演示: evolve命令的工作流程")
    print("=" * 60)
    
    workflow = [
        ("初始化", "加载配置，创建必要目录，设置信号处理"),
        ("第N轮开始", "记录开始时间，显示进度信息"),
        ("步骤1: 自我对弈", "AI与自己对弈生成训练数据"),
        ("步骤2: 模型优化", "使用对弈数据训练和优化模型"),
        ("步骤3: 模型评估", "评估新模型与当前最佳模型的性能（可选）"),
        ("第N轮完成", "记录耗时，更新统计信息"),
        ("检查停止条件", "用户中断、达到最大迭代次数等"),
        ("继续下一轮", "如果未满足停止条件，继续下一轮训练")
    ]
    
    for i, (step, desc) in enumerate(workflow, 1):
        print(f"{i:2d}. {step:12s} - {desc}")

def demo_evolve_tips():
    """演示evolve命令的使用技巧"""
    print("\n" + "=" * 60)
    print("演示: evolve命令使用技巧")
    print("=" * 60)
    
    tips = [
        {
            "场景": "快速测试功能",
            "建议": "使用 --type mini --max-iterations 1 --skip-eval",
            "原因": "快速验证功能是否正常工作"
        },
        {
            "场景": "日常训练",
            "建议": "使用 --type mini 或 --type normal",
            "原因": "平衡训练效果和时间消耗"
        },
        {
            "场景": "长期训练",
            "建议": "使用 --type normal 不设置max-iterations",
            "原因": "让模型持续进化直到手动停止"
        },
        {
            "场景": "快速迭代",
            "建议": "使用 --skip-eval 参数",
            "原因": "跳过评估步骤可以显著加快训练速度"
        },
        {
            "场景": "高性能环境",
            "建议": "使用 --type distribute",
            "原因": "充分利用多核CPU和高性能GPU"
        }
    ]
    
    for tip in tips:
        print(f"\n📌 {tip['场景']}:")
        print(f"   建议: {tip['建议']}")
        print(f"   原因: {tip['原因']}")

def demo_evolve_monitoring():
    """演示如何监控evolve命令的运行"""
    print("\n" + "=" * 60)
    print("演示: 监控evolve命令运行")
    print("=" * 60)
    
    print("1. 日志文件位置:")
    print("   - 主日志: data/log/main.log")
    print("   - 自我对弈日志: data/log/self_play.log")
    print("   - 优化日志: data/log/opt.log")
    print("   - 评估日志: data/log/eval.log")
    
    print("\n2. 模型文件位置:")
    print("   - 当前最佳模型: data/model/model_best_*")
    print("   - 下一代模型: data/model/next_generation/next_generation_*")
    
    print("\n3. 训练数据位置:")
    print("   - 对弈记录: data/play_data/")
    print("   - 游戏记录: data/play_record/")
    
    print("\n4. 监控命令示例:")
    print("   # 实时查看主日志")
    print("   tail -f data/log/main.log")
    print("   ")
    print("   # 查看模型文件")
    print("   ls -la data/model/")
    print("   ")
    print("   # 查看训练数据")
    print("   ls -la data/play_data/")

def main():
    """主演示函数"""
    print("🎯 ChineseChess-AlphaZero evolve命令演示")
    
    demos = [
        demo_evolve_basic,
        demo_evolve_commands,
        demo_evolve_features,
        demo_evolve_workflow,
        demo_evolve_tips,
        demo_evolve_monitoring
    ]
    
    for demo in demos:
        demo()
        input("\n按Enter键继续...")
    
    print("\n" + "=" * 60)
    print("🎉 演示完成！")
    print("现在你可以开始使用evolve命令进行自动训练了：")
    print("python cchess_alphazero/run.py evolve --type mini --gpu 0")
    print("=" * 60)

if __name__ == "__main__":
    main()
