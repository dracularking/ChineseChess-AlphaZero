#!/usr/bin/env python3
"""
强制使用CPU进行训练的脚本
"""

import os
import sys

# 强制使用CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数"""
    print("=== 强制CPU训练模式 ===")
    print("设置环境变量强制使用CPU...")
    
    try:
        # 导入必要的模块
        from cchess_alphazero.config import Config
        from cchess_alphazero.worker.optimize import OptimizeWorker
        from cchess_alphazero.lib.tf_util import set_session_config
        
        # 使用mini配置
        config = Config(config_type='mini')
        
        # 强制CPU配置
        print("配置TensorFlow使用CPU...")
        session = set_session_config(per_process_gpu_memory_fraction=None, 
                                   allow_growth=None, 
                                   device_list='')
        
        print("开始CPU训练...")
        
        # 创建优化器并开始训练
        optimizer = OptimizeWorker(config)
        optimizer.start()
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
