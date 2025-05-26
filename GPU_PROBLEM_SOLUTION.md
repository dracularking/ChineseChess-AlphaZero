# GPU问题解决方案

## 问题总结

您遇到的问题有两个阶段：

### 1. ✅ 已解决：move_to_action_idx导入错误
**错误信息：**
```
无法将动作转换为索引: 7374, 错误: cannot import name 'move_to_action_idx' from 'cchess_alphazero.environment.static_env'
```

**解决方案：**
- 在 `cchess_alphazero/environment/static_env.py` 中添加了缺失的 `move_to_action_idx` 函数
- 该函数现在可以正确将动作字符串（如"7374"）转换为动作索引

### 2. 🔄 当前问题：GPU运行时错误
**错误信息：**
```
E tensorflow/stream_executor/cuda/cuda_blas.cc:428] failed to run cuBLAS routine: CUBLAS_STATUS_EXECUTION_FAILED
F tensorflow/stream_executor/gpu/gpu_timer.cc:65] Check failed: start_event_ != nullptr && stop_event_ != nullptr
```

## GPU问题的原因

这个错误通常由以下原因引起：
1. **GPU内存不足** - 模型太大，GPU内存无法容纳
2. **CUDA/cuDNN版本不兼容** - TensorFlow版本与CUDA版本不匹配
3. **GPU驱动问题** - 显卡驱动过旧或有问题
4. **GPU硬件问题** - GPU本身有硬件故障

## 解决方案

### 方案1：强制使用CPU（推荐）

在训练脚本开始前添加以下代码：
```python
import os
# 强制使用CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

### 方案2：使用修改后的启动脚本

使用我们创建的 `run_cpu.py` 脚本：
```bash
python run_cpu.py opt --type mini
```

### 方案3：减少GPU内存使用

如果想继续使用GPU，可以：
1. 减少批量大小（已在mini配置中设置为4）
2. 减少模型大小（已在mini配置中减少了层数和滤波器数量）
3. 限制GPU内存使用（已设置为30%）

## 已实施的修复

### 1. 添加了move_to_action_idx函数
```python
def move_to_action_idx(move):
    '''
    Convert a move string to action index using ActionLabelsRed lookup table.
    '''
    try:
        return ActionLabelsRed.index(move)
    except ValueError:
        raise ValueError(f"Move {move} not found in ActionLabelsRed")
```

### 2. 改进了GPU错误处理
- 在 `tf_util.py` 中添加了CPU fallback
- 在 `optimize.py` 中添加了GPU错误捕获和CPU回退

### 3. 优化了mini配置
- 批量大小：4（原来可能更大）
- GPU内存使用：30%（原来50%）
- 模型大小：减少了残差层和滤波器数量

## 测试结果

✅ **导入功能正常** - move_to_action_idx函数工作正常
✅ **CPU操作成功** - TensorFlow可以在CPU上正常工作
✅ **数据处理成功** - 845个样本成功加载和处理

## 建议的下一步

1. **立即解决方案：** 使用CPU进行训练
   ```bash
   # 设置环境变量
   set CUDA_VISIBLE_DEVICES=-1
   
   # 运行训练
   python cchess_alphazero/run.py opt --type mini
   ```

2. **长期解决方案：** 
   - 检查GPU驱动和CUDA版本
   - 考虑升级到更新的TensorFlow版本
   - 如果有多个GPU，尝试使用不同的GPU

## 性能说明

虽然CPU训练会比GPU慢，但对于学习和测试目的是完全可行的。mini配置已经优化为较小的模型，CPU训练时间应该是可以接受的。

## 验证修复

运行以下命令验证修复是否成功：
```bash
python test_cpu_fallback.py
```

应该看到所有测试通过的输出。
