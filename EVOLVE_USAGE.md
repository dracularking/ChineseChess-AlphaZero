# Evolve命令使用指南

## 概述

`evolve` 命令是一个新增的自动化训练命令，它整合了 `self`（自我对弈）和 `opt`（模型优化）功能，能够不断地自我进化，直到用户中断为止。

## 基本用法

### 1. 无限循环自动训练（推荐 - 混合模式）

```bash
# 默认混合模式：self-play用GPU，optimize用CPU（推荐）
python cchess_alphazero/run.py evolve --type mini
```

这将启动无限循环的自动训练过程：
- **自我对弈**：使用GPU（推理速度快）
- **模型优化**：使用CPU（训练稳定）
- **模型评估**：使用GPU（评估速度快）
- 重复上述过程

### 2. 限制迭代次数

```bash
# 只进行5轮训练（混合模式）
python cchess_alphazero/run.py evolve --type mini --max-iterations 5
```

### 3. 跳过评估步骤（加快训练速度）

```bash
# 混合模式 + 跳过评估（推荐）
python cchess_alphazero/run.py evolve --type mini --skip-eval
```

### 4. 不同配置类型

```bash
# 标准配置（混合模式）
python cchess_alphazero/run.py evolve --type normal

# 分布式配置（混合模式）
python cchess_alphazero/run.py evolve --type distribute
```

### 5. 全GPU训练（仅在GPU环境完全稳定时使用）

```bash
# 全程使用GPU训练（self和opt都用GPU）
python cchess_alphazero/run.py evolve --type mini --gpu 0

# 全GPU训练 + 跳过评估
python cchess_alphazero/run.py evolve --type mini --gpu 0 --skip-eval

# 全GPU训练 + 限制迭代次数
python cchess_alphazero/run.py evolve --type mini --gpu 0 --max-iterations 5
```

## 命令参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--type` | 配置类型（mini/normal/distribute） | mini |
| `--gpu` | 指定GPU设备（不指定则使用混合模式） | 无（混合模式） |
| `--max-iterations` | 最大迭代次数（0表示无限） | 0 |
| `--skip-eval` | 跳过评估步骤 | False |

**重要变更**: 现在**默认使用混合模式**（self-play用GPU，optimize用CPU），只有明确指定`--gpu`参数时才全程使用GPU。

## 特性

### 🔄 自动循环执行
- **步骤1**: 自我对弈生成训练数据
- **步骤2**: 使用数据优化模型
- **步骤3**: 评估模型性能（可选）
- **重复**: 自动进入下一轮训练

### ⚡ 智能文件管理
- **智能停止**: 达到配置的文件数量上限后自动停止自我对弈
- **避免浪费**: 不会无限生成超出需要的训练文件
- **实时监控**: 每3秒检查一次文件生成进度
- **配置感知**: 根据不同配置（mini/normal/distribute）自动调整

### ⏹️ 优雅停止
- 支持 `Ctrl+C` 中断
- 会等待当前步骤完成后停止
- 保存所有训练进度和模型

### 📊 详细日志
- 每轮训练的详细时间统计
- 总训练时间和平均每轮时间
- 各步骤的执行状态
- 文件生成进度实时显示

### 🛡️ 异常处理
- 自动处理训练过程中的异常
- 评估失败不会中断整个训练过程
- 完整的错误日志记录
- 超时保护机制

## 使用场景

### 快速测试
```bash
# 测试功能是否正常（混合模式，推荐）
python cchess_alphazero/run.py evolve --type mini --max-iterations 1 --skip-eval
```

### 日常训练
```bash
# 混合模式训练（默认，推荐）
python cchess_alphazero/run.py evolve --type mini

# 全GPU训练（仅在GPU环境完全稳定时）
python cchess_alphazero/run.py evolve --type mini --gpu 0
```

### 长期训练
```bash
# 混合模式持续训练（默认，推荐）
python cchess_alphazero/run.py evolve --type normal

# 全GPU持续训练（仅在GPU完全稳定时）
python cchess_alphazero/run.py evolve --type normal --gpu 0
```

### 快速迭代
```bash
# 混合模式快速迭代（默认，推荐）
python cchess_alphazero/run.py evolve --type mini --skip-eval

# 全GPU快速迭代（仅在GPU完全稳定时）
python cchess_alphazero/run.py evolve --type mini --gpu 0 --skip-eval
```

## 监控训练过程

### 日志文件
- **主日志**: `data/log/main.log`
- **自我对弈日志**: `data/log/self_play.log`
- **优化日志**: `data/log/opt.log`
- **评估日志**: `data/log/eval.log`

### 模型文件
- **当前最佳模型**: `data/model/model_best_*`
- **下一代模型**: `data/model/next_generation/next_generation_*`

### 训练数据
- **对弈记录**: `data/play_data/`
- **游戏记录**: `data/play_record/`

### 实时监控命令
```bash
# 查看主日志
tail -f data/log/main.log

# 查看模型文件
ls -la data/model/

# 查看训练数据
ls -la data/play_data/
```

## 示例输出

```
============================================================
开始自我进化训练
配置类型: cchess_alphazero.configs.mini
最大迭代次数: 无限
跳过评估: 否
开始时间: 2025-05-26 16:30:00
============================================================

==================================================
开始第 1 轮进化
==================================================
[第1轮] 步骤1: 开始自我对弈生成训练数据...
[第1轮] 自我对弈完成，耗时: 120.50秒
[第1轮] 步骤2: 开始优化模型...
[第1轮] 模型优化完成，耗时: 85.30秒
[第1轮] 步骤3: 开始评估模型性能...
[第1轮] 模型评估完成，耗时: 45.20秒

第 1 轮进化完成
总耗时: 0:04:11
平均每轮耗时: 0:04:11
```

## 智能文件管理说明

evolve命令针对您提到的"浪费动作"问题进行了优化：

### 问题分析
- **Mini配置**: 每个文件包含1局游戏，最多保留10个文件
- **原始问题**: 自我对弈会无限循环，超过10个文件后会删除旧文件，造成浪费

### 解决方案
- **智能监控**: 实时监控训练文件数量
- **达标即停**: 当文件数达到配置上限（如mini的10个）时，自动停止监控
- **进度显示**: 显示文件生成进度，如 "文件生成进度: 8/10"
- **避免浪费**: 不再无限生成超出需要的文件

### 配置对比
| 配置类型 | 每文件游戏数 | 最大文件数 | 总游戏数 |
|---------|-------------|-----------|---------|
| mini | 1 | 10 | 10 |
| normal | 5 | 300 | 1500 |
| distribute | 5 | 300 | 1500 |

## 混合训练模式说明

### 🚀 新特性：智能混合训练模式

根据项目经验和性能优化，**evolve命令现在默认使用混合训练模式**：

### 混合模式优势
- ✅ **Self-play用GPU**: 推理速度快，生成训练数据效率高
- ✅ **Optimize用CPU**: 训练稳定，避免GPU内存错误
- ✅ **最佳平衡**: 兼顾速度和稳定性
- ✅ **智能切换**: 自动在GPU和CPU之间切换

### 训练模式对比
| 训练模式 | Self-play | Optimize | 稳定性 | 速度 | 推荐场景 |
|---------|-----------|----------|--------|------|---------|
| **混合模式** | GPU | CPU | ✅ 高 | 🚀 快 | **推荐用于所有场景** |
| 全GPU模式 | GPU | GPU | ⚠️ 中等 | 🚀 很快 | 仅在GPU环境完全稳定时 |
| 全CPU模式 | CPU | CPU | ✅ 最高 | 🐌 较慢 | 无GPU或GPU不稳定时 |

### 推荐命令
```bash
# 最推荐的命令（混合模式，速度与稳定性兼顾）
python cchess_alphazero/run.py evolve --type mini --skip-eval
```

## 注意事项

1. **硬件要求**: 确保有足够的GPU内存和存储空间
2. **时间安排**: 每轮训练可能需要较长时间，建议在空闲时运行
3. **中断恢复**: 可以随时中断并重新开始，训练进度会保存
4. **配置选择**:
   - `mini`: 适合测试和快速验证（10个文件即可开始优化）
   - `normal`: 适合日常训练（需要300个文件）
   - `distribute`: 适合高性能环境（需要300个文件）
5. **文件管理**: evolve命令会智能管理文件生成，避免不必要的浪费

## 故障排除

### 常见问题

1. **GPU内存不足**
   - 使用 `--type mini` 减少内存使用
   - 检查其他程序是否占用GPU

2. **训练速度慢**
   - 使用 `--skip-eval` 跳过评估
   - 考虑升级硬件配置

3. **中断后无法恢复**
   - 检查模型文件是否完整
   - 查看日志文件了解错误原因

4. **训练文件没有删除**
   - 训练文件会被移动到 `data/trained/` 目录，而不是删除
   - 这是正常行为，用于备份已训练的数据
   - evolve命令会自动管理文件数量

5. **模型没有更新**
   - 检查优化过程是否有错误
   - 查看日志中的"最佳模型更新时间"
   - 确保有足够的训练数据（mini配置需要至少1个文件）

6. **不再继续迭代**
   - 检查是否使用了 `--max-iterations` 参数
   - 如果要无限循环，不要设置此参数
   - 使用 `python cchess_alphazero/run.py evolve --type mini --gpu 0` 进行无限训练

### 获取帮助

```bash
# 查看完整帮助
python cchess_alphazero/run.py evolve --help

# 查看所有可用命令
python cchess_alphazero/run.py --help
```
