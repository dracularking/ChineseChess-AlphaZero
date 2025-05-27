"""
自我进化模块 - 整合self-play和optimize，实现不断自我进化
"""
import os
import signal
import time
from logging import getLogger
from datetime import datetime

from cchess_alphazero.config import Config
from cchess_alphazero.worker import self_play, optimize, evaluator

logger = getLogger(__name__)

class EvolutionWorker:
    def __init__(self, config: Config, max_iterations=0, skip_eval=False, use_gpu=False, force_gpu_opt=False):
        """
        初始化进化工作器

        Args:
            config: 配置对象
            max_iterations: 最大迭代次数，0表示无限循环
            skip_eval: 是否跳过评估步骤
            use_gpu: 是否使用GPU进行self-play（默认混合模式：self用GPU，opt用CPU）
            force_gpu_opt: 是否强制optimization也使用GPU（默认False，总是用CPU）
        """
        self.config = config
        self.max_iterations = max_iterations
        self.skip_eval = skip_eval
        self.use_gpu = use_gpu
        self.force_gpu_opt = force_gpu_opt
        self.current_iteration = 0
        self.should_stop = False
        self.start_time = None

        # 检测配置类型（通过文件数量判断）
        self.is_large_config = self._detect_large_config()

        # 默认混合模式：self-play用GPU，optimize用CPU
        # 不在初始化时设置CPU环境，而是在具体步骤中动态设置

        # 注册信号处理器，用于优雅地停止进化过程
        signal.signal(signal.SIGINT, self._signal_handler)
        # Windows上不支持SIGTERM，只在非Windows系统上注册
        if os.name != 'nt':
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _detect_large_config(self):
        """检测是否为大型配置（normal/distribute）"""
        try:
            max_files = getattr(self.config.play_data, 'max_file_num', 10)
            # 如果最大文件数大于200，认为是大型配置（normal/distribute有300个文件）
            # mini配置现在有100个文件，仍然使用混合模式
            return max_files > 200
        except:
            return False

    def _setup_cpu_training(self):
        """设置CPU训练环境"""
        import os
        logger.info("设置CPU训练模式（默认）...")

        # 设置环境变量强制使用CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # 修改配置以适应CPU训练
        self.config.opts.device_list = ''

        # 设置数据格式为channels_first（CPU训练推荐）
        try:
            from tensorflow.keras import backend as K
            K.set_image_data_format('channels_first')
            logger.info("已设置数据格式为channels_first")
        except Exception as e:
            logger.warning(f"设置数据格式失败: {e}")

        logger.info("CPU训练模式设置完成")

    def _setup_gpu_for_selfplay(self):
        """设置GPU环境用于自我对弈"""
        import os
        if not self.use_gpu:  # 混合模式下，检查是否应该使用GPU
            # 检查配置类型，如果是大型配置（normal/distribute），使用CPU避免GPU冲突
            if self.is_large_config:
                logger.info("检测到大型配置（normal/distribute），使用CPU进行self-play以避免GPU冲突...")
                # 为大型配置强制使用CPU
                self._setup_cpu_for_selfplay()
                return

            # 对于mini配置，使用GPU进行self-play
            logger.info("设置GPU环境用于自我对弈...")

            # 清除CPU强制设置，允许使用GPU
            if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
                del os.environ['CUDA_VISIBLE_DEVICES']

            # 清除其他CPU强制设置
            if 'TF_FORCE_GPU_ALLOW_GROWTH' in os.environ:
                del os.environ['TF_FORCE_GPU_ALLOW_GROWTH']

            # 恢复GPU设备列表
            if hasattr(self.config.opts, 'device_list'):
                self.config.opts.device_list = '0'  # 使用GPU 0

            # 清理TensorFlow会话以应用新设置
            try:
                from tensorflow.keras import backend as K
                K.clear_session()
                logger.info("已清理TensorFlow会话以应用GPU设置")
            except Exception as e:
                logger.warning(f"清理TensorFlow会话失败: {e}")

            logger.info("GPU环境设置完成（用于自我对弈）")

    def _setup_cpu_for_selfplay(self):
        """设置CPU环境用于自我对弈（normal/distribute配置）"""
        import os
        logger.info("设置CPU环境用于自我对弈（避免GPU冲突）...")

        # 设置环境变量强制使用CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

        # 修改配置以适应CPU
        self.config.opts.device_list = ''
        self.config.opts.use_multiple_gpus = False

        # 清理TensorFlow会话
        try:
            from tensorflow.keras import backend as K
            K.clear_session()
            K.set_image_data_format('channels_last')
            logger.info("已设置CPU环境用于自我对弈")
        except Exception as e:
            logger.warning(f"设置CPU环境失败: {e}")

        logger.info("CPU环境设置完成（用于自我对弈）")

    def _setup_cpu_for_optimize(self):
        """设置CPU环境用于模型优化"""
        import os
        logger.info("设置CPU环境用于模型优化...")

        # 设置环境变量强制使用CPU（必须在TensorFlow初始化之前设置）
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

        # 修改配置以适应CPU训练
        self.config.opts.device_list = ''
        self.config.opts.use_multiple_gpus = False

        # 清理TensorFlow会话和图
        try:
            import tensorflow as tf
            # 清理默认图
            if hasattr(tf, 'reset_default_graph'):
                tf.reset_default_graph()

            # 清理Keras会话
            from tensorflow.keras import backend as K
            K.clear_session()

            # 设置数据格式为channels_last（与现有模型兼容）
            K.set_image_data_format('channels_last')

            logger.info("已清理TensorFlow会话并设置CPU模式")
        except Exception as e:
            logger.warning(f"清理TensorFlow会话失败: {e}")

        logger.info("CPU环境设置完成（用于模型优化）")

    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        _ = frame  # 忽略frame参数
        logger.info(f"收到中断信号 {signum}，正在优雅地停止进化过程...")
        self.should_stop = True

    def start(self):
        """开始进化过程"""
        self.start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("开始自我进化训练")
        logger.info(f"配置类型: {self.config.__class__.__module__}")
        logger.info(f"最大迭代次数: {'无限' if self.max_iterations == 0 else self.max_iterations}")
        logger.info(f"跳过评估: {'是' if self.skip_eval else '否'}")

        # 根据配置类型显示训练模式
        if self.force_gpu_opt:
            logger.info(f"训练模式: 全GPU模式（self-play和optimize都用GPU）")
        elif self.is_large_config:
            logger.info(f"训练模式: 全CPU模式（大型配置避免GPU冲突）")
        else:
            selfplay_device = "GPU" if self.use_gpu and not self.is_large_config else "CPU"
            logger.info(f"训练模式: 混合模式（self-play用{selfplay_device}，optimize用CPU）")

        logger.info(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        try:
            while not self.should_stop:
                self.current_iteration += 1

                # 检查是否达到最大迭代次数
                if self.max_iterations > 0 and self.current_iteration > self.max_iterations:
                    logger.info(f"达到最大迭代次数 {self.max_iterations}，停止进化")
                    break

                logger.info(f"\n{'='*50}")
                logger.info(f"开始第 {self.current_iteration} 轮进化")
                logger.info(f"{'='*50}")

                # 步骤1: 自我对弈生成训练数据
                if not self._run_self_play():
                    break

                # 步骤2: 优化模型
                if not self._run_optimize():
                    break

                # 步骤3: 评估模型（可选）
                if not self.skip_eval:
                    if not self._run_evaluation():
                        break

                # 记录本轮完成信息
                elapsed_time = datetime.now() - self.start_time
                logger.info(f"\n第 {self.current_iteration} 轮进化完成")
                logger.info(f"总耗时: {elapsed_time}")
                logger.info(f"平均每轮耗时: {elapsed_time / self.current_iteration}")

                # 清理训练数据（可选）
                self._cleanup_training_data()

                # 检查是否需要停止
                if self.should_stop:
                    logger.info("收到停止信号，结束进化过程")
                    break

        except KeyboardInterrupt:
            logger.info("用户中断，停止进化过程")
        except Exception as e:
            logger.error(f"进化过程中发生错误: {e}")
            raise
        finally:
            self._cleanup()

    def _run_self_play(self):
        """运行自我对弈（使用GPU）"""
        try:
            logger.info(f"[第{self.current_iteration}轮] 步骤1: 开始自我对弈生成训练数据（GPU模式）...")
            step_start_time = time.time()

            # 设置GPU环境用于自我对弈
            self._setup_gpu_for_selfplay()

            # 检查当前训练文件数量
            from cchess_alphazero.lib.data_helper import get_game_data_filenames
            files_before = get_game_data_filenames(self.config.resource)
            target_files = self.config.play_data.max_file_num
            current_files = len(files_before)

            logger.info(f"[第{self.current_iteration}轮] 当前训练文件数: {current_files}/{target_files}")

            if current_files >= target_files:
                logger.info(f"[第{self.current_iteration}轮] 训练文件已达到上限({target_files})，跳过自我对弈")
                return True

            # 导入并运行自我对弈
            if hasattr(self.config, 'play') and hasattr(self.config.play, 'max_processes'):
                import multiprocessing as mp
                if mp.get_start_method() == 'spawn':
                    from cchess_alphazero.worker import self_play_windows as sp_module
                else:
                    sp_module = self_play
            else:
                sp_module = self_play

            # 使用智能自我对弈
            self._run_smart_self_play(sp_module, target_files, current_files)

            step_time = time.time() - step_start_time
            logger.info(f"[第{self.current_iteration}轮] 自我对弈完成，耗时: {step_time:.2f}秒")
            return True

        except Exception as e:
            logger.error(f"[第{self.current_iteration}轮] 自我对弈失败: {e}")
            return False

    def _run_smart_self_play(self, sp_module, target_files, current_files):
        """智能自我对弈 - 只生成需要的文件数量"""
        # 计算需要生成的文件数量
        needed_files = target_files - current_files
        logger.info(f"[第{self.current_iteration}轮] 需要生成 {needed_files} 个训练文件")

        # 如果不需要生成文件，直接返回
        if needed_files <= 0:
            logger.info(f"[第{self.current_iteration}轮] 无需生成新文件")
            return

        # 使用监控式自我对弈
        logger.info(f"[第{self.current_iteration}轮] 开始智能自我对弈，目标: {target_files} 个文件")
        self._run_monitored_self_play(sp_module, target_files, needed_files)

    def _run_monitored_self_play(self, sp_module, target_files, needed_files):
        """监控式自我对弈 - 达到目标文件数后停止"""
        import threading
        import time
        from cchess_alphazero.lib.data_helper import get_game_data_filenames

        # 创建停止标志
        stop_flag = threading.Event()
        self_play_exception = None

        def run_self_play():
            """在单独线程中运行自我对弈"""
            nonlocal self_play_exception
            try:
                sp_module.start(self.config)
            except Exception as e:
                # 过滤掉线程池关闭后的正常错误
                if "cannot schedule new futures after shutdown" in str(e):
                    logger.debug(f"自我对弈线程正常结束: {e}")
                else:
                    self_play_exception = e
                    logger.error(f"自我对弈线程异常: {e}")

        # 启动自我对弈线程
        self_play_thread = threading.Thread(target=run_self_play, daemon=True)
        self_play_thread.start()

        # 监控文件生成进度
        start_time = time.time()
        check_interval = 3  # 每3秒检查一次
        max_wait_time = 1800  # 最多等待30分钟
        initial_file_count = len(get_game_data_filenames(self.config.resource))
        last_file_count = initial_file_count

        logger.info(f"[第{self.current_iteration}轮] 开始监控文件生成，当前: {initial_file_count}, 目标: {target_files}")

        while not stop_flag.is_set():
            time.sleep(check_interval)

            # 检查是否超时
            elapsed_time = time.time() - start_time
            if elapsed_time > max_wait_time:
                logger.warning(f"[第{self.current_iteration}轮] 自我对弈超时({max_wait_time}秒)，停止监控")
                break

            # 检查是否收到停止信号
            if self.should_stop:
                logger.info(f"[第{self.current_iteration}轮] 收到停止信号，中断自我对弈监控")
                break

            # 检查自我对弈线程是否异常
            if self_play_exception:
                logger.error(f"[第{self.current_iteration}轮] 自我对弈异常，停止监控: {self_play_exception}")
                break

            # 检查文件数量
            current_file_count = len(get_game_data_filenames(self.config.resource))
            # 修正计算逻辑：本轮生成的文件数 = 当前文件数 - 初始文件数
            generated_this_round = current_file_count - initial_file_count

            if current_file_count != last_file_count:
                logger.info(f"[第{self.current_iteration}轮] 文件生成进度: {current_file_count}/{target_files} "
                           f"(本轮已生成: {max(0, generated_this_round)})")
                last_file_count = current_file_count

            # 如果达到目标文件数量，停止监控
            if current_file_count >= target_files:
                logger.info(f"[第{self.current_iteration}轮] 已达到目标文件数量({target_files})，停止自我对弈监控")
                break

        # 记录最终状态
        final_file_count = len(get_game_data_filenames(self.config.resource))
        final_generated = final_file_count - initial_file_count
        logger.info(f"[第{self.current_iteration}轮] 自我对弈监控结束，最终文件数: {final_file_count} (本轮生成: {max(0, final_generated)})")

        # 等待自我对弈线程结束（最多等待5秒）
        if self_play_thread.is_alive():
            logger.info(f"[第{self.current_iteration}轮] 等待自我对弈线程结束...")
            self_play_thread.join(timeout=5)
            if self_play_thread.is_alive():
                logger.warning(f"[第{self.current_iteration}轮] 自我对弈线程未能及时结束，继续执行")

        # 注意：自我对弈进程可能在后台继续运行，但我们已经达到了目标
        # 新生成的文件会替换旧文件，不会影响训练过程

    def _run_optimize(self):
        """运行模型优化（默认使用CPU以避免GPU错误）"""
        try:
            # 决定优化使用的设备
            use_cpu_for_opt = not self.force_gpu_opt  # 除非强制使用GPU，否则总是用CPU
            device_name = "CPU" if use_cpu_for_opt else "GPU"
            logger.info(f"[第{self.current_iteration}轮] 步骤2: 开始优化模型（{device_name}模式）...")
            step_start_time = time.time()

            # 设置CPU环境用于优化（除非强制使用GPU）
            if use_cpu_for_opt:
                self._setup_cpu_for_optimize()

            # 检查训练数据是否足够
            from cchess_alphazero.lib.data_helper import get_game_data_filenames
            files_before = get_game_data_filenames(self.config.resource)
            min_games = self.config.trainer.min_games_to_begin_learn

            if len(files_before) < min_games:
                logger.warning(f"[第{self.current_iteration}轮] 训练文件不足: {len(files_before)}/{min_games}，跳过优化")
                return True

            logger.info(f"[第{self.current_iteration}轮] 使用 {len(files_before)} 个训练文件进行优化")

            # 运行优化
            optimize.start(self.config)

            # 检查优化后的状态
            files_after = get_game_data_filenames(self.config.resource)
            logger.info(f"[第{self.current_iteration}轮] 优化后训练文件数: {len(files_after)} (优化前: {len(files_before)})")

            # 检查模型是否更新
            self._check_model_update()

            step_time = time.time() - step_start_time
            logger.info(f"[第{self.current_iteration}轮] 模型优化完成，耗时: {step_time:.2f}秒")
            return True

        except Exception as e:
            logger.error(f"[第{self.current_iteration}轮] 模型优化失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return False

    def _check_model_update(self):
        """检查模型是否更新"""
        try:
            import os
            model_path = self.config.resource.model_best_weight_path
            if os.path.exists(model_path):
                mtime = os.path.getmtime(model_path)
                from datetime import datetime
                update_time = datetime.fromtimestamp(mtime)
                logger.info(f"[第{self.current_iteration}轮] 最佳模型更新时间: {update_time}")
            else:
                logger.warning(f"[第{self.current_iteration}轮] 最佳模型文件不存在: {model_path}")
        except Exception as e:
            logger.error(f"[第{self.current_iteration}轮] 检查模型更新失败: {e}")

    def _cleanup_training_data(self):
        """清理训练数据"""
        try:
            from cchess_alphazero.lib.data_helper import get_game_data_filenames
            import os

            # 检查trained文件夹中的文件
            trained_dir = os.path.join(self.config.resource.data_dir, 'trained')
            if os.path.exists(trained_dir):
                trained_files = os.listdir(trained_dir)
                if trained_files:
                    logger.info(f"[第{self.current_iteration}轮] 已训练文件已移动到 {trained_dir}，共 {len(trained_files)} 个文件")

            # 检查当前训练文件数量
            current_files = get_game_data_filenames(self.config.resource)
            logger.info(f"[第{self.current_iteration}轮] 当前可用训练文件: {len(current_files)} 个")

            # 如果文件数量过多，保留最新的文件
            max_files = self.config.play_data.max_file_num
            if len(current_files) > max_files:
                files_to_remove = current_files[:-max_files]
                for file_path in files_to_remove:
                    try:
                        os.remove(file_path)
                        logger.debug(f"删除旧训练文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"删除文件失败 {file_path}: {e}")
                logger.info(f"[第{self.current_iteration}轮] 清理了 {len(files_to_remove)} 个旧训练文件")

        except Exception as e:
            logger.error(f"[第{self.current_iteration}轮] 清理训练数据失败: {e}")

    def _run_evaluation(self):
        """运行模型评估"""
        try:
            logger.info(f"[第{self.current_iteration}轮] 步骤3: 开始评估模型性能...")
            step_start_time = time.time()

            # 设置评估配置
            self.config.eval.update_play_config(self.config.play)

            # 运行评估
            evaluator.start(self.config)

            step_time = time.time() - step_start_time
            logger.info(f"[第{self.current_iteration}轮] 模型评估完成，耗时: {step_time:.2f}秒")
            return True

        except Exception as e:
            logger.error(f"[第{self.current_iteration}轮] 模型评估失败: {e}")
            # 评估失败不应该停止整个进化过程
            logger.warning("评估失败，但继续进化过程")
            return True

    def _cleanup(self):
        """清理资源"""
        end_time = datetime.now()
        total_time = end_time - self.start_time if self.start_time else None

        logger.info("\n" + "=" * 60)
        logger.info("自我进化训练结束")
        logger.info(f"总迭代次数: {self.current_iteration}")
        logger.info(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else '未知'}")
        logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"总耗时: {total_time if total_time else '未知'}")
        if total_time and self.current_iteration > 0:
            logger.info(f"平均每轮耗时: {total_time / self.current_iteration}")
        logger.info("=" * 60)


def start(config: Config, max_iterations=0, skip_eval=False, use_gpu=False, force_gpu_opt=False):
    """
    启动自我进化训练

    Args:
        config: 配置对象
        max_iterations: 最大迭代次数，0表示无限循环
        skip_eval: 是否跳过评估步骤
        use_gpu: 是否使用GPU进行self-play（默认混合模式：self用GPU，opt用CPU）
        force_gpu_opt: 是否强制optimization也使用GPU（默认False，总是用CPU）
    """
    worker = EvolutionWorker(config, max_iterations, skip_eval, use_gpu, force_gpu_opt)
    return worker.start()
