from collections import defaultdict
from logging import getLogger
import signal
import sys
import os

# 设置Windows终端编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.environment.chessboard import Chessboard
from cchess_alphazero.environment.chessman import *
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_move
from cchess_alphazero.lib.model_helper import load_best_model_weight
from cchess_alphazero.lib.tf_util import set_session_config

logger = getLogger(__name__)

# 全局标志，用于优雅退出
_should_exit = False

def signal_handler(signum, frame):
    """处理退出信号"""
    global _should_exit
    print("\n\n收到退出信号，正在清理资源...")
    _should_exit = True
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Windows 支持 Ctrl+Break
if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, signal_handler)

def start(config: Config, human_move_first=True):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    play = PlayWithHuman(config)
    play.start(human_move_first)

class PlayWithHuman:
    def __init__(self, config: Config):
        self.config = config
        self.env = CChessEnv()
        self.model = None
        self.pipe = None
        self.ai = None
        self.chessmans = None
        self.human_move_first = True

    def load_model(self):
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

    def start(self, human_first=True):
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                              enable_resign=True, debugging=False)
        self.human_move_first = human_first

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        self.env.board.print_to_cl()
        print("\n=== 游戏开始 ===")
        print("输入格式: xy (如: 00 表示左上角)")
        print("按 Ctrl+C 或 Ctrl+Break 退出游戏\n")

        try:
            while not self.env.board.is_end():
                global _should_exit
                if _should_exit:
                    break

                if human_first == self.env.red_to_move:
                    self.env.board.calc_chessmans_moving_list()
                    is_correct_chessman = False
                    is_correct_position = False
                    chessman = None
                    while not is_correct_chessman:
                        if _should_exit:
                            return
                        try:
                            title = "请输入棋子位置 (或 'q' 退出): "
                            input_chessman_pos = input(title).strip().lower()
                            if input_chessman_pos in ('q', 'quit', 'exit'):
                                print("游戏已退出")
                                return
                            if len(input_chessman_pos) < 2:
                                print("输入格式错误，请输入两位数字，如: 00")
                                continue
                            x, y = int(input_chessman_pos[0]), int(input_chessman_pos[1])
                            chessman = self.env.board.chessmans[x][y]
                            if chessman != None and chessman.is_red == self.env.board.is_red_turn:
                                is_correct_chessman = True
                                print(f"当前棋子为{chessman.name_cn}，可以落子的位置有：")
                                for point in chessman.moving_list:
                                    print(f"  ({point.x}, {point.y})")
                            else:
                                print("没有找到此名字的棋子或未轮到此方走子")
                        except (ValueError, IndexError) as e:
                            print(f"输入错误: {e}，请输入两位数字，如: 00")
                        except KeyboardInterrupt:
                            print("\n游戏已中断")
                            return

                    while not is_correct_position:
                        if _should_exit:
                            return
                        try:
                            title = "请输入落子的位置 (或 'q' 退出): "
                            input_chessman_pos = input(title).strip().lower()
                            if input_chessman_pos in ('q', 'quit', 'exit'):
                                print("游戏已退出")
                                return
                            if len(input_chessman_pos) < 2:
                                print("输入格式错误，请输入两位数字，如: 00")
                                continue
                            x, y = int(input_chessman_pos[0]), int(input_chessman_pos[1])
                            is_correct_position = chessman.move(x, y)
                            if is_correct_position:
                                self.env.board.print_to_cl()
                                self.env.board.clear_chessmans_moving_list()
                            else:
                                print("无效的移动位置，请重新输入")
                        except (ValueError, IndexError) as e:
                            print(f"输入错误: {e}，请输入两位数字，如: 00")
                        except KeyboardInterrupt:
                            print("\n游戏已中断")
                            return
                else:
                    print("AI 正在思考...")
                    action, policy = self.ai.action(self.env.get_state(), self.env.num_halfmoves)
                    if not self.env.red_to_move:
                        action = flip_move(action)
                    if action is None:
                        print("AI投降了!")
                        break
                    self.env.step(action)
                    print(f"AI选择移动 {action}")
                    self.env.board.print_to_cl()
        except KeyboardInterrupt:
            print("\n\n游戏被用户中断")
        finally:
            self.ai.close()
            if self.env.board.is_end():
                print(f"胜者是 is {self.env.board.winner} !!!")
                self.env.board.print_record()
            else:
                print("游戏未结束，已退出")
