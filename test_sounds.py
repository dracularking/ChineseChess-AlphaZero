#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试音效功能
"""

import os
import sys
import pygame
import time

def test_sounds():
    """测试音效播放"""
    try:
        # 初始化pygame mixer
        pygame.mixer.init()
        print("pygame.mixer 初始化成功")
        
        # 音效文件路径
        sounds_dir = os.path.join(os.path.dirname(__file__), "cchess_alphazero", "play_games", "sounds")
        click_sound_path = os.path.join(sounds_dir, "click.wav")
        drop_sound_path = os.path.join(sounds_dir, "drop.wav")
        
        print(f"音效目录: {sounds_dir}")
        print(f"点击音效文件: {click_sound_path}")
        print(f"落子音效文件: {drop_sound_path}")
        
        # 检查文件是否存在
        if not os.path.exists(click_sound_path):
            print(f"错误: 点击音效文件不存在: {click_sound_path}")
            return False
            
        if not os.path.exists(drop_sound_path):
            print(f"错误: 落子音效文件不存在: {drop_sound_path}")
            return False
            
        # 加载音效
        click_sound = pygame.mixer.Sound(click_sound_path)
        drop_sound = pygame.mixer.Sound(drop_sound_path)
        
        click_sound.set_volume(0.5)
        drop_sound.set_volume(0.7)
        
        print("音效文件加载成功")
        
        # 播放测试
        print("\n播放点击音效...")
        click_sound.play()
        time.sleep(1)
        
        print("播放落子音效...")
        drop_sound.play()
        time.sleep(1)
        
        print("\n音效测试完成！")
        return True
        
    except Exception as e:
        print(f"音效测试失败: {e}")
        return False
    finally:
        try:
            pygame.mixer.quit()
        except:
            pass

if __name__ == "__main__":
    print("开始测试音效功能...")
    success = test_sounds()
    if success:
        print("\n✓ 音效功能测试通过")
    else:
        print("\n✗ 音效功能测试失败")
        sys.exit(1)