#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成象棋游戏音效文件
"""

import numpy as np
import wave
import struct

def generate_click_sound(filename, duration=0.1, sample_rate=44100):
    """
    生成点击棋子的音效 - 短促的高频音
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # 生成高频音调 (800Hz) 配合快速衰减
    frequency = 800
    wave_data = np.sin(2 * np.pi * frequency * t)
    
    # 添加快速衰减效果
    envelope = np.exp(-t * 20)
    wave_data = wave_data * envelope
    
    # 归一化并转换为16位整数
    wave_data = (wave_data * 32767).astype(np.int16)
    
    # 保存为WAV文件
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(wave_data.tobytes())

def generate_drop_sound(filename, duration=0.3, sample_rate=44100):
    """
    生成棋子落下的音效 - 低频音配合回响
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # 生成低频音调 (200Hz) 模拟棋子落在棋盘上的声音
    frequency = 200
    wave_data = np.sin(2 * np.pi * frequency * t)
    
    # 添加一些高频成分模拟撞击声
    high_freq = 1200
    high_component = 0.3 * np.sin(2 * np.pi * high_freq * t)
    
    # 高频成分快速衰减
    high_envelope = np.exp(-t * 15)
    high_component = high_component * high_envelope
    
    # 低频成分慢速衰减
    low_envelope = np.exp(-t * 3)
    wave_data = wave_data * low_envelope
    
    # 合并高低频成分
    final_wave = wave_data + high_component
    
    # 归一化并转换为16位整数
    final_wave = final_wave / np.max(np.abs(final_wave))  # 归一化
    final_wave = (final_wave * 32767 * 0.8).astype(np.int16)  # 降低音量
    
    # 保存为WAV文件
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(final_wave.tobytes())

if __name__ == "__main__":
    import os
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sounds_dir = os.path.join(current_dir, "sounds")
    
    # 确保sounds目录存在
    os.makedirs(sounds_dir, exist_ok=True)
    
    # 生成音效文件
    click_sound_path = os.path.join(sounds_dir, "click.wav")
    drop_sound_path = os.path.join(sounds_dir, "drop.wav")
    
    print("正在生成点击音效...")
    generate_click_sound(click_sound_path)
    print(f"点击音效已保存到: {click_sound_path}")
    
    print("正在生成落子音效...")
    generate_drop_sound(drop_sound_path)
    print(f"落子音效已保存到: {drop_sound_path}")
    
    print("音效文件生成完成！")