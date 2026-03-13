#!/usr/bin/env python3
"""
Phigros IN 难度谱面批量下载工具

文件操作说明：
- 从 GitHub 仓库 7aGiven/Phigros_Resource 下载资源
- 下载目录: /home/adolph/Downloads/phigros_official/
- 格式: pez 文件 (zip 格式)，包含:
  - {id}.json (谱面)
  - {id}.ogg (音乐)
  - {id}.png (曲绘)
  - info.txt (信息文件)

使用方法:
  python3 download_phigros_in_charts.py
"""

import os
import json
import random
import zipfile
import requests
import time
from pathlib import Path

# 配置
DOWNLOAD_DIR = Path("/home/adolph/Downloads/phigros_official")
SCRIPT_DIR = Path(__file__).parent

# GitHub Raw 资源路径
BASE_URL = "https://raw.githubusercontent.com/7aGiven/Phigros_Resource"
CHART_URL = f"{BASE_URL}/chart"
ILLUSTRATION_URL = f"{BASE_URL}/illustration"
MUSIC_URL = f"{BASE_URL}/music"
INFO_URL = f"{BASE_URL}/info/info.tsv"
DIFFICULTY_URL = f"{BASE_URL}/info/difficulty.tsv"

# 确保下载目录存在
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def generate_random_id() -> str:
    """生成随机数字 ID (像网页下载的那样)"""
    return str(random.randint(1000000000000000, 9999999999999999))


def fetch_text(url: str) -> str:
    """获取文本内容"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"  ✗ 获取失败: {url} - {e}")
        return ""


def fetch_binary(url: str) -> bytes:
    """获取二进制内容"""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"  ✗ 下载失败: {url} - {e}")
        return b""


def parse_info_tsv(content: str) -> dict:
    """解析歌曲信息，返回 song_id -> 歌曲信息的映射"""
    songs = {}
    lines = content.strip().split('\n')
    
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 2:
            song_id = parts[0]
            # 歌曲名在第2列 (索引1)
            song_name = parts[1] if len(parts) > 1 else song_id
            # 曲师在第3列 (索引2)
            composer = parts[2] if len(parts) > 2 else "Unknown"
            songs[song_id] = {
                'name': song_name,
                'composer': composer
            }
    
    return songs


def parse_difficulty_tsv(content: str) -> list:
    """解析难度信息，返回有 IN 难度的歌曲列表"""
    songs = []
    lines = content.strip().split('\n')
    
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 4:
            song_id = parts[0]
            ez = parts[1] if len(parts) > 1 else ""
            hd = parts[2] if len(parts) > 2 else ""
            in_diff = parts[3] if len(parts) > 3 else ""
            at = parts[4] if len(parts) > 4 else ""
            
            # 检查是否有 IN 难度
            if in_diff and in_diff.strip():
                songs.append({
                    'id': song_id,
                    'ez': ez,
                    'hd': hd,
                    'in': in_diff,
                    'at': at
                })
    
    return songs


def get_charter_from_info(song_id: str) -> str:
    """从 info.tsv 获取谱师信息"""
    try:
        content = fetch_text(INFO_URL)
        lines = content.strip().split('\n')
        for line in lines:
            parts = line.split('\t')
            if parts[0] == song_id and len(parts) >= 6:
                # 谱师在第6列 (索引5)，对应 IN 难度
                return parts[5] if len(parts) > 5 else "Unknown"
    except:
        pass
    return "Unknown"


def download_and_create_pez(song: dict, song_info: dict, index: int, total: int) -> bool:
    """下载资源并创建 pez 文件"""
    song_id = song['id']
    in_diff = song['in']
    song_name = song_info.get('name', song_id)
    composer = song_info.get('composer', 'Unknown')
    
    # 使用歌曲名作为 pez 文件名 (清理特殊字符)
    safe_name = "".join(c for c in song_name if c.isalnum() or c in '._- ').strip()
    pez_name = f"{safe_name}_IN.pez"
    pez_path = DOWNLOAD_DIR / pez_name
    
    print(f"[{index}/{total}] 处理: {song_name} (IN {in_diff})")
    
    # 检查是否已存在
    if pez_path.exists():
        print(f"  ⏭ 已存在，跳过")
        return True
    
    # 生成随机 ID
    file_id = generate_random_id()
    
    # 资源 URL
    chart_url = f"{CHART_URL}/{song_id}.0/IN.json"
    illustration_url = f"{ILLUSTRATION_URL}/{song_id}.png"
    music_url = f"{MUSIC_URL}/{song_id}.ogg"
    
    # 下载谱面 (必需)
    chart_data = fetch_binary(chart_url)
    if not chart_data:
        print(f"  ✗ 谱面下载失败: {song_id}")
        return False
    
    # 下载曲绘
    illustration_data = fetch_binary(illustration_url)
    if not illustration_data:
        print(f"  ⚠ 曲绘下载失败")
    
    # 下载音乐
    music_data = fetch_binary(music_url)
    if not music_data:
        print(f"  ⚠ 音乐下载失败")
    
    # 获取谱师信息
    charter = get_charter_from_info(song_id)
    
    # 创建 info.txt 内容
    info_content = f"""#
Name: {song_name}
Path: {file_id}
Song: {file_id}.ogg
Picture: {file_id}.png
Chart: {file_id}.json
Level: IN Lv.{in_diff}
Composer: {composer}
Charter: {charter}
"""
    
    # 创建 pez 文件 (zip 格式)
    try:
        with zipfile.ZipFile(pez_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 添加谱面文件
            zf.writestr(f"{file_id}.json", chart_data)
            
            # 添加曲绘 (如果下载成功)
            if illustration_data:
                zf.writestr(f"{file_id}.png", illustration_data)
            
            # 添加音乐 (如果下载成功)
            if music_data:
                zf.writestr(f"{file_id}.ogg", music_data)
            
            # 添加 info.txt
            zf.writestr("info.txt", info_content)
        
        print(f"  ✓ 成功创建: {pez_name}")
        return True
        
    except Exception as e:
        print(f"  ✗ 创建 pez 失败: {e}")
        # 清理失败的文件
        if pez_path.exists():
            pez_path.unlink()
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("Phigros IN 难度谱面批量下载工具")
    print("=" * 60)
    print(f"下载目录: {DOWNLOAD_DIR}")
    print()
    
    # 获取歌曲信息
    print("正在获取歌曲信息...")
    info_content = fetch_text(INFO_URL)
    if not info_content:
        print("✗ 无法获取歌曲信息")
        return
    
    song_info_dict = parse_info_tsv(info_content)
    print(f"获取到 {len(song_info_dict)} 首歌曲信息")
    
    # 获取难度信息
    print("正在获取难度信息...")
    difficulty_content = fetch_text(DIFFICULTY_URL)
    if not difficulty_content:
        print("✗ 无法获取难度信息")
        return
    
    songs = parse_difficulty_tsv(difficulty_content)
    print(f"找到 {len(songs)} 首有 IN 难度的歌曲")
    print()
    
    # 统计
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # 批量下载
    print("开始下载...")
    print("-" * 60)
    
    for i, song in enumerate(songs, 1):
        song_id = song['id']
        song_info = song_info_dict.get(song_id, {'name': song_id, 'composer': 'Unknown'})
        
        # 检查是否已存在
        safe_name = "".join(c for c in song_info['name'] if c.isalnum() or c in '._- ').strip()
        pez_name = f"{safe_name}_IN.pez"
        pez_path = DOWNLOAD_DIR / pez_name
        
        if pez_path.exists():
            print(f"[{i}/{len(songs)}] {song_info['name']} - 已存在，跳过")
            skip_count += 1
            continue
        
        # 下载并创建 pez
        if download_and_create_pez(song, song_info, i, len(songs)):
            success_count += 1
        else:
            fail_count += 1
        
        # 添加延迟避免请求过快
        time.sleep(0.5)
    
    print("-" * 60)
    print()
    print("=" * 60)
    print("下载完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  跳过(已存在): {skip_count}")
    print(f"  总计: {len(songs)}")
    print(f"输出目录: {DOWNLOAD_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
