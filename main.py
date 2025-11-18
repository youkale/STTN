# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import importlib
import os
import argparse
import copy
import datetime
import random
import sys
import json
import subprocess

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp
from torchvision import transforms

# My libs
from src.core.utils import Stack, ToTorchFormatTensor


parser = argparse.ArgumentParser(description="STTN")
parser.add_argument("-v", "--video", type=str, required=True, help="输入视频路径")
parser.add_argument("-m", "--mask",   type=str, help="mask图片目录路径（与--regions二选一）")
parser.add_argument("-c", "--ckpt",   type=str, required=True, help="模型checkpoint路径")
parser.add_argument("--model",   type=str, default='sttn', help="模型名称")
parser.add_argument("--regions", type=str, help="需要擦除的区域坐标，格式：[[left,bottom,right,top],...]，坐标为相对比例(0-1)，原点在左下角")
parser.add_argument("--output", type=str, help="输出视频路径（默认：{mask}_result.mp4）")
parser.add_argument("--resolution", type=str, help="处理分辨率，格式：WxH (如 1280x720)，默认自动检测")
parser.add_argument("--scale", type=float, default=1.0, help="分辨率缩放比例 (0-1]，默认1.0保持原分辨率")
parser.add_argument("--short-side", type=int, choices=[270, 360, 480, 540, 720, 1080],
                    help="等比缩放到指定短边尺寸，保持宽高比 (270/360/480/540/720/1080)")
parser.add_argument("--crop", action="store_true",
                    help="裁剪模式：仅处理区域周围部分，大幅降低显存占用（需配合--regions使用）")
parser.add_argument("--crop-padding", type=int, default=32,
                    help="裁剪时的边界padding（像素），避免边界artifacts，默认32")
args = parser.parse_args()


# 默认值，将在 get_video_info 中更新
w, h = 432, 240
ref_length = 10
neighbor_stride = 5
default_fps = 24

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])


def get_video_info_ffprobe(video_path):
    """
    使用 ffprobe 获取视频信息

    返回: (width, height, fps) 或 None
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate',
            '-of', 'json',
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            data = json.loads(result.stdout)
            if 'streams' in data and len(data['streams']) > 0:
                stream = data['streams'][0]
                width = stream.get('width')
                height = stream.get('height')

                # 解析帧率 (格式可能是 "30/1" 或 "30000/1001")
                fps_str = stream.get('r_frame_rate', '24/1')
                if '/' in fps_str:
                    num, den = map(int, fps_str.split('/'))
                    fps = num / den if den != 0 else 24
                else:
                    fps = float(fps_str)

                return width, height, fps
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️  ffprobe 获取视频信息失败: {e}")

    return None


def get_video_info_opencv(video_path):
    """
    使用 OpenCV 获取视频信息（备用方案）

    返回: (width, height, fps)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    # 如果 fps 无效，使用默认值
    if fps <= 0 or fps > 120:
        fps = 24

    return width, height, fps


def setup_resolution(video_path, resolution_arg=None, scale=1.0, short_side=None):
    """
    设置处理分辨率

    参数:
        video_path: 视频路径
        resolution_arg: 用户指定的分辨率字符串 (如 "1280x720")
        scale: 缩放比例
        short_side: 短边目标尺寸（等比缩放）

    返回: (width, height, fps)
    """
    global w, h, default_fps

    # 1. 如果用户指定了分辨率，直接使用
    if resolution_arg:
        try:
            w, h = map(int, resolution_arg.lower().split('x'))
            print(f"✓ 使用指定分辨率: {w}x{h}")
        except ValueError:
            print(f"⚠️  无效的分辨率格式: {resolution_arg}，将自动检测")
            resolution_arg = None

    # 2. 获取视频信息
    if not resolution_arg:
        # 优先使用 ffprobe
        video_info = get_video_info_ffprobe(video_path)

        # 如果 ffprobe 失败，使用 OpenCV
        if video_info is None:
            print("⚠️  ffprobe 不可用，使用 OpenCV 获取视频信息")
            video_info = get_video_info_opencv(video_path)

        orig_w, orig_h, fps = video_info

        # 应用短边缩放（优先级高于 scale）
        if short_side:
            current_short = min(orig_w, orig_h)
            current_long = max(orig_w, orig_h)

            # 计算缩放比例
            scale_ratio = short_side / current_short

            # 应用缩放
            if orig_w < orig_h:  # 竖屏
                w = short_side
                h = int(current_long * scale_ratio)
            else:  # 横屏
                h = short_side
                w = int(current_long * scale_ratio)

            print(f"✓ 原始分辨率: {orig_w}x{orig_h}")
            print(f"✓ 等比缩放到短边 {short_side}: {w}x{h}")
        # 应用缩放比例
        elif scale != 1.0:
            if scale <= 0 or scale > 1:
                print(f"⚠️  无效的缩放比例 {scale}，使用 1.0")
                scale = 1.0
            w = int(orig_w * scale)
            h = int(orig_h * scale)
            print(f"✓ 原始分辨率: {orig_w}x{orig_h}")
            print(f"✓ 缩放比例: {scale}")
            print(f"✓ 处理分辨率: {w}x{h}")
        else:
            w = orig_w
            h = orig_h
            print(f"✓ 自动检测分辨率: {w}x{h}")

        # 更新帧率
        default_fps = int(fps)
        print(f"✓ 检测到帧率: {default_fps} fps")

    # 3. 确保分辨率是偶数（视频编码要求）
    w = w if w % 2 == 0 else w + 1
    h = h if h % 2 == 0 else h + 1

    # 4. 显示显存估算
    estimate_memory(w, h)

    return w, h, default_fps


def estimate_memory(width, height):
    """
    估算所需显存
    """
    pixels = width * height
    # 粗略估算：每百万像素约需 6GB 显存
    estimated_gb = (pixels / 1_000_000) * 6

    print(f"📊 分辨率信息:")
    print(f"   像素数: {pixels:,} ({pixels/1_000_000:.2f}M)")
    print(f"   估算显存: ~{estimated_gb:.1f}GB")

    # 检测实际可用显存
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU 总显存: {gpu_memory_gb:.1f}GB")

        if estimated_gb > gpu_memory_gb * 0.8:  # 使用超过 80% 显存
            print(f"\n   ❌ 错误: 估算显存 ({estimated_gb:.1f}GB) 超过 GPU 容量 ({gpu_memory_gb:.1f}GB)")
            print(f"   必须降低分辨率！")
            print(f"\n   推荐配置 (GPU {gpu_memory_gb:.0f}GB):")

            short_side = min(width, height)
            if gpu_memory_gb < 6:
                print(f"   --short-side 270  (估算 ~0.8GB)")
                print(f"   --short-side 360  (估算 ~1.4GB)")
            elif gpu_memory_gb < 8:
                print(f"   --short-side 360  (估算 ~1.4GB)")
                print(f"   --short-side 480  (估算 ~2.5GB)")
            elif gpu_memory_gb < 12:
                print(f"   --short-side 480  (估算 ~2.5GB)")
                print(f"   --short-side 540  (估算 ~3.1GB)")
            else:
                print(f"   --short-side 540  (估算 ~3.1GB)")
                print(f"   --short-side 720  (估算 ~5.5GB)")

            print(f"\n   示例命令:")
            print(f"   python main.py -v video.mp4 -c model.pth --regions '...' --short-side 360")
            print()
            import sys
            sys.exit(1)
        elif estimated_gb > gpu_memory_gb * 0.6:  # 使用超过 60% 显存
            print(f"   ⚠️  警告: 显存使用率可能较高，建议降低分辨率")

    if estimated_gb > 8:
        print(f"   💡 建议:")
        print(f"   方式1: --scale 0.5 (降低到 {width//2}x{height//2})")
        # 推荐短边尺寸
        short_side = min(width, height)
        if short_side > 540:
            print(f"   方式2: --short-side 540 (推荐用于显存 4-6GB)")
        if short_side > 720:
            print(f"   方式3: --short-side 720 (推荐用于显存 6-8GB)")


# sample reference frames from the whole video
def get_ref_index(neighbor_ids, length):
    ref_index = []
    for i in range(0, length, ref_length):
        if not i in neighbor_ids:
            ref_index.append(i)
    return ref_index


# read frame-wise masks
def read_mask(mpath, video_length=None):
    """
    读取mask图片

    参数:
        mpath: mask目录路径
        video_length: 视频总帧数，如果提供且目录只有一张图片，则复用该图片

    返回:
        masks: mask图片列表
    """
    masks = []
    mnames = [f for f in os.listdir(mpath) if f.endswith('.png') or f.endswith('.jpg')]
    mnames.sort()

    # 如果只有一张mask图片，且指定了视频长度，则复用这张图片
    if len(mnames) == 1 and video_length is not None:
        print(f"检测到单张mask，将复用于所有 {video_length} 帧")
        m = Image.open(os.path.join(mpath, mnames[0]))
        m = m.resize((w, h), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
        mask_img = Image.fromarray(m*255)
        # 复用同一张图片
        masks = [mask_img] * video_length
    else:
        # 逐帧读取
        for m in mnames:
            m = Image.open(os.path.join(mpath, m))
            m = m.resize((w, h), Image.NEAREST)
            m = np.array(m.convert('L'))
            m = np.array(m > 0).astype(np.uint8)
            m = cv2.dilate(m, cv2.getStructuringElement(
                cv2.MORPH_CROSS, (3, 3)), iterations=4)
            masks.append(Image.fromarray(m*255))

    return masks


#  read frames from video
def read_frame_from_videos(vname):
    frames = []
    vidcap = cv2.VideoCapture(vname)
    success, image = vidcap.read()
    count = 0
    while success:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image.resize((w,h)))
        success, image = vidcap.read()
        count += 1
    return frames


def calculate_crop_region(regions, video_width, video_height, padding=32):
    """
    根据regions计算裁剪区域（加padding）

    参数:
        regions: 区域列表 [[left,bottom,right,top],...]，相对坐标(0-1)
        video_width, video_height: 原视频尺寸
        padding: 边界padding（像素）

    返回:
        (x, y, crop_w, crop_h): 裁剪参数（像素坐标）
        (min_left, min_bottom, max_right, max_top): 合并后的区域边界（相对坐标）
    """
    # 合并所有区域的边界
    all_lefts = [r[0] for r in regions]
    all_bottoms = [r[1] for r in regions]
    all_rights = [r[2] for r in regions]
    all_tops = [r[3] for r in regions]

    min_left = min(all_lefts)
    min_bottom = min(all_bottoms)
    max_right = max(all_rights)
    max_top = max(all_tops)

    # 转换为像素坐标
    x1_pixel = int(min_left * video_width)
    x2_pixel = int(max_right * video_width)

    # Y坐标：左下角原点 -> 左上角原点
    y1_pixel = int((1 - max_top) * video_height)
    y2_pixel = int((1 - min_bottom) * video_height)

    # 添加padding
    x1_pixel = max(0, x1_pixel - padding)
    y1_pixel = max(0, y1_pixel - padding)
    x2_pixel = min(video_width, x2_pixel + padding)
    y2_pixel = min(video_height, y2_pixel + padding)

    # 计算裁剪区域尺寸
    crop_w = x2_pixel - x1_pixel
    crop_h = y2_pixel - y1_pixel

    # 确保尺寸是偶数
    crop_w = crop_w if crop_w % 2 == 0 else crop_w - 1
    crop_h = crop_h if crop_h % 2 == 0 else crop_h - 1

    return (x1_pixel, y1_pixel, crop_w, crop_h), (min_left, min_bottom, max_right, max_top)


def transform_regions_to_crop_space(regions, crop_region_bounds, video_width, video_height, crop_w, crop_h):
    """
    将全局坐标的regions转换为裁剪空间的相对坐标

    参数:
        regions: 原始区域列表（全局相对坐标）
        crop_region_bounds: (min_left, min_bottom, max_right, max_top) 裁剪区域边界（全局相对坐标）
        video_width, video_height: 原视频尺寸
        crop_w, crop_h: 裁剪后的尺寸

    返回:
        transformed_regions: 转换后的区域列表（裁剪空间相对坐标）
    """
    min_left, min_bottom, max_right, max_top = crop_region_bounds

    # 裁剪区域的尺寸（相对坐标）
    crop_rel_width = max_right - min_left
    crop_rel_height = max_top - min_bottom

    transformed_regions = []
    for region in regions:
        left, bottom, right, top = region

        # 转换到裁剪空间（相对于裁剪区域的左下角）
        new_left = (left - min_left) / crop_rel_width
        new_right = (right - min_left) / crop_rel_width
        new_bottom = (bottom - min_bottom) / crop_rel_height
        new_top = (top - min_bottom) / crop_rel_height

        # 确保在[0,1]范围内
        new_left = max(0, min(1, new_left))
        new_right = max(0, min(1, new_right))
        new_bottom = max(0, min(1, new_bottom))
        new_top = max(0, min(1, new_top))

        transformed_regions.append([new_left, new_bottom, new_right, new_top])

    return transformed_regions


def crop_video_ffmpeg(input_video, output_video, x, y, width, height):
    """
    使用ffmpeg裁剪视频

    参数:
        input_video: 输入视频路径
        output_video: 输出视频路径
        x, y: 裁剪起始坐标
        width, height: 裁剪尺寸
    """
    cmd = [
        'ffmpeg', '-y', '-i', input_video,
        '-filter:v', f'crop={width}:{height}:{x}:{y}',
        '-c:a', 'copy',  # 音频直接复制
        output_video
    ]

    print(f"裁剪命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg裁剪失败: {result.stderr}")

    return output_video


def merge_video_ffmpeg(original_video, inpainted_video, output_video, x, y):
    """
    使用ffmpeg将修复后的视频合并回原视频

    参数:
        original_video: 原始视频路径
        inpainted_video: 修复后的裁剪视频路径
        output_video: 输出视频路径
        x, y: overlay位置
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', original_video,
        '-i', inpainted_video,
        '-filter_complex', f'[0:v][1:v]overlay={x}:{y}',
        '-c:a', 'copy',  # 保留原视频音频
        output_video
    ]

    print(f"合并命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg合并失败: {result.stderr}")

    return output_video


def generate_masks_from_regions(video_path, regions, num_frames=None):
    """
    根据区域坐标生成mask图片，保存到output目录

    参数:
        video_path: 视频路径
        regions: 区域列表，格式为 [[left, bottom, right, top], ...]
                 坐标为相对比例，范围 0-1，以左下角为原点
                 left: 左边距离 (0=最左, 1=最右)
                 bottom: 底边距离 (0=最底, 1=最顶)
                 right: 右边距离 (0=最左, 1=最右)
                 top: 顶边距离 (0=最底, 1=最顶)
        num_frames: 帧数（如果为None则从视频中获取）

    返回:
        mask_dir: mask目录路径
    """
    import datetime

    # 获取视频信息
    vidcap = cv2.VideoCapture(video_path)

    if num_frames is None:
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 获取原始视频尺寸
    orig_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vidcap.release()

    print(f"视频原始尺寸: {orig_width}x{orig_height}")
    print(f"处理尺寸: {w}x{h}")
    print(f"视频总帧数: {num_frames}")
    print(f"区域数量: {len(regions)}")

    # 创建output目录下的mask子目录
    output_base = "output"
    os.makedirs(output_base, exist_ok=True)

    # 使用时间戳创建唯一的mask目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    mask_dir = os.path.join(output_base, f"masks_{video_name}_{timestamp}")
    os.makedirs(mask_dir, exist_ok=True)
    print(f"Mask目录: {mask_dir}")

    # 创建黑色背景 (使用处理尺寸)
    mask = np.zeros((h, w), dtype=np.uint8)

    # 在指定区域绘制白色
    for region in regions:
        left, bottom, right, top = region

        # 验证坐标范围
        if not (0 <= left < right <= 1 and 0 <= bottom < top <= 1):
            raise ValueError(f"无效的区域坐标: {region}，坐标必须在[0,1]范围内且left<right, bottom<top")

        # 转换相对坐标到实际像素坐标（基于处理尺寸）
        # 注意：以左下角为原点，需要转换Y坐标
        x1 = int(left * w)
        x2 = int(right * w)

        # Y坐标转换：左下角原点 -> 左上角原点（图像坐标系）
        # bottom=0 表示最底部，在图像坐标系中是 h
        # top=1 表示最顶部，在图像坐标系中是 0
        y1 = int((1 - top) * h)      # 顶部在图像坐标系中的位置
        y2 = int((1 - bottom) * h)   # 底部在图像坐标系中的位置

        # 确保坐标在有效范围内
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        # 绘制白色区域
        mask[y1:y2, x1:x2] = 255

        print(f"  区域 [left={left}, bottom={bottom}, right={right}, top={top}]")
        print(f"    -> 像素坐标: x=[{x1},{x2}], y=[{y1},{y2}] (图像坐标系)")

    # 应用膨胀操作（与read_mask保持一致）
    mask = cv2.dilate(mask, cv2.getStructuringElement(
        cv2.MORPH_CROSS, (3, 3)), iterations=4)

    # 只保存一张mask图片
    mask_path = os.path.join(mask_dir, 'mask.png')
    Image.fromarray(mask).save(mask_path)

    print(f"✓ 成功生成单张mask (将复用于所有 {num_frames} 帧)")
    print(f"✓ 节省空间: ~{(num_frames - 1) * mask.nbytes / 1024 / 1024:.1f}MB")
    return mask_dir


def main_worker():
    # 验证参数
    if args.mask is None and args.regions is None:
        raise ValueError("必须指定 --mask 或 --regions 参数之一")
    if args.mask is not None and args.regions is not None:
        raise ValueError("--mask 和 --regions 参数不能同时使用")
    if args.crop and args.regions is None:
        raise ValueError("--crop 模式必须配合 --regions 使用")

    # 裁剪模式变量
    crop_mode = args.crop and args.regions is not None
    cropped_video = None
    crop_x, crop_y = 0, 0
    original_video = args.video

    # 裁剪模式处理
    if crop_mode:
        print("\n" + "="*60)
        print("🎯 裁剪模式：仅处理擦除区域周围部分")
        print("="*60)

        regions = json.loads(args.regions)

        # 获取原视频信息
        video_info = get_video_info_ffprobe(original_video)
        if video_info is None:
            video_info = get_video_info_opencv(original_video)
        orig_w, orig_h, orig_fps = video_info

        print(f"原视频尺寸: {orig_w}x{orig_h}")
        print(f"擦除区域: {regions}")
        print(f"边界padding: {args.crop_padding}px")

        # 计算裁剪区域
        (crop_x, crop_y, crop_w, crop_h), crop_bounds = calculate_crop_region(
            regions, orig_w, orig_h, padding=args.crop_padding
        )

        print(f"\n裁剪参数:")
        print(f"  位置: x={crop_x}, y={crop_y}")
        print(f"  尺寸: {crop_w}x{crop_h}")
        print(f"  显存降低: ~{(orig_w*orig_h)/(crop_w*crop_h):.1f}x")

        # 转换regions到裁剪空间
        transformed_regions = transform_regions_to_crop_space(
            regions, crop_bounds, orig_w, orig_h, crop_w, crop_h
        )
        print(f"  转换后区域: {transformed_regions}")

        # 裁剪视频
        print("\n裁剪视频...")
        output_base = "output"
        os.makedirs(output_base, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cropped_video = os.path.join(output_base, f"cropped_{timestamp}.mp4")

        crop_video_ffmpeg(original_video, cropped_video, crop_x, crop_y, crop_w, crop_h)
        print(f"✓ 裁剪完成: {cropped_video}")

        # 更新处理目标
        video_to_process = cropped_video
        regions_to_use = transformed_regions

        print("="*60 + "\n")
    else:
        video_to_process = original_video
        regions_to_use = json.loads(args.regions) if args.regions else None

    # 设置处理分辨率
    print("\n" + "="*60)
    print("视频信息检测")
    print("="*60)
    setup_resolution(video_to_process, args.resolution, args.scale, args.short_side)
    print("="*60 + "\n")

    # 先生成或准备 mask（在加载模型之前）
    temp_mask_dir = None  # 用于跟踪临时目录
    if args.regions is not None:
        # 从区域坐标生成mask
        print("="*60)
        print("生成 Mask")
        print("="*60)
        print(f"区域坐标: {regions_to_use}")

        # 获取视频帧数
        vidcap = cv2.VideoCapture(video_to_process)
        video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        vidcap.release()

        temp_mask_dir = generate_masks_from_regions(video_to_process, regions_to_use, video_length)
        print("="*60 + "\n")
    else:
        # 验证mask目录存在
        print("="*60)
        print("验证 Mask 目录")
        print("="*60)
        if not os.path.exists(args.mask):
            raise ValueError(f"Mask目录不存在: {args.mask}")
        mask_files = [f for f in os.listdir(args.mask) if f.endswith('.png') or f.endswith('.jpg')]
        if not mask_files:
            raise ValueError(f"Mask目录中没有图片文件: {args.mask}")
        print(f"✓ Mask目录: {args.mask}")
        print(f"✓ 找到 {len(mask_files)} 个mask文件")
        print("="*60 + "\n")

    # 加载模型
    print("="*60)
    print("加载模型")
    print("="*60)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ 使用设备: {device} (GPU: {torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"✓ 使用设备: CPU")

    net = importlib.import_module('src.model.' + args.model)
    model = net.InpaintGenerator().to(device)
    model_path = args.ckpt
    data = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(data['netG'])
    print(f'✓ 模型加载完成: {args.ckpt}')
    model.eval()
    print("="*60 + "\n")

    # 加载视频帧
    print("="*60)
    print("加载视频帧")
    print("="*60)
    frames = read_frame_from_videos(video_to_process)
    video_length = len(frames)
    print(f"✓ 加载 {video_length} 帧")
    feats = _to_tensors(frames).unsqueeze(0)*2-1
    frames = [np.array(f).astype(np.uint8) for f in frames]
    print("="*60 + "\n")

    # 加载mask
    print("="*60)
    print("加载 Mask")
    print("="*60)
    if temp_mask_dir is not None:
        masks = read_mask(temp_mask_dir, video_length)
    else:
        masks = read_mask(args.mask, video_length)
    print(f"✓ 加载 {len(masks)} 个mask")
    print("="*60 + "\n")

    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
    masks = _to_tensors(masks).unsqueeze(0)
    feats, masks = feats.to(device), masks.to(device)
    comp_frames = [None]*video_length

    # 编码特征
    print("="*60)
    print("编码视频特征")
    print("="*60)
    with torch.no_grad():
        feats = model.encoder((feats*(1-masks).float()).view(video_length, 3, h, w))
        _, c, feat_h, feat_w = feats.size()
        feats = feats.view(1, video_length, c, feat_h, feat_w)
    print(f'✓ 特征编码完成: {video_length} 帧')
    print("="*60 + "\n")

    # 修复视频
    print("="*60)
    print("开始修复视频")
    print("="*60)
    total_steps = (video_length + neighbor_stride - 1) // neighbor_stride
    for step, f in enumerate(range(0, video_length, neighbor_stride), 1):
        neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
        ref_ids = get_ref_index(neighbor_ids, video_length)

        print(f"处理进度: {step}/{total_steps} (帧 {f}-{min(f+neighbor_stride, video_length)}/{video_length})", end='\r')

        with torch.no_grad():
            pred_feat = model.infer(
                feats[0, neighbor_ids+ref_ids, :, :, :], masks[0, neighbor_ids+ref_ids, :, :, :])
            pred_img = torch.tanh(model.decoder(
                pred_feat[:len(neighbor_ids), :, :, :])).detach()
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(
                    np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32)*0.5 + img.astype(np.float32)*0.5
    print()  # 换行
    print(f'✓ 视频修复完成')
    print("="*60 + "\n")

    # 确定输出路径
    if args.output:
        final_output_path = args.output
    else:
        # 默认输出到output目录
        output_base = "output"
        os.makedirs(output_base, exist_ok=True)

        if args.mask:
            # 使用mask目录名称
            mask_basename = os.path.basename(args.mask.rstrip('/'))
            final_output_path = os.path.join(output_base, f"{mask_basename}_result.mp4")
        else:
            # 使用视频名称
            video_name = os.path.splitext(os.path.basename(args.video))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output_path = os.path.join(output_base, f"{video_name}_inpainted_{timestamp}.mp4")

    # 保存修复后的视频
    print("="*60)
    print("保存输出视频")
    print("="*60)

    # 裁剪模式：先保存裁剪区域的修复视频
    if crop_mode:
        inpainted_crop_path = final_output_path.replace('.mp4', '_crop.mp4')
        print(f"中间文件: {inpainted_crop_path}")
    else:
        inpainted_crop_path = final_output_path
        print(f"输出路径: {final_output_path}")

    writer = cv2.VideoWriter(inpainted_crop_path, cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (w, h))
    for f in range(video_length):
        comp = np.array(comp_frames[f]).astype(
            np.uint8)*binary_masks[f] + frames[f] * (1-binary_masks[f])
        writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
        print(f"写入帧: {f+1}/{video_length}", end='\r')
    writer.release()
    print()  # 换行
    print(f'✓ 裁剪区域修复完成: {inpainted_crop_path}')
    print("="*60 + "\n")

    # 裁剪模式：合并回原视频
    if crop_mode:
        print("="*60)
        print("合并到原视频")
        print("="*60)
        print(f"原视频: {original_video}")
        print(f"修复区域: {inpainted_crop_path}")
        print(f"最终输出: {final_output_path}")
        print(f"Overlay位置: x={crop_x}, y={crop_y}")

        merge_video_ffmpeg(original_video, inpainted_crop_path, final_output_path, crop_x, crop_y)

        print(f'✓ 合并完成: {final_output_path}')

        # 清理临时文件
        print("\n清理临时文件...")
        try:
            if cropped_video and os.path.exists(cropped_video):
                os.remove(cropped_video)
                print(f"✓ 删除: {cropped_video}")
            if os.path.exists(inpainted_crop_path):
                os.remove(inpainted_crop_path)
                print(f"✓ 删除: {inpainted_crop_path}")
        except Exception as e:
            print(f"⚠️  清理临时文件失败: {e}")

        print("="*60 + "\n")

    # 保留mask目录供用户查看
    if temp_mask_dir is not None:
        print(f'✓ Mask文件保存在: {temp_mask_dir}')

    print("="*60)
    print("🎉 全部完成！")
    print(f"📹 最终输出: {final_output_path}")
    print("="*60)



if __name__ == '__main__':
    main_worker()
