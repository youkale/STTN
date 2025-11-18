#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用mask图片测试脚本
用于验证mask生成和擦除效果
"""

import subprocess
import os
import sys
import cv2
import numpy as np
from PIL import Image


def create_test_video_and_mask():
    """创建测试视频和mask"""

    INPUT_VIDEO = "examples/subtitle_4.mp4"
    OUTPUT_DIR = "test_output"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*60)
    print("准备测试数据")
    print("="*60)

    # 步骤1: 截取前5秒
    print("\n1. 截取前5秒视频...")
    test_video = f"{OUTPUT_DIR}/test_5sec.mp4"
    cmd = ["ffmpeg", "-y", "-i", INPUT_VIDEO, "-t", "5", "-c", "copy", test_video]
    subprocess.run(cmd, capture_output=True)
    print(f"✓ 保存: {test_video}")

    # 步骤2: 获取视频信息
    cap = cv2.VideoCapture(test_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"\n2. 视频信息:")
    print(f"   尺寸: {width}x{height}")

    # 步骤3: 调整分辨率到432x240的倍数
    # 假设要处理底部20%区域
    target_height = 240  # 432x240的高度
    target_width = 432 * (width // 432)  # 对齐到432的倍数
    if target_width > width:
        target_width = 432

    print(f"\n3. 调整分辨率到: {target_width}x{target_height}")

    resized_video = f"{OUTPUT_DIR}/test_resized.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", test_video,
        "-vf", f"scale={target_width}:{target_height}",
        "-c:a", "copy",
        resized_video
    ]
    subprocess.run(cmd, capture_output=True)
    print(f"✓ 保存: {resized_video}")

    # 步骤4: 生成mask（底部20%区域）
    print(f"\n4. 生成mask图片...")
    mask_dir = f"{OUTPUT_DIR}/masks"
    os.makedirs(mask_dir, exist_ok=True)

    # 创建mask：底部20%为白色，其余为黑色
    mask = np.zeros((target_height, target_width), dtype=np.uint8)

    # 底部20%区域
    bottom_20_start = int(target_height * 0.8)
    mask[bottom_20_start:, :] = 255

    # 应用膨胀
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)

    mask_path = f"{mask_dir}/mask.png"
    Image.fromarray(mask).save(mask_path)

    print(f"✓ 保存: {mask_path}")
    print(f"   尺寸: {target_width}x{target_height}")
    print(f"   白色像素: {np.sum(mask == 255)} / {mask.size}")

    return resized_video, mask_dir


def test_inpainting(video_path, mask_dir, checkpoint):
    """测试擦除"""

    print("\n" + "="*60)
    print("开始擦除测试")
    print("="*60)

    cmd = [
        "python", "main.py",
        "-v", video_path,
        "-m", mask_dir,
        "-c", checkpoint,
        "--output", "test_output/result_with_mask.mp4"
    ]

    print(f"命令: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✓ 擦除完成!")
        print(f"\n查看结果:")
        print(f"  原视频: open {video_path}")
        print(f"  Mask:   open {mask_dir}/mask.png")
        print(f"  结果:   open test_output/result_with_mask.mp4")
    else:
        print("\n❌ 擦除失败!")
        sys.exit(1)


def main():
    """主流程"""

    checkpoint = "checkpoints/sttn.pth"

    if not os.path.exists(checkpoint):
        print(f"❌ 模型文件不存在: {checkpoint}")
        print("请先下载模型文件")
        sys.exit(1)

    if not os.path.exists("examples/subtitle_4.mp4"):
        print("❌ 测试视频不存在: examples/subtitle_4.mp4")
        sys.exit(1)

    print("\n" + "="*60)
    print("🧪 Mask图片测试流程")
    print("="*60)

    # 准备测试数据
    video_path, mask_dir = create_test_video_and_mask()

    # 运行测试
    test_inpainting(video_path, mask_dir, checkpoint)

    print("\n" + "="*60)
    print("🎉 测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
