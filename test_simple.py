#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的测试脚本
用于快速测试裁剪+分段+擦除工作流程
"""

import subprocess
import os
import sys

def run_cmd(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    print(f"命令: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"❌ 失败: {description}")
        sys.exit(1)

    print(f"✓ 完成: {description}")


def main():
    """主测试流程"""

    # 配置
    INPUT_VIDEO = "examples/subtitle_4.mp4"
    OUTPUT_DIR = "test_output"
    CHECKPOINT = "checkpoints/sttn.pth"

    # 字幕区域（相对坐标，左下角原点）
    # left, bottom, right, top
    REGIONS = "[[0.1, 0.2, 0.9, 0.35]]"  # 底部20%区域

    print("="*60)
    print("🧪 测试：裁剪 + 分段 + 字幕擦除")
    print("="*60)
    print(f"输入视频: {INPUT_VIDEO}")
    print(f"擦除区域: {REGIONS}")
    print(f"模型: {CHECKPOINT}")

    # 检查文件
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ 视频文件不存在: {INPUT_VIDEO}")
        sys.exit(1)

    if not os.path.exists(CHECKPOINT):
        print(f"❌ 模型文件不存在: {CHECKPOINT}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 测试1: 仅裁剪模式（短视频）
    print("\n\n" + "="*60)
    print("测试1: 仅裁剪模式（不分段）")
    print("="*60)

    cmd1 = [
        "python", "main.py",
        "-v", INPUT_VIDEO,
        "-c", CHECKPOINT,
        "--regions", REGIONS,
        "--crop",
        "--output", f"{OUTPUT_DIR}/result_crop_only.mp4"
    ]

    run_cmd(cmd1, "测试1: 裁剪模式")

    # 测试2: 裁剪 + 自动分段
    print("\n\n" + "="*60)
    print("测试2: 裁剪 + 自动分段")
    print("="*60)

    cmd2 = [
        "python", "main.py",
        "-v", INPUT_VIDEO,
        "-c", CHECKPOINT,
        "--regions", REGIONS,
        "--crop",
        "--auto-split",
        "--max-frames", "30",  # 强制分段（每段30帧）
        "--output", f"{OUTPUT_DIR}/result_crop_split.mp4"
    ]

    run_cmd(cmd2, "测试2: 裁剪+分段模式")

    # 总结
    print("\n\n" + "="*60)
    print("🎉 所有测试完成！")
    print("="*60)
    print("\n生成的文件:")
    print(f"  1. 仅裁剪: {OUTPUT_DIR}/result_crop_only.mp4")
    print(f"  2. 裁剪+分段: {OUTPUT_DIR}/result_crop_split.mp4")
    print("\n查看结果:")
    print(f"  open {OUTPUT_DIR}/result_crop_only.mp4")
    print(f"  open {OUTPUT_DIR}/result_crop_split.mp4")
    print("\n对比原视频:")
    print(f"  open {INPUT_VIDEO}")


if __name__ == "__main__":
    main()
