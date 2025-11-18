#!/bin/bash

# 测试工作流程脚本
# 功能：
# 1. 裁剪视频到指定区域
# 2. 截取前5秒
# 3. 生成mask图片
# 4. 测试擦除效果

set -e  # 遇到错误立即退出

echo "=========================================="
echo "测试工作流程"
echo "=========================================="

# 配置参数
INPUT_VIDEO="examples/subtitle_4.mp4"
OUTPUT_DIR="test_output"
REGION='[[0.1, 0, 0.9, 0.2]]'  # 底部20%区域
CHECKPOINT="checkpoints/sttn.pth"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo ""
echo "步骤1: 获取原视频信息"
echo "=========================================="
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,duration \
  -of default=noprint_wrappers=1:nokey=0 \
  "$INPUT_VIDEO"

echo ""
echo "步骤2: 截取前5秒视频"
echo "=========================================="
TEST_VIDEO="$OUTPUT_DIR/test_5sec.mp4"
ffmpeg -y -i "$INPUT_VIDEO" -t 5 -c copy "$TEST_VIDEO"
echo "✓ 已保存: $TEST_VIDEO"

echo ""
echo "步骤3: 裁剪视频到字幕区域"
echo "=========================================="
# 假设原视频是1920x1080，底部20%就是1920x216
# 需要对齐到432x240的倍数，所以调整为1728x240
CROPPED_VIDEO="$OUTPUT_DIR/test_cropped.mp4"

# 先获取视频尺寸
VIDEO_INFO=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$TEST_VIDEO")
VIDEO_WIDTH=$(echo $VIDEO_INFO | cut -d'x' -f1)
VIDEO_HEIGHT=$(echo $VIDEO_INFO | cut -d'x' -f2)

echo "原视频尺寸: ${VIDEO_WIDTH}x${VIDEO_HEIGHT}"

# 计算裁剪参数（底部20%）
CROP_HEIGHT=$((VIDEO_HEIGHT * 20 / 100))
CROP_Y=$((VIDEO_HEIGHT - CROP_HEIGHT))

# 对齐到240的倍数
ALIGNED_HEIGHT=240
CROP_Y=$((VIDEO_HEIGHT - ALIGNED_HEIGHT))

echo "裁剪参数: width=${VIDEO_WIDTH}, height=${ALIGNED_HEIGHT}, y=${CROP_Y}"

ffmpeg -y -i "$TEST_VIDEO" \
  -filter:v "crop=${VIDEO_WIDTH}:${ALIGNED_HEIGHT}:0:${CROP_Y}" \
  "$CROPPED_VIDEO"

echo "✓ 已保存: $CROPPED_VIDEO"

echo ""
echo "步骤4: 生成mask图片"
echo "=========================================="
MASK_DIR="$OUTPUT_DIR/masks"
mkdir -p "$MASK_DIR"

# 获取裁剪后的视频尺寸
CROP_INFO=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$CROPPED_VIDEO")
CROP_WIDTH=$(echo $CROP_INFO | cut -d'x' -f1)
CROP_HEIGHT=$(echo $CROP_INFO | cut -d'x' -f2)

echo "裁剪后尺寸: ${CROP_WIDTH}x${CROP_HEIGHT}"

# 使用Python生成mask
python3 << EOF
import cv2
import numpy as np
from PIL import Image

# 创建mask（整个区域都是白色，因为整个裁剪区域都要擦除）
mask = np.ones((${CROP_HEIGHT}, ${CROP_WIDTH}), dtype=np.uint8) * 255

# 应用膨胀（与main.py保持一致）
mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)

# 保存
Image.fromarray(mask).save('${MASK_DIR}/mask.png')
print(f"✓ 已生成mask: ${MASK_DIR}/mask.png")
print(f"  尺寸: ${CROP_WIDTH}x${CROP_HEIGHT}")
print(f"  白色像素: {np.sum(mask == 255)}")
EOF

echo ""
echo "步骤5: 测试擦除效果（不使用--crop和--auto-split）"
echo "=========================================="
python main.py \
  -v "$CROPPED_VIDEO" \
  -m "$MASK_DIR" \
  -c "$CHECKPOINT"

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo "生成的文件："
echo "  1. 5秒测试视频: $TEST_VIDEO"
echo "  2. 裁剪后视频: $CROPPED_VIDEO"
echo "  3. Mask图片: $MASK_DIR/mask.png"
echo "  4. 擦除结果: output/masks_result.mp4"
echo ""
echo "查看结果："
echo "  原始裁剪视频: open $CROPPED_VIDEO"
echo "  擦除后视频:   open output/masks_result.mp4"
echo "  Mask图片:     open $MASK_DIR/mask.png"
