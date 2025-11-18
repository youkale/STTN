# 测试指南

本目录包含三个测试脚本，用于验证视频擦除功能。

## 测试脚本说明

### 1. `test_mask.py` - Mask图片测试（推荐用于理解流程）

**功能**：
- 截取5秒测试视频
- 调整分辨率到432x240
- 生成mask图片（底部20%区域）
- 使用mask模式进行擦除

**使用方法**：
```bash
python test_mask.py
```

**适用场景**：
- 理解mask如何工作
- 验证mask生成是否正确
- 测试基础擦除功能

---

### 2. `test_simple.py` - 完整工作流测试（推荐用于验证功能）

**功能**：
- 测试1: 裁剪模式（--crop）
- 测试2: 裁剪+自动分段模式（--crop --auto-split）

**使用方法**：
```bash
python test_simple.py
```

**适用场景**：
- 验证裁剪功能
- 验证自动分段功能
- 对比不同模式的效果

---

### 3. `test_workflow.sh` - Shell脚本测试

**功能**：
- 使用ffmpeg手动裁剪视频
- 生成mask图片
- 测试擦除效果

**使用方法**：
```bash
./test_workflow.sh
```

**适用场景**：
- 需要更精细的控制
- 理解每个步骤的细节
- macOS/Linux环境

---

## 快速开始

### 最简单的测试

```bash
# 1. 确保模型文件存在
ls checkpoints/sttn.pth

# 2. 确保测试视频存在
ls examples/subtitle_4.mp4

# 3. 运行测试
python test_mask.py
```

### 完整功能测试

```bash
python test_simple.py
```

---

## 输出文件

所有测试脚本都会将结果保存在 `test_output/` 目录：

```
test_output/
├── test_5sec.mp4          # 截取的5秒视频
├── test_resized.mp4       # 调整分辨率后的视频
├── masks/
│   └── mask.png           # 生成的mask图片
├── result_with_mask.mp4   # 使用mask的擦除结果
├── result_crop_only.mp4   # 裁剪模式结果
└── result_crop_split.mp4  # 裁剪+分段模式结果
```

---

## 查看结果

### macOS
```bash
open test_output/result_with_mask.mp4
open test_output/masks/mask.png
```

### Linux
```bash
xdg-open test_output/result_with_mask.mp4
```

---

## 参数说明

### 擦除区域坐标（--regions）

格式：`[[left, bottom, right, top]]`

- 使用**相对坐标**（0-1之间）
- 原点在**左下角**
- 示例：
  - `[[0.1, 0, 0.9, 0.2]]` - 底部20%，左右各留10%
  - `[[0, 0, 1, 0.15]]` - 整个底部15%
  - `[[0.3, 0.4, 0.7, 0.6]]` - 中央区域

### 模式参数

| 参数 | 说明 | 推荐场景 |
|------|------|----------|
| `--crop` | 仅处理区域周围部分 | 固定区域擦除（如字幕） |
| `--auto-split` | 自动分段处理 | 视频太长导致显存不足 |
| `--max-frames N` | 每段最大帧数 | 手动控制分段大小 |
| `--crop-padding N` | 裁剪边界padding | 避免边界artifacts |

---

## 故障排查

### 显存不足

```bash
# 方案1: 使用裁剪模式
python main.py -v video.mp4 -c model.pth --regions '...' --crop

# 方案2: 裁剪+分段
python main.py -v video.mp4 -c model.pth --regions '...' --crop --auto-split

# 方案3: 手动指定更小的分段
python main.py -v video.mp4 -c model.pth --regions '...' --crop --auto-split --max-frames 30
```

### 分辨率错误

确保视频分辨率是**432x240的倍数**，脚本会自动调整。

### Mask尺寸不匹配

确保mask图片尺寸与处理视频一致（不是原视频）。

---

## 进阶用法

### 仅测试裁剪（不擦除）

```bash
# 查看裁剪后的视频尺寸
python -c "
import json
import sys
sys.path.insert(0, '.')
from main import calculate_crop_region

regions = [[0.1, 0, 0.9, 0.2]]
(x, y, w, h), bounds = calculate_crop_region(regions, 1920, 1080, 32)
print(f'裁剪: x={x}, y={y}, w={w}, h={h}')
"
```

### 自定义测试区域

修改 `test_simple.py` 中的 `REGIONS` 变量：

```python
# 擦除顶部10%
REGIONS = "[[0, 0.9, 1, 1]]"

# 擦除中央矩形
REGIONS = "[[0.3, 0.4, 0.7, 0.6]]"

# 擦除多个区域
REGIONS = "[[0.1, 0, 0.4, 0.2], [0.6, 0, 0.9, 0.2]]"
```

---

## 性能对比

| 模式 | 显存占用 | 处理速度 | 质量 |
|------|---------|---------|------|
| 原视频处理 | 高 | 快 | 最佳 |
| 裁剪模式 | 低 | 快 | 最佳 |
| 裁剪+分段 | 最低 | 中 | 最佳 |

**建议**：
- 短视频(<5秒) + 大显存(>12GB): 原视频处理
- 长视频 或 小显存: 裁剪+分段模式
