# MTTrack 多目标智能追踪系统

[![Version](https://img.shields.io/badge/version-0.2.0-blue)](https://pypi.org/project/mttrack/)
[![Python](https://img.shields.io/badge/python-3.8+-green)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/mttrack)](https://pypi.org/project/mttrack/)

多目标实时追踪系统，支持结合视觉语言模型进行目标分类。





https://github.com/user-attachments/assets/371a0c8f-ac1a-4532-9363-7152a953e1f1






## 功能特性

### 基础功能
- **多种追踪算法**: 支持 ByteTrack 和 SORT 经典算法
- **YOLO 集成**: 支持任意 YOLO 模型 (YOLOv8, YOLOv10, YOLOv11 等)
- **VL 分类增强**: 可选的视觉语言模型，用于细粒度目标分类
- **模块化设计**: 清晰的领域层、服务层、基础设施层分离
- **视频读写**: 便捷的视频读写工具

### 增强模式 (v0.2.0+)
- **外观特征提取**: 轻量级外观 embedding，实现鲁棒的目标重识别
- **自适应 VL 触发**: 智能决策何时调用 VL 模型
- **多特征融合关联**: IoU + 外观 + 运动多信号联合决策
- **自适应阈值**: 根据场景密度动态调整关联阈值

---

## 安装

### 从 PyPI 安装 (推荐)

```bash
pip install mttrack
```

### 从源码安装

```bash
pip install -e .
```

### 依赖

MTTrack 需要以下依赖：
- `numpy`
- `opencv-python`
- `Pillow` (图像处理)
- `scipy` (科学计算)
- `ultralytics` (YOLO 支持)
- `requests` (VLLM API 调用)

---

## 快速开始

### 1. 准备 YOLO 模型

将 YOLO 模型文件放入 `models/` 目录：

```bash
# 下载 YOLO 模型，例如：
# https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt
# 放入 models/ 目录
```

### 2. 运行追踪

```bash
# 标准模式
mttrack --input data/test.mp4 --output result.mp4

# 或者使用 Python API（见下文）
```

---

## 命令行使用

### 基础命令

```bash
# 标准模式 - 使用默认配置
mttrack --input 输入视频路径 --output 输出视频路径

# 示例
mttrack --input data/test.mp4 --output out/result.mp4
```

### 增强模式

增强模式包含外观特征提取、自适应 VL 触发、多特征融合关联等功能：

```bash
mttrack \
    --input data/test.mp4 \
    --output out/result_enhanced.mp4 \
    --enhanced
```

### 完整参数示例

```bash
mttrack \
    --input data/test.mp4 \
    --output out/result.mp4 \
    --tracker bytetrack \
    --yolo-model ./models/yolo11x.pt \
    --confidence 0.25 \
    --device cuda \
    --enable-vl \
    --vl-interval 30 \
    --vl-timeout 30 \
    --enhanced \
    --no-appearance \
    --vl-min-interval 30 \
    --vl-max-interval 150
```

### 命令行参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--input` | `-i` | 输入视频路径（必需） | - |
| `--output` | `-o` | 输出视频路径（必需） | - |
| `--tracker` | - | 追踪算法: `bytetrack` 或 `sort` | `bytetrack` |
| `--yolo-model` | - | YOLO 模型路径 | `./models/yolo11x.pt` |
| `--confidence` | - | 检测置信度阈值 | `0.25` |
| `--device` | - | YOLO 设备: `cuda`, `cpu`, `0`, `1` | `cuda` |
| `--enable-vl` | - | 启用 VL 分类（需要 VLLM 服务） | False |
| `--vl-interval` | - | VL 分类间隔帧数（标准模式） | `30` |
| `--vl-timeout` | - | VL API 超时秒数 | `30` |
| `--enhanced` | - | 启用增强模式 | False |
| `--no-appearance` | - | 增强模式下禁用外观特征 | False |
| `--vl-min-interval` | - | 增强模式 VL 最小间隔 | `30` |
| `--vl-max-interval` | - | 增强模式 VL 最大间隔 | `150` |
| `--show-fps` | - | 在输出视频上显示 FPS | False |

---

## Python API 使用

### 方式一：标准模式

```python
import cv2
from mttrack import (
    TrackerService,
    YoloDetector,
    TrackingAnnotator,
    VideoReader,
    VideoWriter,
)

# 1. 初始化检测器
detector = YoloDetector(
    model_path="./models/yolo11x.pt",
    confidence_threshold=0.25,
    device="cuda"
)

# 2. 初始化追踪服务
tracker_service = TrackerService(
    detector=detector,
    tracker_type="bytetrack"  # 或 "sort"
)

# 3. 初始化标注器
annotator = TrackingAnnotator()

# 4. 处理视频
input_video = "input.mp4"
output_video = "output.mp4"

with VideoReader(input_video) as reader:
    # 创建写入器
    writer = VideoWriter(
        output_video,
        fps=reader.fps,
        frame_size=(reader.width, reader.height)
    )

    for frame_id, frame in reader:
        # 处理每一帧
        result = tracker_service.process_frame(frame)

        # 标注跟踪结果
        annotated = annotator.annotate(frame, result.tracks)

        # 写入输出视频
        writer.write(annotated)

    writer.close()

print("处理完成！")
```

### 方式二：增强模式（推荐生产使用）

```python
import cv2
from mttrack import (
    EnhancedTrackerService,
    YoloDetector,
    TrackingAnnotator,
    VideoReader,
    VideoWriter,
)

# 1. 初始化检测器
detector = YoloDetector(
    model_path="./models/yolo11x.pt",
    confidence_threshold=0.25,
    device="cuda"
)

# 2. 初始化增强版追踪服务
tracker_service = EnhancedTrackerService(
    detector=detector,
    tracker_type="bytetrack",
    enable_appearance=True,       # 启用外观特征提取
    enable_adaptive_vl=True,     # 启用自适应 VL 触发
    vl_min_interval=30,         # VL 最小间隔帧数
    vl_max_interval=150,         # VL 最大间隔帧数
)

# 3. 初始化标注器
annotator = TrackingAnnotator()

# 4. 处理视频
with VideoReader("input.mp4") as reader:
    writer = VideoWriter(
        "output.mp4",
        fps=reader.fps,
        frame_size=(reader.width, reader.height)
    )

    for frame_id, frame in reader:
        result = tracker_service.process_frame(frame)
        annotated = annotator.annotate(frame, result.tracks)
        writer.write(annotated)

    writer.close()
```

### 方式三：结合 VL 分类

```python
import cv2
from mttrack import (
    EnhancedTrackerService,
    YoloDetector,
    TrackingAnnotator,
    VideoReader,
    VideoWriter,
)
from mttrack.infrastructure import VllmClient
from mttrack.service import LabelService


def crop_track(frame, bbox, margin=10):
    """裁剪跟踪区域"""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


# 1. 初始化 VL 客户端（可选，需要 VLLM 服务）
vllm_client = None  # 或 VllmClient(base_url="http://localhost:8000", api_key="sk-xxx")

# 2. 初始化检测器
detector = YoloDetector(
    model_path="./models/yolo11x.pt",
    confidence_threshold=0.25,
    device="cuda"
)

# 3. 初始化追踪服务
tracker_service = EnhancedTrackerService(
    detector=detector,
    tracker_type="bytetrack",
    enable_appearance=True,
    enable_adaptive_vl=vllm_client is not None,
)

# 4. 初始化标签服务
label_service = LabelService(
    vllm_client=vllm_client,
    enabled=vllm_client is not None,
    label_interval=30,
    cache_ttl=60.0
)

# 5. 处理视频
annotator = TrackingAnnotator()

with VideoReader("input.mp4") as reader:
    writer = VideoWriter(
        "output.mp4",
        fps=reader.fps,
        frame_size=(reader.width, reader.height)
    )

    for frame_id, frame in reader:
        result = tracker_service.process_frame(frame)

        # 获取活跃跟踪 ID
        active_track_ids = {t.track_id for t in result.tracks}

        # 清理过期缓存
        label_service.cleanup_old_tracks(active_track_ids)

        # VL 分类（如果启用）
        if vllm_client:
            for track in result.tracks:
                # 检查是否需要分类
                should_label, reason = tracker_service.should_classify_vl(
                    track.track_id,
                    track.bbox,
                    track.label_confidence if track.label_confidence > 0 else track.confidence
                )

                if should_label:
                    # 裁剪目标区域
                    crop = crop_track(frame, track.bbox)
                    if crop is not None and crop.size > 0:
                        # 调用 VL 分类
                        vl_result = label_service.label_track(
                            track.track_id,
                            crop,
                            frame_id
                        )
                        if vl_result and vl_result.class_name != "unknown":
                            # 更新标签
                            track.label = vl_result.class_name
                            track.label_confidence = vl_result.confidence
                            tracker_service.update_track_label(
                                track.track_id,
                                vl_result.class_name,
                                vl_result.confidence
                            )

        # 标注并输出
        annotated = annotator.annotate(frame, result.tracks)
        writer.write(annotated)

    writer.close()
```

### 高级：自定义检测器

```python
import numpy as np
from mttrack.infrastructure import BaseDetector, DetectorResult


class MyDetector(BaseDetector):
    """自定义检测器示例"""

    def __init__(self, model_path):
        # 这里加载你的模型
        pass

    def detect(self, image):
        # 返回检测结果
        return DetectorResult(
            boxes=np.array([[100, 100, 200, 200]], dtype=np.float32),
            confidences=np.array([0.9]),
            class_ids=np.array([0]),
            class_names=["person"]
        )

    def warmup(self):
        pass


# 使用自定义检测器
detector = MyDetector("my_model.pt")
tracker_service = TrackerService(detector=detector)
```

---

## VL 模型配置

### 什么是 VL 分类？

VL（Vision-Language）分类使用视觉语言大模型（如 Qwen2-VL）对跟踪目标进行细粒度分类，可以识别比 YOLO 更丰富的目标类别。

### 配置步骤

#### 1. 设置环境变量

```bash
# 方式一：命令行设置（临时）
export VLLM_BASE_URL="http://localhost:8000"
export VLLM_API_KEY="sk-your-api-key"
export VLLM_MODEL="/models/Qwen/Qwen3-VL-8B-Instruct"

# 方式二：运行命令时设置
VLLM_BASE_URL="http://localhost:8000" \
VLLM_API_KEY="sk-your-api-key" \
VLLM_MODEL="/models/Qwen/Qwen3-VL-8B-Instruct" \
mttrack --input video.mp4 --output result.mp4 --enable-vl
```

#### 2. 环境变量说明

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `VLLM_BASE_URL` | VLLM API 服务器地址 | `http://localhost:8000` |
| `VLLM_API_KEY` | API 密钥（可随意设置） | `sk-xxxx` |
| `VLLM_MODEL` | VL 模型名称 | `/models/Qwen/Qwen3-VL-8B-Instruct` |

#### 3. 验证 VL 是否启用

运行后观察日志：
- **启用成功**：
  ```
  [Info] Initializing VLLM client...
  [Info] VLLM client initialized (base_url: http://xxx:8000)
  ```
- **启用失败**：
  ```
  [Warning] VLLM_BASE_URL not set, VL classification disabled
  ```

---

## 配置参数详解

### YOLO 检测器参数

```python
detector = YoloDetector(
    model_path="./models/yolo11x.pt",  # 模型路径
    confidence_threshold=0.25,         # 置信度阈值（0-1）
    device="cuda"                      # 设备：cuda/cpu/0/1
)
```

### ByteTrack 参数

```python
from mttrack.domain import ByteTrackTracker

tracker = ByteTrackTracker(
    lost_track_buffer=30,              # 丢失缓冲帧数
    frame_rate=30.0,                   # 帧率
    track_activation_threshold=0.7,    # 轨迹激活阈值
    minimum_consecutive_frames=2,     # 最小连续帧数
    minimum_iou_threshold=0.1,         # 最小 IoU 阈值
    high_conf_det_threshold=0.6        # 高置信度检测阈值
)
```

### SORT 参数

```python
from mttrack.domain import SORTTracker

tracker = SORTTracker(
    lost_track_buffer=30,
    frame_rate=30.0,
    track_activation_threshold=0.25,
    minimum_consecutive_frames=3,
    minimum_iou_threshold=0.3
)
```

### 增强模式参数

```python
from mttrack import EnhancedTrackerService

tracker = EnhancedTrackerService(
    detector=detector,
    tracker_type="bytetrack",

    # 外观特征
    enable_appearance=True,
    appearance_memory_size=10,

    # 自适应 VL
    enable_adaptive_vl=True,
    vl_min_interval=30,
    vl_max_interval=150,

    # 多特征关联
    enable_multi_feature=True,
    use_appearance_in_association=True,
)
```

---

## API 参考

### 核心类

| 类名 | 说明 |
|------|------|
| `YoloDetector` | YOLO 检测器 |
| `TrackerService` | 标准追踪服务 |
| `EnhancedTrackerService` | 增强版追踪服务（支持外观特征、自适应 VL） |
| `LabelService` | VL 标签服务 |
| `TrackingAnnotator` | 跟踪结果标注器 |
| `VideoReader` | 视频读取器 |
| `VideoWriter` | 视频写入器 |
| `VllmClient` | VLLM API 客户端 |

### 数据类

| 类名 | 说明 |
|------|------|
| `DetectorResult` | 检测结果 |
| `TrackInfo` | 单个跟踪目标信息 |
| `FrameTracks` | 单帧所有跟踪目标 |
| `VLClassificationResult` | VL 分类结果 |
| `LabelCache` | VL 标签缓存 |

### 1. YoloDetector - YOLO 检测器

YOLO 目标检测器封装类，用于检测视频帧中的目标。

```python
from mttrack import YoloDetector

detector = YoloDetector(
    model_path="./models/yolo11x.pt",
    confidence_threshold=0.25,
    device="cuda"
)
```

#### 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | `str` | 必需 | YOLO 模型文件路径，支持 YOLOv8/YOLOv10/YOLOv11 等 |
| `confidence_threshold` | `float` | `0.25` | 检测置信度阈值，范围 0-1 |
| `device` | `str` | `"cuda"` | 运行设备，可选 `"cuda"`, `"cpu"`, `"0"`, `"1"` 等 |

#### 方法

##### `detect(image: np.ndarray) -> DetectorResult`

检测图像中的目标。

**参数：**
- `image`: BGR 格式的图像 (numpy array)

**返回：**
- `DetectorResult` 对象

##### `warmup() -> None`

预热检测器，加载模型到设备。

---

### 2. TrackerService - 标准追踪服务

标准模式的多目标追踪服务，不包含增强功能。

```python
from mttrack import TrackerService

tracker_service = TrackerService(
    detector=detector,
    tracker_type="bytetrack",  # 或 "sort"
    tracker_kwargs={}
)
```

#### 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `detector` | `BaseDetector` | 必需 | 目标检测器实例 |
| `tracker_type` | `str` | `"bytetrack"` | 追踪算法类型，可选 `"bytetrack"` 或 `"sort"` |
| `tracker_kwargs` | `dict` | `{}` | 追踪器额外参数 |

#### 方法

##### `process_frame(frame: np.ndarray) -> FrameTracks`

处理单帧图像，返回追踪结果。

**参数：**
- `frame`: BGR 格式的图像

**返回：**
- `FrameTracks` 对象，包含该帧的所有追踪目标

##### `update_track_label(track_id: int, label: str, confidence: float) -> None`

更新追踪目标的标签（用于 VL 分类结果）。

**参数：**
- `track_id`: 追踪 ID
- `label`: 标签名称
- `confidence`: 标签置信度

##### `reset() -> None`

重置追踪器状态。

##### `warmup() -> None`

预热检测器。

---

### 3. EnhancedTrackerService - 增强版追踪服务

增强模式的多目标追踪服务，支持外观特征提取、自适应 VL 触发、多特征融合关联。

```python
from mttrack import EnhancedTrackerService

tracker_service = EnhancedTrackerService(
    detector=detector,
    tracker_type="bytetrack",
    tracker_kwargs={},
    enable_appearance=True,
    appearance_memory_size=10,
    enable_adaptive_vl=True,
    vl_min_interval=30,
    vl_max_interval=150,
    enable_multi_feature=True,
    use_appearance_in_association=True,
)
```

#### 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `detector` | `BaseDetector` | 必需 | 目标检测器实例 |
| `tracker_type` | `str` | `"bytetrack"` | 追踪算法类型，可选 `"bytetrack"` 或 `"sort"` |
| `tracker_kwargs` | `dict` | `{}` | 追踪器额外参数 |
| `enable_appearance` | `bool` | `True` | 是否启用外观特征提取 |
| `appearance_memory_size` | `int` | `10` | 外观特征内存大小 |
| `enable_adaptive_vl` | `bool` | `True` | 是否启用自适应 VL 触发 |
| `vl_min_interval` | `int` | `30` | VL 分类最小间隔帧数 |
| `vl_max_interval` | `int` | `150` | VL 分类最大间隔帧数 |
| `enable_multi_feature` | `bool` | `True` | 是否启用多特征融合关联 |
| `use_appearance_in_association` | `bool` | `True` | 是否在关联中使用外观特征 |

#### 方法

##### `process_frame(frame: np.ndarray) -> FrameTracks`

处理单帧图像，返回增强版追踪结果。

##### `should_classify_vl(track_id: int, bbox: tuple, current_confidence: float) -> tuple[bool, str]`

判断是否需要触发 VL 分类。

**参数：**
- `track_id`: 追踪 ID
- `bbox`: 边界框 (x1, y1, x2, y2)
- `current_confidence`: 当前置信度

**返回：**
- `(should_trigger, reason)` 元组

##### `update_track_label(track_id: int, label: str, confidence: float) -> None`

更新追踪目标标签。

##### `get_appearance_feature(track_id: int) -> Optional[np.ndarray]`

获取追踪目标的外观特征。

##### `reset() -> None`

重置追踪器状态。

##### `warmup() -> None`

预热检测器。

---

### 4. LabelService - VL 标签服务

视觉语言模型标签服务，用于对追踪目标进行细粒度分类。

```python
from mttrack import LabelService

label_service = LabelService(
    vllm_client=vllm_client,
    enabled=True,
    label_interval=30,
    cache_ttl=60.0,
    max_concurrent=1
)
```

#### 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vllm_client` | `VllmClient` | `None` | VLLM 客户端实例 |
| `enabled` | `bool` | `True` | 是否启用标签服务 |
| `label_interval` | `int` | `30` | VL 分类间隔帧数 |
| `cache_ttl` | `float` | `60.0` | 缓存有效期（秒） |
| `max_concurrent` | `int` | `1` | 最大并发请求数 |

#### 方法

##### `should_label(track_id: int, frame_id: int) -> bool`

判断是否应该对目标进行 VL 分类。

##### `label_track(track_id: int, crop: np.ndarray, frame_id: int) -> Optional[VLClassificationResult]`

对裁剪图像进行 VL 分类。

##### `get_cached_label(track_id: int) -> Optional[LabelCache]`

获取缓存的标签。

##### `cleanup_old_tracks(active_track_ids: set) -> None`

清理非活跃追踪目标的缓存。

##### `is_available() -> bool`

检查 VL 服务是否可用。

---

### 5. VllmClient - VLLM 客户端

视觉语言模型 API 客户端，用于调用 VLLM 服务进行目标分类。

```python
from mttrack.infrastructure import VllmClient

vllm_client = VllmClient(
    base_url="http://localhost:8000",
    api_key="sk-your-api-key",
    model="/models/Qwen/Qwen3-VL-8B-Instruct",
    timeout=30,
    max_retries=3
)
```

#### 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_url` | `Optional[str]` | `None` | VLLM API 基础 URL，可通过环境变量 `VLLM_BASE_URL` 设置 |
| `api_key` | `Optional[str]` | `None` | API 密钥，可通过环境变量 `VLLM_API_KEY` 设置 |
| `model` | `Optional[str]` | `None` | 模型名称，可通过环境变量 `VLLM_MODEL` 设置 |
| `timeout` | `int` | `30` | 请求超时时间（秒） |
| `max_retries` | `int` | `3` | 最大重试次数 |

**注意：** 如果不传入参数，会尝试从环境变量读取，默认值为：
- `base_url`: `"http://10.132.19.82:50100"`
- `api_key`: `"sk-8fA3kP2QxR7mJ9WZC6dE0T1B4yH5VnL"`
- `model`: `"/models/Qwen/Qwen3-VL-8B-Instruct"`

#### 方法

##### `is_available() -> bool`

检查 VLLM 服务是否可用。

##### `classify_crop(image: np.ndarray, track_id: int) -> VLClassificationResult`

对裁剪图像进行 VL 分类。

**参数：**
- `image`: BGR 格式的裁剪图像
- `track_id`: 追踪 ID（用于日志）

**返回：**
- `VLClassificationResult` 对象

---

### 6. TrackingAnnotator - 追踪结果标注器

用于在视频帧上绘制追踪结果的可视化工具。

```python
from mttrack import TrackingAnnotator

annotator = TrackingAnnotator(
    thickness=2,
    font_scale=0.5,
    text_thickness=1
)
```

#### 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `thickness` | `int` | `2` | 边界框线条粗细 |
| `font_scale` | `float` | `0.5` | 文本字体大小 |
| `text_thickness` | `int` | `1` | 文本线条粗细 |

#### 方法

##### `annotate(frame: np.ndarray, tracks: list[TrackInfo]) -> np.ndarray`

在帧上绘制追踪结果。

**参数：**
- `frame`: BGR 格式的图像
- `tracks`: 追踪目标列表

**返回：**
- 绘制后的图像

---

### 7. VideoReader - 视频读取器

用于读取视频文件或摄像头流。

```python
from mttrack import VideoReader

with VideoReader("input.mp4") as reader:
    print(f"分辨率: {reader.width}x{reader.height}")
    print(f"帧率: {reader.fps}")
    print(f"总帧数: {reader.frame_count}")

    for frame_id, frame in reader:
        # 处理每一帧
        pass
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `width` | `int` | 视频宽度 |
| `height` | `int` | 视频高度 |
| `fps` | `float` | 帧率 |
| `frame_count` | `int` | 总帧数 |

#### 方法

##### `read() -> tuple[bool, Optional[np.ndarray]]`

读取单帧。返回 `(success, frame)` 元组。

---

### 8. VideoWriter - 视频写入器

用于将处理后的帧写入视频文件。

```python
from mttrack import VideoWriter

writer = VideoWriter(
    output_path="output.mp4",
    fps=30.0,
    frame_size=(1920, 1080),
    codec="mp4v"
)

# 写入帧
writer.write(frame)

# 关闭写入器
writer.close()
```

#### 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_path` | `str` | 必需 | 输出视频路径 |
| `fps` | `float` | `30.0` | 输出视频帧率 |
| `frame_size` | `Optional[tuple[int, int]]` | `None` | 视频尺寸 (width, height) |
| `codec` | `str` | `"mp4v"` | 视频编码器 |

#### 方法

##### `write(frame: np.ndarray) -> None`

写入单帧。

##### `close() -> None`

关闭写入器并释放资源。

---

### 9. 数据类

#### DetectorResult - 检测结果

```python
from mttrack.infrastructure import DetectorResult
import numpy as np

result = DetectorResult(
    boxes=np.array([[100, 100, 200, 200]], dtype=np.float32),  # (N, 4) xyxy 格式
    confidences=np.array([0.9]),                                  # (N,) 置信度
    class_ids=np.array([0]),                                     # (N,) 类别 ID
    class_names=["person"]                                        # 类别名称列表
)
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `boxes` | `np.ndarray` | 边界框数组，形状 (N, 4)，xyxy 格式 |
| `confidences` | `np.ndarray` | 置信度数组，形状 (N,) |
| `class_ids` | `np.ndarray` | 类别 ID 数组，形状 (N,) |
| `class_names` | `list[str]` | 类别名称列表 |

#### TrackInfo - 追踪目标信息

**标准模式 (TrackerService)：**

```python
from mttrack.service import TrackInfo

track = TrackInfo(
    track_id=1,
    bbox=(100, 100, 200, 200),
    class_name="person",
    class_id=0,
    confidence=0.9,
    label="person",
    label_confidence=0.9
)
```

**增强模式 (EnhancedTrackerService)：**

增强模式下的 TrackInfo 额外包含以下字段：

```python
from mttrack.service import EnhancedTrackerService
# 通过 EnhancedTrackerService.process_frame() 返回的 tracks
# 每个 track 包含以下额外字段:
track.velocity = (0.0, 0.0)                    # 目标速度 (vx, vy)
track.appearance_feature = np.array([...])    # 外观特征向量
track.appearance_change = 0.0                  # 外观变化程度
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `track_id` | `int` | 追踪 ID |
| `bbox` | `tuple[float, float, float, float]` | 边界框 (x1, y1, x2, y2) |
| `class_name` | `str` | YOLO 检测类别名称 |
| `class_id` | `int` | YOLO 检测类别 ID |
| `confidence` | `float` | 检测置信度 |
| `label` | `Optional[str]` | VL 分类标签（如果有） |
| `label_confidence` | `float` | VL 分类置信度 |
| `velocity` | `tuple[float, float]` | 目标速度（增强模式） |
| `appearance_feature` | `Optional[np.ndarray]` | 外观特征向量（增强模式） |
| `appearance_change` | `float` | 外观变化程度（增强模式） |

#### FrameTracks - 单帧追踪结果

```python
from mttrack.service import FrameTracks

frame_tracks = FrameTracks(
    frame_id=1,
    tracks=[track1, track2, ...]
)
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `frame_id` | `int` | 帧 ID |
| `tracks` | `list[TrackInfo]` | 该帧的所有追踪目标 |

#### VLClassificationResult - VL 分类结果

```python
from mttrack.infrastructure import VLClassificationResult

result = VLClassificationResult(
    class_name="person",
    confidence=0.95,
    raw_response='{"class": "person", "confidence": 0.95}'
)
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `class_name` | `str` | 分类类别名称 |
| `confidence` | `float` | 分类置信度 (0-1) |
| `raw_response` | `str` | 原始 API 响应 |

#### LabelCache - 标签缓存

```python
from mttrack.service import LabelCache

cache = LabelCache(
    class_name="person",
    confidence=0.9,
    timestamp=1234567890.0
)
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `class_name` | `str` | 缓存的类别名称 |
| `confidence` | `float` | 缓存的置信度 |
| `timestamp` | `float` | 缓存时间戳 |

---

### 10. ByteTrackTracker / SORTTracker - 追踪器

直接使用追踪器类进行更精细的控制。

#### ByteTrackTracker 参数

```python
from mttrack.domain import ByteTrackTracker

tracker = ByteTrackTracker(
    lost_track_buffer=30,           # 丢失轨迹缓冲帧数
    frame_rate=30.0,                 # 视频帧率
    track_activation_threshold=0.7,  # 轨迹激活阈值
    minimum_consecutive_frames=2,    # 成为稳定轨迹的最小连续帧数
    minimum_iou_threshold=0.1,       # 最小 IoU 匹配阈值
    high_conf_det_threshold=0.6      # 高/低置信度检测的分界阈值
)
```

#### SORTTracker 参数

```python
from mttrack.domain import SORTTracker

tracker = SORTTracker(
    lost_track_buffer=30,           # 丢失轨迹缓冲帧数
    frame_rate=30.0,                 # 视频帧率
    track_activation_threshold=0.25, # 轨迹激活阈值
    minimum_consecutive_frames=3,    # 成为稳定轨迹的最小连续帧数
    minimum_iou_threshold=0.3        # 最小 IoU 匹配阈值
)
```

---

### 11. 自定义检测器

如果需要使用其他检测器，可以实现 `BaseDetector` 接口：

```python
from mttrack.infrastructure import BaseDetector, DetectorResult
import numpy as np

class MyDetector(BaseDetector):
    def __init__(self, model_path: str):
        # 加载你的模型
        pass

    def detect(self, image: np.ndarray) -> DetectorResult:
        # 返回检测结果
        return DetectorResult(
            boxes=np.array([[100, 100, 200, 200]], dtype=np.float32),
            confidences=np.array([0.9]),
            class_ids=np.array([0]),
            class_names=["person"]
        )

    def warmup(self) -> None:
        # 预热模型
        pass

# 使用自定义检测器
detector = MyDetector("my_model.pt")
tracker_service = TrackerService(detector=detector)
```

---

## 项目结构

```
target-tracking/                 # Git 仓库根目录
├── mttrack/                      # 主包
│   ├── __init__.py              # 包初始化，导出主要 API
│   ├── domain/                   # 领域层（核心算法）
│   │   ├── __init__.py
│   │   ├── models.py            # 数据模型
│   │   ├── tracker.py           # 追踪器基类
│   │   ├── kalman.py            # 卡尔曼滤波器
│   │   ├── bytetrack.py         # ByteTrack 实现
│   │   ├── sort.py              # SORT 实现
│   │   ├── appearance.py        # 外观特征提取
│   │   ├── adaptive_trigger.py  # 自适应 VL 触发
│   │   └── association.py       # 多特征融合关联
│   │
│   ├── infrastructure/          # 基础设施层
│   │   ├── __init__.py
│   │   ├── detector.py          # YOLO 检测器
│   │   ├── vllm_client.py       # VLLM 客户端
│   │   └── video_io.py          # 视频读写
│   │
│   ├── service/                 # 服务层
│   │   ├── __init__.py
│   │   ├── tracker_service.py   # 标准追踪服务
│   │   ├── label_service.py     # VL 标签服务
│   │   └── enhanced_tracker_service.py  # 增强版追踪服务
│   │
│   └── annotators/               # 可视化
│       └── __init__.py          # 标注器实现
│
├── mttrack.py                   # CLI 入口脚本（直接运行）
├── tests/                       # 测试目录
│   ├── test_domain.py
│   ├── test_infrastructure.py
│   └── test_service.py
├── data/                        # 测试数据
│   └── test_multi_target_tracker_video.mp4
├── yolo/                        # YOLO 模型目录
│   └── yolo26x.pt
├── .env.example                 # 环境变量示例
├── pytest.ini                   # pytest 配置
└── README.md                    # 本文档
```

---

## 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_domain.py

# 带覆盖率
pytest --cov=mttrack --cov-report=html
```

---

## 常见问题

### Q1: VL 模型没有启用？

**A**: 检查环境变量是否设置：
```bash
echo $VLLM_BASE_URL  # 应该显示地址
```

### Q2: GPU 不可用？

**A**: 使用 CPU 模式：
```bash
mttrack --input video.mp4 --output result.mp4 --device cpu
```

### Q3: 追踪效果不好？

**A**: 尝试增强模式：
```bash
mttrack --input video.mp4 --output result.mp4 --enhanced
```

### Q4: 视频太大处理太慢？

**A**:
1. 使用 GPU：`--device cuda`
2. 降低分辨率：使用 FFmpeg 预处理
3. 跳过帧：修改代码中的迭代逻辑

---

## 许可证

Apache License 2.0 - 详见 [LICENSE](LICENSE)

---

## 致谢

- [ByteTrack](https://github.com/ifzhang/ByteTrack) - 多目标追踪
- [SORT](https://github.com/abewley/sort) - 简单在线实时追踪
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO 框架
