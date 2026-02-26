# MTTrack

多目标实时追踪系统，支持结合视觉语言模型进行目标分类

## 功能特性

- **多种追踪算法**: 支持 ByteTrack 和 SORT 算法
- **YOLO 集成**: 支持任意 YOLO 模型 (YOLOv8, YOLOv10 等)
- **VL 分类**: 可选的视觉语言模型，用于增强目标分类
- **模块化设计**: 清晰的领域层、服务层、基础设施层分离
- **视频读写**: 便捷的视频读写工具

## 环境要求

- Python >= 3.8
- ultralytics (用于 YOLO)
- openai (用于 VLLM)
- numpy
- opencv-python
- scipy

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/mttrack.git
cd mttrack

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 基本用法

```python
from mttrack import TrackerService, YoloDetector, TrackingAnnotator, VideoReader, VideoWriter

# 初始化
detector = YoloDetector("yolo26x.pt")
tracker_service = TrackerService(detector, tracker_type="bytetrack")
annotator = TrackingAnnotator()

# 处理视频
with VideoReader("input.mp4") as reader:
    with VideoWriter("output.mp4", fps=reader.fps) as writer:
        for frame_id, frame in reader:
            result = tracker_service.process_frame(frame)
            annotated = annotator.annotate(frame, result.tracks)
            writer.write(annotated)
```

### 命令行用法

```bash
python mttrack.py --input ./data/test_multi_target_tracker_video.mp4 --output ./out/result.mp4
```

### 启用 VL 分类

```bash
# 设置环境变量
export VLLM_BASE_URL="http://your-vllm-server:port"
export VLLM_API_KEY="your-api-key"
export VLLM_MODEL="/models/your-vl-model"

# 运行并启用 VL 分类
python mttrack.py --input video.mp4 --output result.mp4 --enable-vl
```

## 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `VLLM_BASE_URL` | VLLM API 地址 | `http://10.132.19.82:50100` |
| `VLLM_API_KEY` | VLLM API 密钥 | - |
| `VLLM_MODEL` | VLLM 模型名称 | `/models/Qwen/Qwen3-VL-8B-Instruct` |

### 命令行参数

```
--input, -i          输入视频路径（必需）
--output, -o         输出视频路径（必需）
--tracker            追踪器类型: bytetrack 或 sort（默认: bytetrack）
--yolo-model         YOLO 模型路径（默认: ./yolo/yolo26x.pt）
--confidence         检测置信度阈值（默认: 0.25）
--device             YOLO 设备（默认: cuda）
--enable-vl          启用 VL 分类（默认: True）
--vl-interval        VL 分类间隔帧数（默认: 30）
--vl-timeout         VL API 超时秒数（默认: 30）
--show-fps           在输出视频显示 FPS
```

## 项目结构

```
mttrack/
├── mttrack/
│   ├── domain/           # 核心追踪算法
│   │   ├── models.py     # 数据模型
│   │   ├── tracker.py    # 追踪器基类
│   │   ├── kalman.py     # 卡尔曼滤波器
│   │   ├── bytetrack.py  # ByteTrack 实现
│   │   └── sort.py       # SORT 实现
│   ├── infrastructure/   # 外部集成
│   │   ├── detector.py   # YOLO 检测器
│   │   ├── vllm_client.py # VLLM 客户端
│   │   └── video_io.py   # 视频读写
│   ├── service/          # 业务逻辑
│   │   ├── tracker_service.py
│   │   └── label_service.py
│   └── annotators/       # 可视化
├── tests/                # 单元测试
├── mttrack.py            # 主入口
└── README.md
```

## 架构设计

项目采用分层架构：

- **领域层 (Domain)**: 核心追踪算法（ByteTrack、SORT）和数据模型
- **基础设施层 (Infrastructure)**: 外部集成（YOLO、VLLM、视频读写）
- **服务层 (Service)**: 业务逻辑编排
- **接口层 (Interface)**: 命令行和可视化

## 测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_domain.py

# 带覆盖率运行
pytest --cov=mttrack
```

## 许可证

Apache License 2.0 - 详见 [LICENSE](LICENSE)

## 致谢

- [ByteTrack](https://github.com/ifzhang/ByteTrack) - 多目标追踪
- [SORT](https://github.com/abewley/sort) - 简单在线实时追踪
- [Supervision](https://github.com/roboflow/supervision) - 计算机视觉工具
