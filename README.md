# RDKS100 Drowning Detection

基于 **地平线 RDK S100P（旭日 5）** 的实时溺水检测系统。支持双路 RTSP 摄像头输入，利用 BPU NPU 硬件加速进行目标检测、姿态估计、语义分割、目标追踪和异常检测。

**项目博客**: http://www.kzzxisgod.top/index.php/2026/01/23/rdks100p-moxingbushuquanliucheng/

---

## 目录

1. [环境要求](#1-环境要求)
2. [快速开始](#2-快速开始)
3. [项目架构](#3-项目架构)
4. [模块详解](#4-模块详解)
5. [数据流](#5-数据流)
6. [测试](#6-测试)
7. [配置文件](#7-配置文件)
8. [模型列表](#8-模型列表)
9. [添加新模型](#9-添加新模型)
10. [常见问题](#10-常见问题)

---

## 1. 环境要求

### 硬件平台

- **开发板**: 地平线 RDK S100P（旭日 5，ARM64 aarch64）
- **NPU**: 地平线 BPU（Bayesian Processing Unit）
- **摄像头**: RTSP 网络摄像头 / 海康威视 IP 摄像头
- **存储**: USB 存储设备用于录像（可选）

### 软件依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| CMake | >= 3.10 | 构建系统 |
| OpenCV | 4.x | 图像处理 |
| Qt5/Qt6 | 5.x/6.x | GUI 框架（自动检测） |
| Eigen3 | 3.x | 线性代数（ByteTrack/BotSort 使用） |
| FFmpeg | 4.x | 软件解码（libavformat, libavcodec, libavutil） |
| OpenMP | - | 并行加速 |
| plog | header-only | 日志库（`3rd-party/plog/include`） |
| cereal | header-only | JSON 序列化（`3rd-party/cereal/include`） |
| HIK SDK | V6.1.9.45 | 海康摄像头 SDK（通过 `cmake/FindHIKSDK.cmake` 查找） |
| Horizon BPU SDK | - | `/usr/hobot/`, `/usr/include/hobot/dnn/` |
| Horizon sp_codec | - | 硬件编解码 API |
| hobot_utils | - | 从 `/app/cdev_demo/bpu/utils/src/` 编译的共享库，封装 BPU 预处理/后处理/多媒体工具 |
| gflags | - | 命令行参数解析 |

### 非标准路径

CMakeLists.txt 中定义了以下路径：

```cmake
# 地平线 SDK
HOBOT_INCLUDE_DIRS: /usr/hobot/include, /usr/include/hobot/dnn, /usr/include/hobot, /app/cdev_demo/bpu/utils/inc
HOBOT_LINK_LIBS: dnn hbucp gflags fmt spcdev
HOBOT_LIB_DIRS: /usr/hobot/lib /usr/lib

# hobot_utils（编译为 libhobot_utils.so）
/usr/hobot/lib/libhobot_utils.so

# 第三方库
cereal: /home/sunrise/Desktop/3rd-party/cereal/include
plog: /home/sunrise/Desktop/3rd-party/plog/include
HIKSDK: /home/sunrise/Desktop/HCNetSDKV6.1.9.45_build20220902_ArmLinux64_ZH
```

> **注意**: 因为是 aarch64 平台，所有依赖（OpenCV, Qt5 等）都须是 ARM64 版本。不可直接在 x86 桌面编译运行。

---

## 2. 快速开始

### 构建

```bash
make build
# 等价于:
# mkdir build && cd build && cmake .. && make -j4
```

### 运行

```bash
make run
# 构建后执行: sudo build/src/RDKS100_Drowning
# （需要 sudo 权限访问硬件解码器）
```

### 清理

```bash
make clean          # 删除 build/
make clean-all      # 删除 build/ + 测试输出 + 运行时数据（截图、录像、日志）
make data-clean     # 仅清理运行时数据
```

### 项目信息

```bash
make info           # 显示项目和可用测试列表
make help           # 显示帮助
make debug          # 调试信息（构建状态检查）
```

### 更多 Makefile 目标

```bash
make rebuild        # 清理后重新构建
make install        # 构建并安装（sudo make install）
make force-clean    # 强制清理运行时数据（sudo）
make dev-setup      # 打印开发环境依赖提示
```

---

## 3. 项目架构

系统是一个 **Qt5 双摄像头 GUI 应用**，采用**多模块共享库**的架构。共 8 个动态库，链接到一个可执行文件中。

```
┌──────────────────────────────────────────────────────────────┐
│                   RDKS100_Drowning (可执行文件)                │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ lib_ui   │ lib_worker│ lib_logic│lib_inference│  lib_algorithm │
│ (Qt GUI) │ (多线程)  │ (业务逻辑)│ (推理引擎) │ (算法实现)       │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│ lib_sensor (图像采集: RTSP/HIK)   │ lib_common (工具库) │ lib_base (基础) │
└──────────────────────────────────────────────────────────────┘
```

### 库依赖关系

| 库名 | 路径 | 类型 | 说明 | 链接依赖 |
|------|------|------|------|---------|
| `lib_base` | `src/base/` | INTERFACE | `ThreadPool.h`, `ThreadSafeQueue.h`（头文件库） | 无 |
| `lib_common` | `src/common/` | SHARED | `ImageSaver`, `MatPool`, `Config.h`, `PlogInitializer` | OpenCV |
| `lib_sensor` | `src/sensor/` | SHARED | `ImageSensor`(基类), `RTSPCamera`, `HikCamera` | OpenCV, HIKSDK, lib_base, lib_common |
| `lib_algorithm` | `src/algorithm/` | SHARED | yolo / yolo_pose / yolo_seg / bytetrack / botsort / SAHI | OpenCV, Eigen3, hobot_utils, OpenMP |
| `lib_inference` | `src/Inference/` | SHARED | `BaseInfer` → `Yolo11Infer` / `YoloPose` / `YoloSeg` / `Patchcore` | OpenCV, lib_algorithm, lib_common, lib_sensor, lib_base |
| `lib_logic` | `src/Logic/` | SHARED | `LogicBase` → `DrowningUnderSurface` / `DrowningState` | OpenCV, lib_common, lib_inference |
| `lib_worker` | `src/Worker/` | SHARED | `DetectionCoordinator` / `VideoCaptureManager` / `AIInferenceManager` / `RecordingManager` | OpenCV, Qt::Widgets, lib_common, lib_inference, lib_sensor, lib_logic, lib_base, lib_algorithm |
| `lib_ui` | `src/ui/` | SHARED | `MainWindow`（Qt5/Qt6 主窗口） | OpenCV, Qt::Widgets, lib_base, lib_algorithm, lib_common, lib_sensor, lib_inference, lib_logic, lib_worker |

### 依赖图（简化）

```
lib_ui → lib_worker → lib_inference → lib_algorithm → OpenCV, Eigen3, hobot_utils
         │            │              lib_common     → OpenCV
         │            │              lib_base
         │            lib_sensor → lib_common
         │                       → lib_base
         │            lib_logic → lib_common
         │                      → lib_inference
         lib_common, lib_base, lib_algorithm
```

### 外部依赖

```
hobot_utils (编译自 /app/cdev_demo/bpu/utils/src/)
  └── OpenCV, HIKSDK, HOBOT_LINK_LIBS (dnn, hbucp, gflags, fmt, spcdev), FFMPEG
```

---

## 4. 模块详解

### 4.1 lib_base — 基础设施

头文件库（INTERFACE），只包含头文件，不产生 .so。

#### `ThreadSafeQueue<T>` — 线程安全队列

- **位置**: `src/base/ThreadSafeQueue.h`
- **功能**: 有界线程安全生产者-消费者队列
- **特性**:
  - 可配置最大容量（`max_size`，默认 `SIZE_MAX`）
  - 满时丢弃最旧元素（类似环形缓冲区行为）
  - 提供阻塞（`dequeue`）和非阻塞（`dequeue_nonblocking`）出队
  - `abort()` 唤醒所有阻塞线程
- **使用场景**: 帧队列、推理队列、截图队列、录像队列

#### `ThreadPool` — 线程池

- **位置**: `src/base/ThreadPool.h`
- **功能**: 通用任务并行化线程池
- **接口**: `enqueue(F&& f, Args&&... args)` 返回 `std::future<return_type>`

---

### 4.2 hobot_utils — BPU 工具库

- **位置**: 由 CMake 从 `/app/cdev_demo/bpu/utils/src/` 编译的共享库（`libhobot_utils.so`）
- **源文件**: `common_utils.cc`, `postprocess_utils.cc`, `preprocess_utils.cc`, `multimedia_utils.cc`
- **功能**: 封装地平线 BPU 的通用预处理（BGR→NV12）、后处理（框解码、NMS）、多媒体工具
- **链接**: `lib_algorithm` 链接此库

### 4.3 lib_common — 公共工具库

#### `PlogInitializer` — 日志初始化器

- **位置**: `src/common/PlogInitializer.h`
- **功能**: 线程安全的 plog 单例初始化器，确保日志系统只初始化一次
- **初始化**（在 `main()` 中调用一次）:
  ```cpp
  PlogInitializer::getInstance().init(plog::verbose);
  ```
- **宏 `ENSURE_PLOG_INITIALIZED()`**: 库代码（可能先于 main 运行）中调用，如果未初始化则自动以 `plog::info` 级别初始化
- **日志级别**: `PLOGV` / `PLOGD` / `PLOGI` / `PLOGW` / `PLOGE` / `PLOGF`
- **Appender**: 彩色控制台输出（`ColorConsoleAppender`）

#### `Config.h` — 配置结构

- **位置**: `src/common/Config.h`
- **功能**: 使用 cereal 库进行 JSON 序列化的配置结构体
- **`HikConfig`**: 海康摄像头配置（ip, port, user, pass）

> 实际配置从 `configs/app_config.json` 文件加载，非通过 Config.h 中的结构体加载。后者部分已注释弃用。

#### `ImageSaver` — 批量图像保存

- **位置**: `src/common/ImageSaver.h/.cpp`
- **功能**: 批量保存图像到以时间命名的子文件夹中
- **使用**:
  ```cpp
  ImageSaver saver("./test_captures");
  saver.addImage(mat, "frame_001");
  saver.flush(); // 保存到 ./test_captures/2026-01-23_14-30-00/frame_001.jpg
  ```

#### `MatPool` — 图像内存池

- **位置**: `src/common/MatPool.h/.cpp`
- **功能**: 内存池重用 `cv::Mat`，减少频繁分配/释放带来的开销
- **`MatPool`**: 管理指定尺寸/类型的 `cv::Mat` 对象池
- **`MatPoolManager`**: 单例管理器，维护多个 `MatPool` 实例（按 `(cv::Size, type)` 键区分）
- **统计**: 跟踪缓存命中率、内存使用量
- **使用**: `MatPoolManager::getPool(cv::Size(1920,1080), CV_8UC3).getMat()`

---

### 4.4 lib_sensor — 图像传感器

#### `ImageSensor` — 采集器基类

- **位置**: `src/sensor/ImageSensor.h/.cpp`
- **功能**: 所有采集设备的抽象基类
- **核心方法**: `start()`, `stop()`, `enqueueData()`, `getData()`（阻塞）, `getDataNoBlock()`（非阻塞）, `getLastestFrame()`, `clear()`, `setMatPool()`
- **队列策略**:
  - `is_full_drop == true`: 队列满时丢弃新帧（保证实时性）
  - `is_full_drop == false`: 队列满时丢弃最老帧（保证完整性）
- **纯虚函数**: `dataCollectionLoop()` — 子类实现具体采集逻辑

#### `RTSPCamera` — RTSP 摄像头（核心）

- **位置**: `src/sensor/RTSPCamera.h/.cpp`
- **功能**: 通过 RTSP 协议获取视频流，支持硬件解码（地平线 VPU）和软件解码（FFmpeg）
- **解码模式**:
  - `HARDWARE`（默认）: 使用地平线 `sp_codec` API 进行硬件解码，输出 NV12 → 转换为 BGR
  - `SOFTWARE`: 使用 `cv::VideoCapture` + FFmpeg 进行 CPU 软解
- **功能接口**:
  - `captureSnapshot(path)`: 截图保存为 JPEG
  - `startRecording(path)` / `stopRecording()`: 视频录制（`cv::VideoWriter` + MJPG 编码）
- **帧缓冲区**: 预分配 `yuv_frame_`（NV12）和 `bgr_frame_`（BGR），从 MatPoolManager 获取以避免内存抖动
- **构造参数**:
  ```cpp
  RTSPCamera(url, width, height, queue_max_length=10, capture_interval_ms=0, is_full_drop=true, decode_mode=HARDWARE);
  ```

#### `HikCamera` — 海康威视摄像头

- **位置**: `src/sensor/HikCamera.h/.cpp`
- **功能**: 使用海康威视 SDK 进行拉流和解码
- **输出**: 裸 H.264 流数据包（不直接解码为 BGR），需要外部解码器处理
- **SDK 回调**: `NET_DVR_RealPlay_V40` + 静态 `RealDataCallBack` 函数
- **注意**: 海康输出的数据包带有海康私有头，地平线硬解可能不兼容。建议改用 RTSPCamera（直接传入海康摄像头的 RTSP 地址），更加简单可靠。

---

### 4.5 lib_algorithm — 算法实现

#### YOLO11 — 目标检测

- **位置**: `src/algorithm/yolo/`
- **类**: `YOLO11`
- **生命週期**:
  1. 构造函数: API 加载 `.hbm` 模型文件 → 获取输入输出张量属性 → 分配张量内存
  2. `pre_process(bgr_mat)`: Letterbox 缩放 → BGR → NV12
  3. `infer()`: 创建推理任务（`hbDNNInferV2`）→ BPU 调度（`hbUCPSubmitTask`）→ 等待完成 → 刷新缓存
  4. `post_process(score_thres, nms_thres, w, h)`: DFL（Distribution Focal Loss）框解码 → 类得分过滤 → NMS → 坐标回映射到原图
  5. 析构函数: 释放张量内存和模型句柄

#### YOLO11-Pose — 姿态估计

- **位置**: `src/algorithm/yolo_pose/`
- **类**: `YOLO11_Pose`
- **输出**: 检测框 + 17 个关键点（COCO 标准格式）
- **后处理**: 从 51 通道（17 KP × 3: dx, dy, score）解码关键点 → 激活函数（sigmoid）→ 类 NMS 保持关键点对齐

#### YOLO11-Seg — 实例分割

- **位置**: `src/algorithm/yolo_seg/`
- **类**: `YOLO11_Seg`
- **输出**: 检测框 + 二进制掩码（通过 MCES 特征向量 + 原型掩码生成）

#### ByteTrack — 多目标追踪

- **位置**: `src/algorithm/bytetrack/`
- **类**: `BYTETracker`
- **算法**: 两阶段关联（高置信度 + 低置信度检测），使用卡尔曼滤波 + 匈牙利匹配
- **默认阈值**: `track_thresh=0.5`, `high_thresh=0.6`, `match_thresh=0.8`
- **辅助类**: `STrack`（追踪状态）、`KalmanFilter`、`lapjv`（LAPJV 求解器）

#### BoT-SORT — 多目标追踪（带相机运动补偿）

- **位置**: `src/algorithm/botsort/`
- **类**: `BotSort`, `BotSortAdapter`
- **特性**: 使用 ORB 特征 + `estimateAffinePartial2D` 进行相机运动补偿（CMC）
- **三阶段关联**: 活跃轨迹 vs 高置信 → 未配活跃 vs 低置信 → 丢失轨迹 vs 未配高置信

#### SAHI — 大图切片推理

- **位置**: `src/algorithm/SAHI/`
- **类**: `SAHI`
- **功能**: 将大图像切分为重叠的切片进行推理，适用于小目标检测
- **方法**: `calculateSliceRegions()` 生成切片区域，`mapToOriginal()` 将切片坐标映射回原图

---

### 4.6 lib_inference — 推理引擎层

采用 **工厂模式**，所有推理引擎继承 `BaseInfer`。

#### `BaseInfer`（命名空间 `Inf`）— 推理基类

- **位置**: `src/Inference/BaseInfer.h`
- **数据结构**:
  - `Inf::Keypoint`: `x`, `y`, `score`
  - `Inf::Detection`: `rect`（cv::Rect）, `score`, `class_id`, `track_id`（默认 -1）, `keypoints`（vector）
- **纯虚接口**: `init()`, `run()`, `cleanup()`, `draw()`, `setLabels()`, `getLabels()`

#### `Yolo11Infer` — YOLO 检测引擎

- **位置**: `src/Inference/Yolo11Infer.h/.cpp`
- **功能**: YOLO11 检测 + ByteTrack 追踪的集成引擎
- **`run()` 流程**:
  1. `pre_process()` → `infer()` → `post_process(0.25, 0.7, w, h)`
  2. 将 Detection 转换为 Object → `m_tracker->update(objects)`
  3. 对每个轨迹，通过 IoU（阈值 0.5）找到最佳检测并恢复 class_id
  4. 返回带 `track_id` 的 `Inf::Detection` 列表
- **`draw()`**: 自适应字体缩放（`frame.cols / 800.0`），彩色边框 + 标签（"ID:class_name"）

#### `YoloPose` — 姿态估计引擎

- **位置**: `src/Inference/YoloPose.h/.cpp`
- **功能**: YOLO11-Pose 姿态估计 + ByteTrack 追踪
- **`draw()`**: 绘制 COCO 骨架（17 关键点，16 骨骼连接），关键点双层颜色圆圈

#### `YoloSeg` — 语义分割引擎

- **位置**: `src/Inference/YoloSeg.h/.cpp`
- **功能**: YOLO11-Seg 语义分割 + ByteTrack 追踪
- **`run()` 流程**: pre_process → infer → post_process，通过 MCES 特征向量 + 原型掩码生成检测框和掩码
- **`draw()`**: 绘制检测框 + 半透明颜色掩码叠加

#### `Patchcore` — 异常检测引擎

- **位置**: `src/Inference/Patchcore.h/.cpp`
- **功能**: Patchcore 异常检测模型，用于"进水检测"
- **特点**: 直接使用地平线 BPU SDK API（`hbDNN*` / `hbUCP*`），不使用 `YOLO11` 封装
- **`run()` 流程**:
  1. 预处理: BGR → RGB → 缩放 224×224 → RGB → YUV_I420 → 手动构造 NV12
  2. 推理: 直接通过 BPU API 执行
  3. 后处理: 读取热力图和全局异常分数
  4. 返回单 `Detection`（得分 > 50.0 → class_id=1 异常）
- **`draw()`**: 叠加 JET 伪彩色热力图，显示 "ALARM: LEAK" 或 "Normal"

---

### 4.7 lib_logic — 业务逻辑层

采用 **策略模式**，所有逻辑引擎继承 `LogicBase`。

#### `LogicBase` — 逻辑基类

- **位置**: `src/Logic/LogicBase.h`
- **方法**: `virtual void process(cv::Mat& frame, const vector<Inf::Detection>& results) = 0`

#### `DrowningUnderSurface` — 溺水检测（YOLO 模型）

- **位置**: `src/Logic/DrowningUnderSurface.h/.cpp`
- **适用模型**: `"YOLO"` 类型（`Under_Surface_v1.hbm`，标签 \[person at surface, person underwater\]）
- **逻辑**:
  - 按 `track_id` 追踪历史轨迹（最多 30 帧）
  - 如果 `class_id == 1`（水下）且 30 帧位移 < 400px → 判定为长时间停留水下
  - 当 10 帧持续判定 → 触发 **DROWNING** 告警（红色警示）
- **告警绘制**: 红框 + 全屏红色横幅 "ALARM: DROWNING!"

#### `DrowningState` — 溺水检测（游泳者模型）

- **位置**: `src/Logic/DrowningState.h/.cpp`
- **适用模型**: `"SWIMMER"` 类型（`drowning_TwoSelect.hbm`，标签 \[drowning, swimming\]）
- **逻辑**: 与 `DrowningUnderSurface` 结构相同，但判断条件为 `class_id == 0`（溺水类别）

---

### 4.8 lib_worker — 多线程工作层

核心的多线程生产者-消费者框架，每条摄像头流由一个 `DetectionCoordinator` 管理。

#### `DetectionCoordinator` — 检测协调器

- **位置**: `src/Worker/DetectionCoordinator.h/.cpp`
- **功能**: 协调 `VideoCaptureManager`、`AIInferenceManager`、`RecordingManager` 三个子管理器
- **职责**:
  - `start()`: 启动采集和推理
  - `stop()`: 停止所有子管理器
  - `switchModel(type, path)`: 切换推理引擎和业务逻辑
  - `startRecording()` / `stopRecording()`: 控制双路录制
  - `triggerSnapshot()`: 触发截图
- **信号**: `frameReady` → UI 显示，`inferenceFrameReady` → UI 显示，`snapshotReady` → 截图保存

#### `VideoCaptureManager` — 视频采集管理器

- **位置**: `src/Worker/VideoCaptureManager.h/.cpp`
- **功能**: 管理摄像头生命周期和帧分发
- **线程**:
  - `captureLoop()`: 采集主循环，从 `RTSPCamera.getData()` 获取帧 → 发送到推理队列和 UI
  - `snapshotLoop()`: 独立线程处理截图
- **暂停机制**: 暂停时睡眠 30ms；截图时临时恢复摄像头获取一帧

#### `AIInferenceManager` — AI 推理管理器

- **位置**: `src/Worker/AIInferenceManager.h/.cpp`
- **功能**: 管理推理引擎和业务逻辑执行
- **`switchModel()` 实现**:
  - `"YOLO"`: 创建 `Yolo11Infer`（COCO 80 类）+ `DrowningUnderSurface` 逻辑
  - `"SWIMMER"`: 创建 `Yolo11Infer`（\[drowning, swimming\]）+ `DrowningState` 逻辑
  - `"YOLOSEG"`: 创建 `YoloSeg`（语义分割）+ 无逻辑层
  - `"Patchcore"`: 创建 `Patchcore` 模型（无逻辑层）
- **引擎热切换**: `switchModel()` 在 `m_engineMutex` 保护下原子交换引擎指针，推理线程不停止，下一帧自动用新引擎运行
- **`inferenceLoop()`**: 队列取帧 → `m_inferEngine->run()` → `m_inferEngine->draw()` → `m_currentLogic->process()` → 发射帧到 UI 和录像
- **背压控制**: `submitFrame()` 限制 `size() > 2` 时清空队列，确保实时性

#### `RecordingManager` — 录制管理器

- **位置**: `src/Worker/RecordingManager.h/.cpp`
- **功能**: 双路同步录制（原始画面 + 推理画面）
- **特点**:
  - 双独立录制管线（原始 + 推理），各有独立队列、线程、`VideoWriter`、互斥锁
  - 录制前测量实际 FPS（缓冲前 10 帧后，3.5 秒内测量帧到达速率）
  - FPS 限幅 5-30，用于初始化 VideoWriter
  - **编码**: Motion JPEG（`M','J','P','G'` 四字符编码）
  - **输出路径**: 硬编码为 `/media/UBUNTU 18_0/records/`（USB 存储）
  - **高性能模式**: FPS 上限设为 20 而非实际 FPS

---

### 4.9 lib_ui — 用户界面

#### `MainWindow` — 主窗口

- **位置**: `src/ui/mainwindow.h/.cpp`
- **功能**: Qt5 双摄像头 GUI 界面
- **控件（来自 `mainwindow.ui`）**:
  - 4 个图像显示区域（`labelOriginal_1/2`, `labelProcessed_1/2`）
  - 按钮: 开启/关闭/暂停/截图
  - 模型选择 ComboBox（`comboBoxModels_1/2`）：溺水检测 / 进水检测 / 游泳检测
  - 确认按钮（`btnConfirm`）：切换模型
  - 录制开关（`radioStartRecord` / `radioStopRecord`）
- **模型切换**（`on_btnConfirm_clicked()`）通过 `DetectionCoordinator::switchModel()` 实现热切换（引擎在互斥锁外初始化后原子交换，推理线程不停止）:
  - 溺水检测 → `"YOLO"` 模型（Cam1: `MyMot_Big.hbm`, Cam2: `YOLO11s.hbm`），逻辑 `DrowningUnderSurface`
  - 游泳检测 → `"SWIMMER"` 模型（`drowning_TwoSelect.hbm`），逻辑 `DrowningState`
  - 进水检测 → `"YOLOSEG"` 模型（`Water_seg.hbm`），无语义逻辑层
  - （`"Patchcore"` 模式注释待用）
- **录制联动**: 模型切换确认后自动开启高性能录制（`setRecordingPerformanceMode(true)`，FPS 上限 20）
- **UI 状态刷新**: 500ms 定时器 → `updateButtonStates()`

---

## 5. 数据流

### 单路摄像头完整数据流

```
RTSPCamera (硬件解码 NV12 → BGR)
    ↓ getData()
VideoCaptureManager::captureLoop()
    ↓
┌─ emit frameReady(frame.clone())
│   ├─→ [信号] MainWindow::updateUI() → labelOriginal 显示
│   ├─→ RecordingManager::submitOriginalFrame() [录制时]
│   └─→ AIInferenceManager::submitFrame(frame)
│          ↓ (队列限制 ≤ 2, 超限丢老帧)
│       AIInferenceManager::inferenceLoop()
│          ↓ 加锁
│       m_inferEngine->run(frame)
│          ├─ pre_process() → infer() → post_process()
│          └─ m_tracker->update() [For YOLO/Pose engines]
│          ↓
│       m_inferEngine->draw(frame, results)
│          ↓
│       m_currentLogic->process(frame, results) [业务逻辑, 可选]
│          ↓
│       emit inferenceFrameReady(frame)
│          ├─→ MainWindow::updateUI() → labelProcessed 显示
│          └─→ RecordingManager::submitInferenceFrame() [录制时]
```

### 线程模型

| 线程 | 数量 | 归属 | 功能 |
|------|------|------|------|
| UI 主线程 | 1 | Qt 事件循环 | 信号/槽处理、界面更新 |
| 采集线程 | 2 | VideoCaptureManager | 从摄像头获取帧 |
| 截图线程 | 2 | VideoCaptureManager | 保存截图 |
| 推理线程 | 2 | AIInferenceManager | 模型推理 + 逻辑处理 |
| 录制线程 | 4 | RecordingManager | 输出 AVI 文件 |

总计 **11 个线程**（包含主线程）。

### 背压控制

```cpp
// AIInferenceManager::submitFrame()
if (m_inferenceQueue.size() > 2) {
    // 推理处理不过来时，丢弃老帧
    m_inferenceQueue.clear();
}
m_inferenceQueue.enqueue(frame.clone());
```

当推理速度跟不上采集速度时，系统自动丢弃队列中的老帧以保证实时性。

---

## 6. 测试

### 运行测试

```bash
make test-all                    # 全部测试
make test-basic                  # 核心功能测试
make test-advanced               # 模型专项测试
make test-<name>                 # 指定某个测试（详见下方列表）
```

Makefile 提供多种特定测试快捷目标，涵盖追踪、检测、分割、姿态估计等：

### 测试清单

| 测试 | 类别 | 需要 sudo | 说明 |
|------|------|-----------|------|
| `test_Bytetrack` | 基础 | × | ByteTrack 追踪 |
| `test_BotSort` | 基础 | × | BoT-SORT 追踪 |
| `test_Detection` | 基础 | × | 目标检测 |
| `test_bytetrack_detection` | 基础 | × | 追踪+检测集成 |
| `test_DetectionVideo` | 基础 | × | 视频目标检测 |
| `test_ImageSaver` | 基础 | × | 图像保存 |
| `test_Cereal_SDK` | 基础 | × | Cereal 序列化 |
| `test_Plog` | 基础 | × | Plog 日志 |
| `test_HikCamera` | 基础 | × | 海康摄像头 |
| `test_Show_HIK` | 基础 | × | 海康显示 |
| `test_rtsp` | 基础 | √ | RTSP 摄像头 |
| `test_RTSPRecord` | 基础 | √ | RTSP 录像 |
| `test_H264` | 高级 | × | H264 编解码 |
| `test_Pose` | 高级 | × | 姿态估计 |
| `test_Pose_img` | 高级 | × | 姿态估计（图像） |
| `test_pose_Base` | 高级 | × | 基础姿态 |
| `test_mmpose` | 高级 | × | MMPose 姿态估计 |
| `test_seg` | 高级 | × | 语义分割 |
| `test_seg_video` | 高级 | × | 语义分割（视频） |
| `test_seg_inference` | 高级 | × | 语义分割推理 |
| `test_deeplabv3+` | 高级 | × | DeepLabV3+ 分割 |
| `test_SAHI` | 高级 | × | 大图切片推理 |
| `test_Drowning` | 高级 | × | 溺水检测 |
| `test_swimmer` | 高级 | × | 游泳者检测 |
| `test_UnderOrSurface` | 高级 | × | 水下/水面状态 |
| `test_patchcore` | 高级 | × | 异常检测 |
| `test_patchcore_hpp` | 高级 | × | PatchCore 头文件测试 |
| `test_roughwaternet` | 高级 | × | 粗糙水体分类 |
| `test_finewaternet` | 高级 | × | 精细水体分类 |
| `test_HIKSDK` | 高级 | × | 海康 SDK |
| 共 30 个测试 | - | - | `test/test_*.cc` |

### 可用 Makefile 测试目标

| 命令 | 测试说明 |
|------|---------|
| `make test-bytetrack` | ByteTrack 追踪 |
| `make test-botsort` | BoT-SORT 追踪 |
| `make test-detection` | 目标检测 |
| `make test-bytetrack-detection` | ByteTrack 检测集成 |
| `make test-detection-video` | 视频目标检测 |
| `make test-imagesaver` | 图像保存 |
| `make test-rtsp` | RTSP 摄像头（需 sudo） |
| `make test-rtsp-record` | RTSP 录像（需 sudo） |
| `make test-h264` | H264 编解码 |
| `make test-pose` | 姿态估计 |
| `make test-pose-img` | 姿态估计（图像） |
| `make test-pose-base` | 基础姿态 |
| `make test-mmpose` | MMPose 姿态估计 |
| `make test-seg` | 语义分割 |
| `make test-seg-video` | 语义分割（视频） |
| `make test-seg-inference` | 语义分割推理 |
| `make test-deeplab` | DeepLabV3+ 分割 |
| `make test-sahi` | SAHI 大图切片推理 |
| `make test-drowning` | 溺水检测 |
| `make test-swimmer` | 游泳者检测 |
| `make test-under-surface` | 水下/水面状态检测 |
| `make test-patchcore` | PatchCore 异常检测 |
| `make test-patchcore-hpp` | PatchCore 头文件测试 |
| `make test-roughwaternet` | 粗糙水体分类 |
| `make test-finewaternet` | 精细水体分类 |
| `make test-hiksdk` | HIK SDK 测试 |
| `make test-hikcamera` | 海康摄像头 |
| `make test-show-hik` | 海康显示 |
| `make test-cereal` | Cereal 序列化 |
| `make test-plog` | Plog 日志 |

### 测试媒体文件

所有测试媒体存储在 `tem/` 目录（已被 `.gitignore` 忽略），支持 `.mp4`、`.jpg`、`.png`、`.h264` 格式。

---

## 7. 配置文件

### `configs/app_config.json`

```json
{
  "system": {
    "snapshot_dir": "./snapshots/",
    "video_dir": "./records/",
    "thread_pool_size": 4
  },
  "cameras": [
    {
      "id": 1,
      "name": "Cam1",
      "url": "rtsp://admin:password@192.168.127.15",
      "width": 1920,
      "height": 1080,
      "queue_max_length": 10,
      "capture_interval_ms": 0,
      "is_full_drop": false
    },
    {
      "id": 2,
      "name": "Cam2",
      "url": "rtsp://127.0.0.1/assets/swim_fixed.h264",
      "width": 1920,
      "height": 1080,
      "queue_max_length": 10,
      "capture_interval_ms": 0,
      "is_full_drop": false
    }
  ],
  "model_defines": {
    "drowning": {
      "type": "YOLO",
      "path": "/home/sunrise/Desktop/RDKS100_Drowning/models/Under_Surface_v1.hbm",
      "labels": ["person at surface", "person underwater"]
    },
    "swimmer": {
      "type": "SWIMMER",
      "path": "/home/sunrise/Desktop/RDKS100_Drowning/models/drowning_TwoSelect.hbm",
      "labels": ["drowning", "swimming"]
    },
    "patchcore": {
      "type": "Patchcore",
      "path": "/home/sunrise/Desktop/RDKS100_Drowning/models/patchcore.hbm"
    }
  }
}
```

### 模型类型说明

| type | 推理引擎 | 业务逻辑 | 标签格式 |
|------|---------|---------|---------|
| `"YOLO"` | `Yolo11Infer` (YOLO11 + ByteTrack) | `DrowningUnderSurface` | 任意标签列表（默认 COCO 80 类） |
| `"SWIMMER"` | `Yolo11Infer` (YOLO11 + ByteTrack) | `DrowningState` | \[drowning, swimming\] |
| `"YOLOSEG"` | `YoloSeg` (YOLO11-Seg + ByteTrack) | 无 | 任意标签列表（默认 `["water"]`） |
| `"Patchcore"` | `Patchcore` | 无 | 无需标签 |

---

## 8. 模型列表

`models/` 目录下包含 33 个 `.hbm` 模型文件。以下是完整列表：

### 检测模型

| 模型文件 | 说明 |
|---------|------|
| `Under_Surface_v1.hbm` | 水面/水下人员检测（默认溺水模型） |
| `new_UnderSurface_1920x1080.hbm` | 新版 1920×1080 水面/水下检测 |
| `UnderSurface0613.hbm` | 水面/水下检测（0613 版） |
| `YOLO11s.hbm` | YOLO11s 通用检测 |
| `YOLO11s_2560x1440.hbm` | YOLO11s 2560×1440 大分辨率检测 |
| `YOLO11s_4k.hbm` | YOLO11s 4K 超高清检测 |
| `ultralytics_YOLO.hbm` | Ultralytics YOLO 通用检测 |
| `MyMot_Big.hbm` | 大尺寸追踪检测模型 |
| `MyMot.hbm` | 追踪检测模型 |
| `yolo_mymot_big.hbm` | YOLO+MyMOT 大模型 |
| `yolo11n_detect_nashe_640x640_nv12.hbm` | 小模型 640×640 检测 |
| `LEAF-YOLO.hbm` | 小目标检测（树叶级别） |
| `LSOD-YOLO_11s_visdrone.hbm` | VisDrone 数据集小目标检测 |
| `s_visdrone.hbm` | VisDrone 检测 |
| `s_PC-YOLO_1920x1080_UnderSurface.hbm` | PC-YOLO 1920×1080 溺水检测 |
| `s_PC-YOLO_UnderSurface.hbm` | PC-YOLO 溺水检测 |
| `Zuoshiyan.hbm` | 实验用检测模型 |

### 溺水/游泳专用

| 模型文件 | 说明 |
|---------|------|
| `drowning_TwoSelect.hbm` | 二分类溺水识别（溺水/游泳） |
| `drowning.hbm` | 溺水检测 |
| `swimmer.hbm` | 游泳者检测 |

### 姿态估计

| 模型文件 | 说明 |
|---------|------|
| `YOLO11n-pose.hbm` | YOLO11n-pose 姿态估计 |
| `Hrnet.hbm` | HRNet 姿态估计 |

### 分割模型

| 模型文件 | 说明 |
|---------|------|
| `YOLO11n-seg.hbm` | YOLO11n-seg 实例分割 |
| `deeplabv3plus_efficientnetb0_1024x2048_nv12.hbm` | DeepLabV3+ 语义分割 |
| `deeplabv3plus_efficientnetm2_1024x2048_nv12.hbm` | DeepLabV3+ 语义分割 |
| `Water_seg.hbm` | 水体分割 |
| `Water_zuoshiyan.hbm` | 水体分割（实验版） |

### 异常检测

| 模型文件 | 说明 |
|---------|------|
| `patchcore.hbm` | PatchCore 异常检测（进水检测） |

### 水体分类

| 模型文件 | 说明 |
|---------|------|
| `rough_waternet.hbm` | 粗糙水体分类 |
| `fine_waternet.hbm` | 精细水体分类 |

### 追踪模型

| 模型文件 | 说明 |
|---------|------|
| `bytetrack_s.hbm` | ByteTrack 轻量版 |
| `yolo11s_mot.hbm` / `yolo11s_mot_1920.hbm` | YOLO+Tracking 联合模型 |

---

## 9. 添加新模型

为系统添加新模型需修改 3 个文件：

### 步骤

#### 1. 创建推理引擎子类

在 `src/Inference/` 中创建新的 `.h` 和 `.cpp`，继承 `Inf::BaseInfer`：

```cpp
class MyNewInfer : public Inf::BaseInfer {
    bool init(const std::string& model_path) override;
    std::vector<Inf::Detection> run(cv::Mat& frame) override;
    void cleanup() override;
    void draw(cv::Mat& frame, const std::vector<Inf::Detection>& results) override;
};
```

同时更新 `src/Inference/CMakeLists.txt` 添加新源文件。

#### 2. 注册到 AIInferenceManager（或 DetectionCoordinator）

在 `AIInferenceManager::switchModel()` 中添加新的分支（引擎在 mutex 外创建和初始化，再原子交换到成员中）：

```cpp
else if (type == "MY_MODEL") {
    auto engine = std::make_unique<Inf::MyNewInfer>();
    engine->setLabels(my_labels);
    nextEngine = std::move(engine);
    nextLogic = std::make_unique<MyNewLogic>();
}
```

不要在分支内直接赋值 `m_inferEngine`，引擎指针的交换由 `switchModel()` 统一在 `m_engineMutex` 保护下完成，推理线程不停止，下一帧自动启用新引擎。

#### 3. 更新 UI 下拉菜单

在 `MainWindow::on_btnConfirm_clicked()` 中添加 ComboBox 对应的模型选择：

```cpp
if (currentModel == "我的新模型") {
    // 选择模型文件并调用 switchModel
}
```

`mainwindow.ui` 中已有 `comboBoxModels_1` 和 `comboBoxModels_2`，可以在 Qt Designer 中添加新选项。

---

## 10. 常见问题

### Q: 为什么需要 sudo 运行？

硬件解码（`sp_codec`）和 BPU 设备文件需要 root 权限访问。

### Q: 是否可以在 x86 上编译？

不可以。项目依赖 Horizon BPU SDK（仅 aarch64 ARM 架构）。所有动态链接库（OpenCV、Qt5 等）都必须是 ARM64 版本。

### Q: HikCamera 无法解码显示？

海康 SDK 输出的 H.264 流包含私有头部数据，与地平线硬件解码器不兼容。建议改用 `RTSPCamera` 直接传入海康摄像头的 RTSP 地址以使用硬件解码。

### Q: 录制文件无法播放？

录制使用 Motion JPEG（MJPG）编码的 `.avi` 文件。如果无法播放，请确保播放器支持 MJPG。也可尝试使用 FFmpeg 转码：

```bash
ffmpeg -i input.avi -c:v libx264 output.mp4
```

### Q: 如何调整检测阈值？

检测阈值硬编码在推理引擎的 `post_process` 调用中：
- YOLO 检测: `post_process(0.25, 0.7, w, h)`（置信度 0.25，NMS IoU 0.7）
- ByteTrack: `track_thresh=0.5`, `high_thresh=0.6`, `match_thresh=0.8`

### Q: 程序内存占用过高？

`MatPool` 默认每个池最多维护 50 个 `cv::Mat`，可根据需要在 `MatPoolManager::getPool()` 中调整大小。

### Q: 点击停止录制时画面卡顿？

这是已知问题（见 `doc/项目的整体框架.md` 中的 TODO），由于录制线程停止时和采集线程存在同步开销。优化中。

---

## 参考

- [详细博客: RDKS100P 模型部署全流程](http://www.kzzxisgod.top/index.php/2026/01/23/rdks100p-moxingbushuquanliucheng/)
- [doc/项目的整体框架.md](doc/项目的整体框架.md) — 框架设计文档
- [doc/RDKS100P 模型部署.md](doc/RDKS100P%20模型部署.md) — 模型部署文档
- [AGENTS.md](AGENTS.md) — AI 助手开发指引（OpenCode 专用）
