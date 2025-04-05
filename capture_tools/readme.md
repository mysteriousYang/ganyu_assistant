# 游戏自动化系统使用指南

## 系统概述

这个游戏自动化系统可以录制和回放键盘与鼠标操作，特别是针对包含加载屏幕的游戏。系统会自动检测游戏的加载屏幕，并在加载结束时设置检查点，确保录制和回放操作正确对齐。

系统主要由以下几个部分组成：
- **输入录制器**：记录键盘和鼠标操作
- **输入播放器**：回放录制的操作
- **屏幕检查点检测器**：检测游戏加载屏幕
- **完整自动化系统**：将上述组件集成在一起

## 安装要求

在使用此系统前，需要安装以下Python库：

```bash
pip install pynput opencv-python numpy mss keyboard pywin32
```

## 使用方法

### 1. 创建加载屏幕参考图像

首先，需要捕获游戏加载屏幕的参考图像，以便系统能够识别加载状态：

```bash
python complete_game_automation.py capture
```

运行此命令后，将游戏切换到加载屏幕，然后按空格键捕获参考图像。参考图像将保存在 `recorded_actions` 目录中。

### 2. 录制模式

使用以下命令开始录制游戏操作：

```bash
python complete_game_automation.py record --reference recorded_actions/loading_reference.png
```

系统会自动检测加载屏幕并设置检查点。录制的操作序列将保存在 `recorded_actions` 目录中。

**停止录制**：按下 `Ctrl+Shift+Esc` 组合键。

### 3. 回放模式

使用以下命令回放先前录制的操作：

```bash
python complete_game_automation.py playback --reference recorded_actions/loading_reference.png
```

系统会自动检测加载屏幕并在加载结束时继续播放下一个序列。

**停止回放**：按下 `Ctrl+Shift+Esc` 组合键。

### 4. 自定义选项

可以通过命令行参数自定义系统行为：

- `--dir`, `-d`：指定录制文件的目录（默认：`recorded_actions`）
- `--monitor`, `-m`：指定要监控的显示器编号（默认：`1`，即主显示器）
- `--reference`, `-r`：指定加载屏幕参考图像的路径

例如：
```bash
python complete_game_automation.py record --dir my_recordings --monitor 2 --reference my_reference.png
```

## 工作原理

1. **录制模式**下：
   - 系统记录所有键盘和鼠标操作，包括时间戳
   - 屏幕检测器监控游戏画面
   - 当检测到加载屏幕结束时，系统设置检查点，保存当前序列并开始新序列

2. **回放模式**下：
   - 系统播放第一个操作序列
   - 屏幕检测器监控游戏画面
   - 序列播放完成后，系统等待检测到加载屏幕结束
   - 加载结束后，系统继续播放下一个序列

## 自定义检测逻辑

如果默认的检测方法不适用于你的游戏，可以修改 `screen_checkpoint_detector.py` 文件中的 `is_loading_screen` 方法，实现更适合你的游戏的检测逻辑。

可以考虑以下几种检测方法：
- 模板匹配（使用参考图像）
- 特定区域颜色检测（如加载条）
- OCR文本识别（如"Loading..."文字）
- 动作检测（加载屏幕通常移动较少）

## 故障排除

1. **检测不准确**：尝试调整阈值参数或使用更好的参考图像
2. **播放不同步**：确保在相同的游戏状态和设置下录制和回放
3. **键盘/鼠标事件未触发**：检查是否以管理员权限运行程序

## 进阶用法

### 集成到其他Python程序

可以将此系统集成到其他Python程序中：

```python
from complete_game_automation import CompleteGameAutomation

# 创建自动化系统
automation = CompleteGameAutomation(
    output_dir="my_recordings",
    reference_image="my_reference.png"
)

# 开始录制
automation.start_recording()

# 或开始回放
# automation.start_playback()

# 停止
automation.stop()
```

### 自定义检查点逻辑

除了基于屏幕检测的检查点外，还可以实现其他检查点策略，如基于内存读取、日志文件分析等。扩展 `CompleteGameAutomation` 类并实现自定义检查点逻辑即可。