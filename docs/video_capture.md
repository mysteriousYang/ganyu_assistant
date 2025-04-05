# 视频采集模块

该模块用于录制游戏视频录制与采集，可以同时采集游戏录像与按键序列

## 函数列表

- terminate_capture_check
- terminate_capture_press
- terminate_capture_release
- gaming_key_press
- gaming_key_release
- gaming_mouse_move
- gaming_mouse_click
- gaming_mouse_scroll
- capture_screen
- _check_admin
- _run_capture
- _elevate_and_continue
- run

## 全局变量


|变量名 | 类型 | 默认值/初始值 | 作用|
| -----| -----| ----- | ----- |
| _C_g_cap_key | str 常量 | 'p' | 用于终止录制的按键|
| _C_g_cap_func_key | str 常量 | 'alt' | 用于终止录制的控制按键 |
|_g_current_keys | set | 无 | 用于检测组合键，存储当前按下的按键|
|_g_key_press_events | list | 无 | 用于充当缓存，记录当前发生的按键按下数据|
|_g_key_release_events | list | 无 | 用于充当缓存，记录当前发生的按键松开数据|
|_g_mouse_move_events | list | 无 | 用于充当缓存，记录当前发生的鼠标移动数据|
|_g_mouse_click_events | list | 无 | 用于充当缓存，记录当前鼠标的按键数据|
|_g_mouse_scroll_events | list |无 | 用于充当缓存，记录当前鼠标的滚轮数据|
|_g_frame_count | int | 1 | 用于共享读取当前录制帧数|
|_g_key_abort | bool | False |用于区分停止录制是被人工打断还是自动截断|