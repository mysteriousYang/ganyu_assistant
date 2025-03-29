@echo off
setlocal enabledelayedexpansion

REM 参数配置
set "ENV_NAME=ganyuEnv"
set "CONDA_PATH=D:\Softwares\anaconda3"
set "SCRIPT1=E:\大学\大四上\毕设\ganyu_assistant\simulate_ctrl.py"
set "SCRIPT2=E:\大学\大四上\毕设\ganyu_assistant\capture_tools\video_capture.py"

REM 激活环境
call "%CONDA_PATH%\Scripts\activate.bat" %ENV_NAME%
if %errorlevel% neq 0 (
    echo [错误] 环境激活失败: %ENV_NAME%
    exit /b 1
)

REM 并发执行
start "进程1" cmd /c "python "%SCRIPT1%" --admin && echo 脚本1执行成功 || echo 脚本1执行失败"
start "进程2" cmd /c "python "%SCRIPT2%" --admin && echo 脚本2执行成功 || echo 脚本2执行失败"

REM 监控窗口（可选）
tasklist /fi "imagename eq python.exe"
pause