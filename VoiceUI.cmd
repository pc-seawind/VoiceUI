@echo off
setlocal
cd /d "%~dp0"
python -m voiceui.service --config config.demo.wake.aliyun.yaml %*
