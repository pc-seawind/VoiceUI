# 就近唤醒说明

本文档说明双 XVF3800 在 Windows PC 上的就近唤醒方案、测试脚本和生产端到端验证方式。

## 目标

PC 同时连接两台 `reSpeaker XVF3800 4-Mic Array`。用户喊唤醒词后，系统需要选择离用户更近的设备响应；如果一端已经进入语音交互，另一端仍然可以独立被唤醒并启动自己的语音交互通路。

当前实现放在 `codex/wake-proximity` 分支，尚未合入 main。

## 设备约定

不要依赖 sounddevice 的数字 device id。Windows 重启后数字 id 可能变化，配置和脚本统一使用完整 WASAPI 设备显示名：

```text
回音消除话筒 (reSpeaker XVF3800 4-Mic Array), Windows WASAPI (2 in, 0 out)
回音消除话筒 (2- reSpeaker XVF3800 4-Mic Array), Windows WASAPI (2 in, 0 out)
回音消除话筒 (reSpeaker XVF3800 4-Mic Array), Windows WASAPI (0 in, 2 out)
回音消除话筒 (2- reSpeaker XVF3800 4-Mic Array), Windows WASAPI (0 in, 2 out)
```

`voiceui.audio.resolve_sounddevice_device()` 会按完整名称、host API、输入/输出方向和通道数解析到当前运行时的 device id。

## 通道约定

XVF3800 的 WASAPI 输入是 2 通道：

- `ch0`: 降噪/AEC 后的语音流，适合唤醒词检测、VAD、STT。
- `ch1`: 原始声音流，噪声更多，但远近清晰度差异更明显，适合做就近判断。

因此就近唤醒脚本默认使用：

- `--wake-channel-a 0`
- `--wake-channel-b 0`
- `--proximity-channel-a 1`
- `--proximity-channel-b 1`

## 仲裁算法

当前算法分两层：

1. ch0 只负责触发唤醒。任意一台设备的 openWakeWord 分数超过阈值后，系统进入仲裁。
2. 使用同一个全局 wake window，从所有设备的 ch1 原始流中提取语音段特征，再选择近端。

全局窗口默认是：

- 唤醒触发点前 `1300ms`
- 唤醒触发点后 `300ms`

特征包括：

- 语音带 RMS
- 语音带 SNR
- speech-band ratio
- wake confidence

如果只有远端 ch0 触发，但近端 ch1 语音证据明显更强，允许近端胜出。前提是至少有一端真实触发 ch0；如果没有任何设备触发，结果是 `no_wake`。

### 清晰度二级仲裁

基础 RMS/SNR 仲裁之后，会再跑一层可选的语音清晰度仲裁。该层只使用同一个 wake window 内的 ch1 原始音频，默认 `--clarity-engine auto`：

- SQUIM: 估计 `STOI`、`PESQ`、`SI-SDR`，用于判断可懂度和信号质量。
- SIGMOS: 估计 `SIG`、`NOISE`、`REVERB`、`OVRL`，用于判断语音信号、噪声和混响质量。
- ch1 RMS: 保留原始能量证据，避免模型把“很安静但没录清楚”的远端通道误判为更好。

组合分数默认使用 `STOI + SI-SDR + SIGMOS(SIG/NOISE/REVERB) + RMS`。当清晰度候选和基础 winner 不一致，并且 `--clarity-override-margin` 达到默认 `0.10` 时，清晰度结果会覆盖基础 winner。

清晰度层默认不会把 `no_wake` 变成某个设备，也就是说它不能凭空制造唤醒，只能在已经有设备唤醒后修正选错端的问题。需要离线实验时可以加 `--clarity-allow-no-wake-override`。

如果新机器没有安装模型依赖，清晰度层会自动退回基础 RMS/SNR 仲裁。需要显式安装时使用：

```powershell
pip install -e ".[clarity]"
```

## 纯唤醒 live 测试

运行：

```powershell
.\scripts\wake_proximity_free.ps1
```

行为：

- 脚本一直监听两台 XVF3800。
- 用户随便喊唤醒词。
- 系统选择近端设备。
- 只在被选中的设备上播放本地 wake ack。
- 不进入 VAD/STT/LLM/TTS。

结果目录：

```text
debug_sessions\wake_proximity_live_wake\<run>\
```

关键输出包括：

- `trials.jsonl`
- `trials.csv`
- `summary.json`
- `audio\wake_<n>_<label>_wake_ch0.wav`
- `audio\wake_<n>_<label>_ch1.wav`

## 生产端到端 live 验证

运行：

```powershell
.\scripts\wake_proximity_e2e.ps1
```

默认使用：

```text
config.demo.wake.aliyun.yaml
```

行为：

- 两台 XVF3800 同时进行近端唤醒监听。
- 被选中的端点会进入正式语音链路：wake ack、VAD、STT、LLM、TTS。
- 正式链路保留 `config.demo.wake.aliyun.yaml` 的 Aliyun NLS STT/TTS、Bailian LLM、工具和多轮 follow-up 配置。
- 每个端点有独立的 `VoiceAssistant` 实例。
- xvf1 busy 时，xvf2 仍然可以唤醒并进入自己的语音交互。
- 如果两端分别被唤醒，会走两条独立交互通路。
- 同一端点 busy 期间不会重复响应新唤醒，等该端 follow-up 超时或会话结束后恢复监听。

结果目录：

```text
debug_sessions\wake_proximity_prod_live\<run>\
```

`trials.csv` 会记录：

- `selected_device`
- `trigger_source_device`
- 每台设备的 ch0 confidence
- 每台设备的 ch1 band RMS/SNR
- `ack_output_device`
- `ack_latency_ms`
- `assistant_transcript`
- `assistant_reply`
- `assistant_error`

## 采集与离线分析

保留实验采集入口，用于标注位置并批量评估算法：

```powershell
python -m voiceui.wake_proximity collect --repetitions 5
python -m voiceui.wake_proximity summarize debug_sessions\wake_proximity\<run>
```

采集入口会提示站位和轮次；live 测试入口不会提示站位，按真实使用方式直接说话即可。

## 调试建议

如果纯唤醒 live 正常，但端到端慢：

- 确认端到端脚本使用的是 `config.demo.wake.aliyun.yaml`。
- 默认关闭了 assistant system input dump，避免额外占用输入流；需要调试时可加：

```powershell
.\scripts\wake_proximity_e2e.ps1 --system-input-dump
```

如果唤醒后命令开头被吃掉，下一步需要做 rolling command buffer，把唤醒点后的 ch0 音频接入 VAD/STT。

如果就近选择错误，优先听这些文件：

```text
audio\wake_<n>_xvf1_ch1.wav
audio\wake_<n>_xvf2_ch1.wav
```

并查看 `trials.csv` 中的 `band_rms`、`band_snr_db`、`triggered` 和 `best_confidence`。
