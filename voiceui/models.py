from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class NodeConfig:
    id: str = "local_node"
    room: str = "default"


@dataclass(slots=True)
class InputConfig:
    mode: Literal["text", "audio"] = "text"


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    block_ms: int = 80
    device: str | int | None = None
    wake_stream_channel: int = 0
    command_stream_channel: int = 0
    input_gain_db: float = 0.0
    debug: bool = False


@dataclass(slots=True)
class WakeConfig:
    engine: Literal["disabled", "manual", "openwakeword", "sherpa_onnx"] = "disabled"
    model: str = "alexa"
    threshold: float = 0.5
    max_wait_seconds: float = 0.0
    inference_framework: Literal["onnx", "tflite"] = "onnx"
    debug: bool = False
    debug_interval_seconds: float = 1.0
    debug_top_predictions: int = 5
    debug_audio_seconds: float = 5.0


@dataclass(slots=True)
class WakeAckConfig:
    enabled: bool = False
    wav_path: str = ""
    playback_device: str | int | None = None


@dataclass(slots=True)
class VadConfig:
    engine: Literal["energy", "silero", "webrtc"] = "energy"
    threshold: float = 450.0
    min_speech_ms: int = 250
    silence_ms: int = 800
    trailing_silence_trim_ms: int = 500
    max_speech_ms: int = 15000
    pre_roll_ms: int = 240
    frame_ms: int = 20
    webrtc_mode: int = 2
    debug: bool = False


@dataclass(slots=True)
class SttConfig:
    provider: Literal[
        "mock",
        "faster_whisper",
        "openai_compatible",
        "mify",
        "mimo",
        "aliyun_nls",
    ] = "mock"
    endpoint: str = "http://localhost:8000/v1/audio/transcriptions"
    api_key_env: str | None = None
    access_key_id_env: str | None = None
    access_key_secret_env: str | None = None
    app_key_env: str | None = None
    timeout_seconds: float = 60.0
    model: str = "small"
    language: str | None = None
    device: str = "cpu"
    compute_type: str = "int8"
    mock_text: str = "Hello from mock STT."
    leading_silence_ms: int = 0
    debug: bool = False


@dataclass(slots=True)
class LlmConfig:
    provider: Literal["mock", "ollama", "openai_compatible", "bailian", "mify", "mimo"] = "mock"
    endpoint: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct"
    api_key_env: str | None = None
    temperature: float = 0.3
    timeout_seconds: float = 60.0
    stream: bool = False
    extra_body: dict[str, object] = field(default_factory=dict)
    system_prompt: str = (
        "You are a concise home voice assistant. Answer briefly and ask for "
        "confirmation before sensitive actions."
    )


@dataclass(slots=True)
class TtsConfig:
    provider: Literal[
        "console",
        "system",
        "mify",
        "mimo",
        "openai_speech",
        "openai_compatible_speech",
        "aliyun_nls",
        "piper_http",
        "piper_cli",
    ] = "console"
    endpoint: str = "https://api.xiaomimimo.com/v1"
    api_key_env: str | None = None
    access_key_id_env: str | None = None
    access_key_secret_env: str | None = None
    app_key_env: str | None = None
    timeout_seconds: float = 60.0
    model: str = "mimo-v2-tts"
    voice: str = "mimo_default"
    audio_format: str = "pcm"
    sample_rate: int = 24000
    playback_sample_rate: int | None = None
    playback_channels: int | None = None
    style_prompt: str = "自然、清晰、适合智能音箱的中文播报。"
    stream: bool = False
    volume: int = 50
    speech_rate: int = 0
    pitch_rate: int = 0
    piper_url: str = "http://localhost:5000"
    piper_model: str | None = None
    playback_device: str | int | None = None
    limiter_enabled: bool = True
    limiter_threshold: float = 0.92


@dataclass(slots=True)
class ConversationConfig:
    follow_up_seconds: int = 10
    max_turns: int = 12
    barge_in_enabled: bool = False


@dataclass(slots=True)
class HomeAssistantConfig:
    enabled: bool = False
    url: str = "http://homeassistant.local:8123"
    token_env: str = "HA_TOKEN"
    default_area: str = "default"


@dataclass(slots=True)
class MusicConfig:
    provider: Literal["disabled", "meting"] = "disabled"
    endpoint: str = "https://meting.mikus.ink/api"
    server: str = "netease"
    timeout_seconds: float = 20.0
    max_results: int = 5
    max_audio_bytes: int = 50_000_000
    playback_enabled: bool = True
    start_delay_seconds: float = 1.0
    playback_device: str | int | None = None
    playback_sample_rate: int | None = None
    playback_channels: int | None = None
    playback_volume: float = 1.0
    ducking_volume_factor: float = 0.2
    limiter_enabled: bool = True
    limiter_threshold: float = 0.92


@dataclass(slots=True)
class SearchConfig:
    provider: Literal["auto", "baidu", "tavily"] = "auto"
    tavily_endpoint: str = "https://api.tavily.com/search"
    tavily_api_key_env: str = "TAVILY_API_KEY"
    tavily_search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic"
    tavily_topic: Literal["general", "news", "finance"] = "general"
    baidu_ai_enabled: bool = True
    baidu_ai_endpoint: str = "https://qianfan.baidubce.com/v2/ai_search/chat/completions"
    baidu_ai_api_key_env: str = "QIANFAN_API_KEY"
    baidu_ai_model: str = "deepseek-v3"
    baidu_ai_search_mode: Literal["auto", "required", "disabled"] = "required"
    baidu_ai_deep_search: bool = False
    baidu_ai_fallback_to_html: bool = True
    baidu_endpoint: str = "https://www.baidu.com/baidu"
    timeout_seconds: float = 15.0
    max_results: int = 5


@dataclass(slots=True)
class XiaomiMiotConfig:
    enabled: bool = False
    cloud_server: str = "cn"
    token_file: str = ".voiceui/miot_token.json"
    token_json_env: str = "XIAOMI_MIOT_TOKEN_JSON"
    access_token_env: str = "XIAOMI_MIOT_ACCESS_TOKEN"
    refresh_token_env: str = "XIAOMI_MIOT_REFRESH_TOKEN"
    uuid_env: str = "XIAOMI_MIOT_UUID"
    uuid_file: str = ".voiceui/miot_uuid"
    redirect_uri: str = "https://mico.api.mijia.tech/login_redirect"
    cache_dir: str = ".voiceui/miot_cache"
    fetch_share_home: bool = False
    auto_refresh: bool = True
    request_timeout_seconds: float = 30.0
    max_devices: int = 200
    control_verify: bool = True
    control_verify_delay_seconds: float = 0.8


@dataclass(slots=True)
class ToolsConfig:
    enabled: bool = False
    max_iterations: int = 4
    allow_time: bool = True
    allow_weather: bool = True
    default_weather_location: str = ""
    allow_volume: bool = False
    allow_music: bool = False
    allow_miot: bool = False
    allow_search: bool = False


@dataclass(slots=True)
class DebugConfig:
    enabled: bool = False
    output_dir: str = "debug_sessions"
    save_audio: bool = True
    save_metadata: bool = True
    system_input_dump_enabled: bool = True
    system_input_dump_segment_seconds: float = 30.0
    voice_path_dump_enabled: bool = True


@dataclass(slots=True)
class LoggingConfig:
    enabled: bool = True
    events: dict[str, bool] = field(default_factory=dict)
    continuous: dict[str, bool] = field(default_factory=dict)


@dataclass(slots=True)
class AssistantConfig:
    node: NodeConfig = field(default_factory=NodeConfig)
    input: InputConfig = field(default_factory=InputConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    wake_ack: WakeAckConfig = field(default_factory=WakeAckConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    home_assistant: HomeAssistantConfig = field(default_factory=HomeAssistantConfig)
    music: MusicConfig = field(default_factory=MusicConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    xiaomi_miot: XiaomiMiotConfig = field(default_factory=XiaomiMiotConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@dataclass(slots=True)
class WakeEvent:
    engine: str
    confidence: float
    label: str
    pcm: bytes = b""
    sample_rate: int = 16000
    duration_ms: int = 0


@dataclass(slots=True)
class Utterance:
    pcm: bytes
    sample_rate: int
    duration_ms: int


@dataclass(slots=True)
class AssistantReply:
    text: str
    routed_to: str = "llm"
