# AI 翻译配音

将英文视频自动翻译为中文配音的一站式工具。适用于技术课程、演讲等英文视频的本地化。

## 功能概览

1. **字幕提取** — 基于 Whisper (stable-whisper) 从视频中提取英文字幕
2. **字幕翻译** — 调用 DeepSeek API 将英文字幕翻译为中文，支持术语表
3. **语音合成** — 基于 IndexTTS2 (v2.0) 生成中文语音，支持音色克隆
4. **音画合成** — 通过 ffmpeg 将生成的语音替换到原视频中

## 技术路线

### 硬件环境

- **GPU**: Intel Arc A770 16GB
- **PyTorch**: 2.8 + IPEX (Intel Extension for PyTorch) XPU 后端
- **环境管理**: Conda (`xpu` 环境)

### 流水线架构

```
视频 (.mp4)
    │
    ▼
┌─────────────────────────────┐
│ 1. 字幕提取 (subtitle_extractor)  │
│    stable-whisper medium.en       │
│    → {video}_ori.srt              │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 2. 字幕翻译 (subtitle_translator) │
│    DeepSeek API (deepseek-v4-flash)│
│    句子合并 → 批量翻译 → 字幕拆分    │
│    → {video}_trans_final.srt      │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 3. 语音合成 (voice_generator)     │
│    IndexTTS2 v2.0                 │
│    时间戳对齐 → 逐句合成 → 完整 WAV  │
│    → {video}_dub.wav              │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 4. 音画合成 (ffmpeg)              │
│    视频流拷贝 + 新音轨替换          │
│    → {video}_dubbed.mp4           │
└─────────────────────────────┘
```

### 各模块详解

#### 1. 字幕提取 (`dub/subtitle_extractor.py`)

- **模型**: `stable-whisper` 的 `medium.en`（平衡速度与质量，RTF ≈ 3.57）
- **关键优化**: 自定义 `regroup` 规则按句末标点切分、合并碎片，减少逐词字幕
- **XPU 适配**: `fp16=False` 时在 CPU 上运行（XPU 不支持 fp16 推理）
- **输出**: 每个视频生成独立子文件夹，存放 `_ori.srt`

#### 2. 字幕翻译 (`dub/subtitle_translator.py`)

- **合并阶段** (`SentenceMerger`): 将 ASR 碎片合并为完整句子（句末标点 + 时长/词数安全阀）
- **翻译阶段**: 调用 DeepSeek API（`deepseek-v4-flash`），每批 10 句，要求 JSON 格式输出
- **拆分阶段** (`SubtitleSplitter`): 将长中文译文按字符数拆分为短字幕，优先在标点处断句
- **术语表**: 支持 CSV 格式术语对照表，确保专业术语翻译一致性

#### 3. 语音合成 (`dub/voice_generator.py`)

- **模型**: IndexTTS2 v2.0，22050 Hz 采样率
- **音色克隆**: 提供 5 秒参考音频即可克隆音色
- **时间轴对齐**: 每句合成音频按 SRT 时间戳放置，不足补静音，超出与下句自然重叠
- **XPU 内存优化**: 每句合成后立即释放 GPU 缓存，防止显存累积
- **术语读法**: 支持 CSV 术语词汇表，控制中英文专业术语的正确读法（如 "PCIe 5.0" → "PCIE 五点零"）

#### 4. 配置管理

| 文件 | 用途 |
|------|------|
| `config.yaml` | API 配置模板（key 占位） |
| `config_dev.yaml` | 开发环境密钥（已 gitignore） |
| `config/config.yaml` | IndexTTS2 模型配置 |

### 依赖

| 组件 | 用途 |
|------|------|
| `stable-whisper` / `faster-whisper` | 语音识别 / 字幕提取 |
| `openai` | DeepSeek API 调用（兼容 OpenAI SDK） |
| `srt` | SRT 字幕解析与生成 |
| `torch` + `intel-extension-for-pytorch` | XPU 推理后端 |
| `torchaudio` | 音频读写 |
| `ffmpeg` | 音视频合成 |
| `IndexTTS2` (v2.0) | 中文语音合成 |

### 项目结构

```
xpu/
├── dub/
│   ├── subtitle_extractor.py    # 字幕提取
│   ├── subtitle_translator.py   # 字幕翻译
│   └── voice_generator.py       # 语音合成 & 视频合成
├── index-tts/                   # IndexTTS2 模型目录
│   └── checkpoints/             # 模型权重
├── utils/
│   └── path_manager.py          # 路径管理
├── test/
│   └── 设备测试.py               # XPU 设备检测
├── config.yaml                  # API 配置模板
├── config_dev.yaml              # 开发环境密钥
└── pyproject.toml
```
