# Changelog

记录本项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

### Added
- 字幕文件按视频分文件夹输出，每个视频生成独立子目录
- `Translator` 和 `VoiceGenerator` 新增 `force` 参数，控制是否强制覆盖已有输出
- `Translator.resplit()` 新增 `force` 参数和跳过已存在文件逻辑
- `VoiceGenerator._free_gpu_cache()` 方法，合成每句后释放 XPU/CUDA 显存
- 语音合成进度始终打印（不再仅 verbose 模式）

### Changed
- Whisper 模型从 `base` 升级为 `medium.en`，提升识别质量
- 合并句子的安全阀放宽：最大时长 15s→20s，最大词数 40→50
- 合成文本最大 token 数：60→70
- `VoiceGenerator.process()` 参数 `skip_existing` 改为 `force`（语义更清晰）
- 文件搜索改用递归 glob（`**/*`），支持子文件夹
- 视频查找增加回退逻辑：先在 SRT 所在子文件夹找，再到父目录找
- 调整 `regroup` 参数：`sg=.1`→`sg=.3` 减少碎段

### Fixed
- 修复 CPU 设备上 fp16 参数错误（`fp16=True` → `fp16=False`）
- 修复语音合成时 GPU 显存逐句累积问题

### Added (记录)
- 新增 Whisper large-v3 / turbo / medium.en 性能对比数据（`执行记录.md`）
