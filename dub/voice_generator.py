import os
import sys
import subprocess
import time
import csv
import srt
from pathlib import Path

import intel_extension_for_pytorch as ipex  # noqa - 注册 XPU 后端
import torch
import torchaudio


class VoiceGenerator:
    """语音合成 + 视频配音流水线：

    trans_full.srt → 逐句合成（时间戳对齐）→ 完整 WAV → ffmpeg 替换音轨 → 新视频

    使用 IndexTTS2 (v2.0)。
    """

    def __init__(
        self,
        model_dir=None,
        device=None,
        use_fp16=True,
        glossary_csv=None,
    ):
        """
        :param model_dir:   模型目录，默认 index-tts/checkpoints
        :param device:      设备（None=自动检测）
        :param use_fp16:    是否启用 FP16
        :param glossary_csv: 术语词汇表 CSV 路径（可选），格式: 术语,中文读法,英文读法
        """
        if model_dir is None:
            PROJECT_ROOT = Path(__file__).resolve().parent.parent
            model_dir = str(PROJECT_ROOT / "index-tts" / "checkpoints")

        print("正在加载 IndexTTS2 (v2.0) 模型...")

        from indextts.infer_v2 import IndexTTS2

        self.tts = IndexTTS2(
            cfg_path=os.path.join(model_dir, "config.yaml"),
            model_dir=model_dir,
            use_fp16=use_fp16,
            device=device,
            use_cuda_kernel=False,
            use_deepspeed=False,  
            use_accel=False, 
            use_torch_compile=False,
        )
        self.sampling_rate = 22050  # v2.0 使用 22050 Hz
        self.device = self.tts.device
        print(f"模型加载完成，使用设备: {self.device}")

        # 加载术语词汇表（CSV）
        if glossary_csv:
            n = self.load_glossary_csv(glossary_csv)
            print(f"术语词汇表已加载: {n} 条 (来自 {glossary_csv})")

    # ────────── GPU 缓存清理 ──────────

    def _free_gpu_cache(self):
        """释放 GPU / XPU 显存缓存，防止逐句合成时显存累积。"""
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ────────── 单句合成 ──────────

    def _synthesize_one(self, text: str, prompt_wav: str) -> torch.Tensor:
        """
        合成单句，返回 (1, samples) float32 tensor（CPU）。

        v2.0 内部会按 max_text_tokens_per_segment 自动分段，
        最终返回拼接后的完整音频。合成后立即释放 GPU 显存。
        """
        result = self.tts.infer(
            spk_audio_prompt=str(prompt_wav),
            text=text,
            output_path=None,       # 返回内存数据，不写文件
            verbose=False,
            num_beams=1,
            max_text_tokens_per_segment=70,
        )
        if result is None:
            raise RuntimeError(f"合成失败，文本: {text[:50]}...")

        sr, wav_np = result
        # v2.0 infer() 返回的 wav_np 经过 .numpy().T 转置，shape 为 (samples, 1)
        # 需要 squeeze 掉最后一维得到 (samples,)，再加 batch 维 → (1, samples)
        wav_tensor = torch.from_numpy(wav_np).float().squeeze(-1).unsqueeze(0)  # (1, samples)

        # 显式移动到 CPU 并释放 GPU 显存，防止逐句合成时显存累积
        wav_tensor = wav_tensor.cpu()
        self._free_gpu_cache()

        # 调试：检查合成音频是否真的有内容
        max_val = wav_tensor.abs().max().item()
        if max_val < 100:
            print(f"  [警告] 合成音频幅度极低 (max={max_val:.1f})，可能为静音！")

        return wav_tensor

    # ────────── 术语词汇表 ──────────

    def load_glossary_csv(self, csv_path: str) -> int:
        """
        从 CSV 加载术语词汇表到 TTS normalizer。

        CSV 格式（无 BOM，UTF-8 编码）：
            术语,中文读法,英文读法
            M.2,M 二,M dot two
            PCIe 5.0,PCIE 五点零,PCIE five

        - 第1列：原始术语（大小写不敏感匹配）
        - 第2列：中文环境下的读法
        - 第3列：英文环境下的读法
        - 当中英文读法相同时，存简单字符串；不同时存 {"zh": ..., "en": ...}

        :param csv_path: CSV 文件路径
        :returns:        加载的术语条数
        """
        glossary_dict = {}
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return 0
            for row in reader:
                if not row or not row[0].strip():
                    continue
                term = row[0].strip()
                zh = row[1].strip() if len(row) > 1 and row[1].strip() else ""
                en = row[2].strip() if len(row) > 2 and row[2].strip() else ""

                if not zh and not en:
                    continue
                if zh and en and zh != en:
                    glossary_dict[term] = {"zh": zh, "en": en}
                elif zh and en and zh == en:
                    glossary_dict[term] = zh  # 相同读法用简单字符串
                elif zh:
                    glossary_dict[term] = {"zh": zh}
                else:
                    glossary_dict[term] = {"en": en}

        self.tts.normalizer.load_glossary(glossary_dict)
        return len(glossary_dict)

    @staticmethod
    def save_glossary_csv(glossary_dict: dict, csv_path: str):
        """
        将 term_glossary 字典保存为 CSV 文件。

        :param glossary_dict: 术语字典（与 normalizer.term_glossary 格式一致）
        :param csv_path:      输出 CSV 路径
        """
        rows = [["术语", "中文读法", "英文读法"]]
        for term, reading in glossary_dict.items():
            if isinstance(reading, dict):
                zh = reading.get("zh", "")
                en = reading.get("en", "")
            else:
                zh = en = str(reading)
            rows.append([term, zh, en])

        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    # ────────── 时间戳对齐合成 ──────────

    def generate_speech(
        self,
        srt_path: str,
        prompt_audio: str,
        output_wav: str = None,
        verbose: bool = False,
    ) -> Path:
        """
        读取 trans_full.srt，逐句合成并按 SRT 时间戳对齐，输出完整音频。

        对齐策略：
        - 每句合成音频的起点 = SRT 中该句的 start 时间（开头对齐）
        - 合成短于字幕窗 → 自然静音间隙（语音长度不足则静音）
        - 合成超出字幕窗 → 不截断，与下一句重叠

        :param srt_path:     trans_full.srt 路径
        :param prompt_audio: 音色参考音频路径（支持 wav/mp3/m4a 等，v2.0 内部用 librosa 加载）
        :param output_wav:   输出 WAV 路径（默认推导）
        :param verbose:      打印每句详情
        """
        srt_path = Path(srt_path)

        # 1. 读取 SRT
        with open(srt_path, "r", encoding="utf-8") as f:
            subtitle_list = list(srt.parse(f.read()))

        entries = [
            (sub.start, sub.end, sub.content.strip())
            for sub in subtitle_list
            if sub.content.strip()
        ]
        if not entries:
            raise ValueError(f"字幕文件为空: {srt_path}")

        print(f"读取字幕: {len(entries)} 个句子 (来自 {srt_path.name})")

        sr = self.sampling_rate
        audio_segments = []     # [(start_sec, audio_tensor), ...]
        total_time = 0.0

        # 2. 逐句合成
        #    v2.0 内部已缓存 spk_cond / emo_cond，同一参考音频只提取一次特征
        total_start = time.perf_counter()
        for idx, (start, end, text) in enumerate(entries):
            start_sec = start.total_seconds()
            end_sec = end.total_seconds()

            # 始终打印进度（不使用 tqdm，避免吞掉内置打印信息）
            text_preview = text[:40] + "..." if len(text) > 40 else text
            print(f"  [{idx+1}/{len(entries)}] {text_preview}", flush=True)

            t0 = time.perf_counter()
            wav = self._synthesize_one(text, prompt_audio)
            synth_dur = wav.shape[-1] / sr
            elapsed = time.perf_counter() - t0

            audio_segments.append((start_sec, wav))
            total_time = max(total_time, start_sec + synth_dur)

            if verbose:
                srt_dur = end_sec - start_sec
                print(f"      {start_sec:.1f}s-{end_sec:.1f}s | "
                      f"耗时 {elapsed:.1f}s | 音频 {synth_dur:.1f}s | "
                      f"字幕窗 {srt_dur:.1f}s"
                      f"{' [补静音]' if synth_dur < srt_dur else ''}")

        total_elapsed = time.perf_counter() - total_start
        print(f"合成完成: {len(entries)} 个句子, 总耗时 {total_elapsed:.1f}s")

        # 最终时长取 max(最后一段音频尾部, 最后字幕 end)
        final_end = entries[-1][1].total_seconds()
        total_duration = max(total_time, final_end)
        total_samples = int(total_duration * sr)

        # 3. 在时间轴上放置每段音频（开头对齐，不足补静音）
        timeline = torch.zeros(1, total_samples, dtype=torch.float32)
        for start_sec, wav in audio_segments:
            start_sample = int(start_sec * sr)
            end_sample = min(start_sample + wav.shape[-1], total_samples)
            copy_len = end_sample - start_sample
            if copy_len > 0:
                timeline[:, start_sample:end_sample] = wav[:, :copy_len]

        # 4. 保存
        if output_wav is None:
            stem = srt_path.stem.replace("_trans_full", "")
            output_wav = srt_path.with_name(f"{stem}_dub.wav")
        else:
            output_wav = Path(output_wav)

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        wav_int16 = torch.clamp(timeline, -32767.0, 32767.0).to(torch.int16)
        torchaudio.save(str(output_wav), wav_int16, sr)

        duration = total_samples / sr
        print(f"语音合成完成 -> {output_wav} (时长 {duration:.1f}s)")

        return output_wav

    # ────────── 视频音轨替换 ──────────

    @staticmethod
    def replace_audio(
        video_path: str,
        audio_wav: str,
        output_video: str = None,
    ) -> Path:
        """
        ffmpeg 替换视频音轨，视频流直接拷贝。

        :param video_path:   原始视频
        :param audio_wav:    新音轨 WAV
        :param output_video: 输出视频（默认同目录，加 _dubbed 后缀）
        """
        video_path = Path(video_path)
        audio_wav = Path(audio_wav)

        if output_video is None:
            output_video = video_path.with_name(
                f"{video_path.stem}_dubbed{video_path.suffix}"
            )
        else:
            output_video = Path(output_video)

        output_video.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-i", str(audio_wav),
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-y",
            str(output_video),
        ]

        print("正在替换音轨...")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 音轨替换失败:\n{result.stderr}")

        print(f"音轨替换完成 -> {output_video}")
        return output_video

    # ────────── 批量处理 ──────────

    def process(
        self,
        input_path: str,
        prompt_audio: str,
        force: bool = False,
    ):
        """
        一键处理：trans_full.srt → 语音合成 → 替换视频音轨。

        :param input_path:   单个 trans_full.srt 或目录
        :param prompt_audio: 音色参考音频（支持 wav/mp3/m4a，v2.0 用 librosa 加载）
        :param force:        是否强制覆盖已存在的输出文件
        """
        input_path = Path(input_path)
        tasks = self._collect_tasks(input_path)

        if not tasks:
            print("未找到可处理的任务（需要 trans_full.srt + 对应 .mp4）")
            return

        # v2.0 通过 librosa 加载音频，无需 ffmpeg 预转换
        print(f"参考音频: {prompt_audio}")
        print(f"\n共找到 {len(tasks)} 个任务\n")

        for idx, (srt_path, video_path) in enumerate(tasks, 1):
            print(f"\n{'=' * 60}")
            stem = srt_path.stem.replace("_trans_full", "")
            print(f"[{idx}/{len(tasks)}] {stem}")
            print(f"{'=' * 60}")

            wav_path = srt_path.with_name(f"{stem}_dub.wav")
            dubbed_video = video_path.with_name(
                f"{stem}_dubbed{video_path.suffix}"
            )

            if dubbed_video.exists():
                if not force:
                    print(f"  配音视频已存在，跳过 -> {dubbed_video.name}")
                    continue
                else:
                    print(f"  配音视频已存在，强制覆盖 -> {dubbed_video.name}")

            try:
                if wav_path.exists():
                    if not force:
                        print(f"  音频已存在，跳过合成 -> {wav_path.name}")
                    else:
                        print(f"  音频已存在，强制重新合成 -> {wav_path.name}")
                        self.generate_speech(
                            srt_path=str(srt_path),
                            prompt_audio=str(prompt_audio),
                            output_wav=str(wav_path),
                        )
                else:
                    self.generate_speech(
                        srt_path=str(srt_path),
                        prompt_audio=str(prompt_audio),
                        output_wav=str(wav_path),
                    )

                self.replace_audio(
                    video_path=str(video_path),
                    audio_wav=str(wav_path),
                    output_video=str(dubbed_video),
                )

            except Exception as e:
                print(f"  X 处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'=' * 60}")
        print("全部处理完成！")

    def _collect_tasks(self, input_path: Path) -> list:
        """收集 (trans_full.srt, video.mp4) 任务对"""
        if input_path.is_file():
            srt_files = [input_path]
        elif input_path.is_dir():
            srt_files = sorted(input_path.glob("**/*_trans_full.srt"))
        else:
            raise FileNotFoundError(f"路径不存在: {input_path}")

        tasks = []
        for srt_path in srt_files:
            stem = srt_path.stem.replace("_trans_full", "")
            # 先在 srt 所在子文件夹找视频，再到父目录找
            video_path = srt_path.with_name(f"{stem}.mp4")
            if not video_path.exists():
                video_path = srt_path.parent.parent / f"{stem}.mp4"

            if not video_path.exists():
                print(f"  警告: 未找到对应视频，跳过 -> {srt_path.name}")
                continue

            tasks.append((srt_path, video_path))

        return tasks


# ======================================================================
if __name__ == "__main__":
    PROMPT_AUDIO = r"E:\CourseVideo\.NET内存专家\参考音频5s.wav"
    INPUT_PATH = r"E:\CourseVideo\.NET内存专家"
    GLOSSARY_CSV = r"E:\CourseVideo\.NET内存专家\terminology_voice.csv"

    generator = VoiceGenerator(
        use_fp16=True,
        glossary_csv=GLOSSARY_CSV,
    )

    generator.process(
        input_path=INPUT_PATH,
        prompt_audio=PROMPT_AUDIO,
        force=False,
    )
