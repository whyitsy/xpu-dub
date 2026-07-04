import yaml
import srt
import json
import re
import csv
from pathlib import Path
from datetime import timedelta
from openai import OpenAI

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


class SentenceMerger:
    """将 ASR 碎片化字幕合并为完整句子"""

    # 句末标点：遇到这些标点视为一个完整句子结束
    SENTENCE_END_PATTERN = re.compile(r'[.?!]\s*$')
    # 安全阀：单句最大合并时长（秒），防止无限合并
    MAX_MERGE_DURATION = 15.0
    # 安全阀：单句最大合并词数
    MAX_MERGE_WORDS = 40

    @classmethod
    def merge(cls, subtitle_list: list) -> list:
        """
        将原始字幕片段合并为完整句子。
        规则：
        - 以 . ? ! 结尾 → 句子结束
        - 以 , 或无标点结尾 → 继续合并下一片段
        - 超过安全阀 → 强制结束

        返回：合并后的字幕列表（每个条目是一个完整句子）
        """
        merged = []
        buffer_segments = []

        for seg in subtitle_list:
            buffer_segments.append(seg)

            text = seg.content.strip()
            total_duration = (
                buffer_segments[-1].end - buffer_segments[0].start
            ).total_seconds()
            total_words = sum(len(s.content.split()) for s in buffer_segments)

            is_sentence_end = bool(cls.SENTENCE_END_PATTERN.search(text))
            is_too_long = total_duration > cls.MAX_MERGE_DURATION
            is_too_many_words = total_words >= cls.MAX_MERGE_WORDS

            if is_sentence_end or is_too_long or is_too_many_words:
                cls._finalize_block(buffer_segments, merged)
                buffer_segments = []

        # 处理末尾残留（无句末标点结尾的最后一句）
        if buffer_segments:
            cls._finalize_block(buffer_segments, merged)

        return merged

    @staticmethod
    def _finalize_block(segments: list, output: list):
        """将一个缓冲区的片段合并为一个字幕条目"""
        if not segments:
            return
        full_text = " ".join(s.content.strip() for s in segments)
        output.append(
            srt.Subtitle(
                index=0,  # 后续重编号
                start=segments[0].start,
                end=segments[-1].end,
                content=full_text,
            )
        )


class SubtitleSplitter:
    """将长字幕按时长比例拆分为短字幕"""

    MAX_CHARS = 25  # 中文字幕单段最大字符数

    @classmethod
    def split(cls, subtitle: srt.Subtitle) -> list:
        """
        将一个字幕条目按字符数拆分为多个短片段。
        优先在标点处断句，其次按字符数硬切。
        时间按字符数比例分配。
        """
        text = subtitle.content.strip()
        total_chars = len(text)

        if total_chars <= cls.MAX_CHARS:
            return [subtitle]

        # 计算需要拆成几段
        num_segments = (total_chars + cls.MAX_CHARS - 1) // cls.MAX_CHARS
        target_per_segment = total_chars / num_segments

        # 在目标位置附近找最佳断点
        segments = cls._split_text(text, num_segments, target_per_segment)

        # 生成拆分后的字幕条目，时间按字符数比例分配
        total_duration = (subtitle.end - subtitle.start).total_seconds()
        result = []
        current_start = subtitle.start

        for seg_text in segments:
            seg_duration = timedelta(
                seconds=total_duration * len(seg_text) / total_chars
            )
            seg_end = current_start + seg_duration
            result.append(
                srt.Subtitle(
                    index=0,
                    start=current_start,
                    end=seg_end,
                    content=seg_text.strip(),
                )
            )
            current_start = seg_end

        return result

    @classmethod
    def _split_text(cls, text: str, num_segments: int, target: float) -> list:
        """
        在目标字符数附近寻找最佳断点。
        断点优先级：。！？ > ，、；： > 空格 > 硬切
        """
        if num_segments <= 1:
            return [text]

        segments = []
        remaining = text

        for _ in range(num_segments - 1):
            ideal_pos = int(round(target))
            # 搜索窗口：目标位置 ±30%
            window_start = max(1, int(ideal_pos * 0.7))
            window_end = min(len(remaining) - 1, int(ideal_pos * 1.3))

            # 优先级1：句末标点
            best = cls._find_best_break(
                remaining, window_start, window_end, priority=1
            )
            if best is None:
                # 优先级2：逗号等
                best = cls._find_best_break(
                    remaining, window_start, window_end, priority=2
                )
            if best is None:
                # 优先级3：空格
                best = cls._find_best_break(
                    remaining, window_start, window_end, priority=3
                )
            if best is None:
                # 兜底：硬切
                best = ideal_pos

            segments.append(remaining[:best])
            remaining = remaining[best:]

        segments.append(remaining)
        return [s.strip() for s in segments]

    @staticmethod
    def _find_best_break(text: str, start: int, end: int, priority: int) -> int | None:
        """在搜索窗口内找最高优先级的断点位置（断在标点之后）"""
        if priority == 1:
            pattern = r'[。！？.!?]'
        elif priority == 2:
            pattern = r'[，、；：,;:—]'
        else:
            pattern = r'\s'

        best = None
        # 从靠近目标位置开始搜索，优先选最近的
        for m in re.finditer(pattern, text[start:end]):
            pos = start + m.end()  # 断在标点之后
            if best is None:
                best = pos
            else:
                # 选择更接近理想位置的
                mid = (start + end) / 2
                if abs(pos - mid) < abs(best - mid):
                    best = pos
        return best


class Translator:
    """字幕翻译流水线：
    ori.srt → (合并) → ori_full.srt → (翻译) → trans_full.srt → (拆分) → trans_final.srt
    """

    def __init__(self, input_path: str, term_file_path=None):
        """
        :param input_path: 单个 SRT 文件路径，或包含 SRT 文件的目录路径
        :param term_file_path: 术语表 CSV 文件路径
        """
        self.input_path = Path(input_path)
        self.term_file_path = term_file_path

        if self.input_path.is_file():
            self.srt_files = [self.input_path]
        elif self.input_path.is_dir():
            self.srt_files = sorted(self.input_path.glob("*_ori.srt"))
            if not self.srt_files:
                print(f"警告：目录 {input_path} 中未找到 *_ori.srt 文件")
        else:
            raise FileNotFoundError(f"路径不存在: {input_path}")

        # API 客户端
        self.model_type = config["api"]["deepseek"]["model_type"]
        self.client = OpenAI(
            api_key=config["api"]["deepseek"]["api_key"],
            base_url=config["api"]["deepseek"]["base_url"],
        )
        self.sys_prompt = self._make_sys_prompt()

    # ────────── 术语表 ──────────

    def _read_term(self) -> dict:
        term_dict = {}
        if self.term_file_path is None:
            return term_dict
        with open(self.term_file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                if row:
                    term_dict[row[0].strip()] = row[1].strip()
        return term_dict

    def _make_sys_prompt(self) -> str:
        term_dict = self._read_term()
        prompt = "你是一位专业的技术文档翻译专家。请将以下英文技术课程字幕句子翻译为中文。\n"
        prompt += "注意上下文衔接和专业术语一致性，正确还原指代关系。\n"

        if term_dict:
            prompt += "必须严格遵守以下术语对照表：\n"
            for en, zh in term_dict.items():
                prompt += f"- {en} -> {zh}\n"

        prompt += """
        要求：
        1. 保持与输入句子相同的数量和顺序，逐条对应，不得合并或拆分句子。
        2. 符合中文技术文档表达习惯，保留原意，准确专业。
        3. 必须严格以 JSON 格式输出，结构如下：
        {
            "translations": [
                "第1句译文",
                "第2句译文",
                "..."
            ]
        }
        只输出JSON，不要输出任何额外解释、标记或前言。
        """
        return prompt

    # ────────── JSON 解析 ──────────

    @staticmethod
    def _clean_json_text(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _parse_translation_result(self, raw_content: str, expected_count: int) -> list:
        clean_text = self._clean_json_text(raw_content)
        try:
            result = json.loads(clean_text)
            translations = result.get("translations", [])
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON解析失败: {e}\n原始内容: {clean_text}") from e

        if len(translations) != expected_count:
            raise RuntimeError(
                f"译文数量不匹配：期望 {expected_count} 条，实际返回 {len(translations)} 条\n"
                f"返回内容: {clean_text}"
            )
        return translations

    # ────────── 主流程 ──────────

    def run(self):
        """处理所有找到的 SRT 文件"""
        for srt_file in self.srt_files:
            print(f"\n{'='*60}")
            print(f"正在处理: {srt_file.name}")
            print(f"{'='*60}")
            self._process_single(srt_file)

    @staticmethod
    def resplit(input_path: str, max_chars: int = 25):
        """
        独立拆分功能：读取 trans_full.srt，重新拆分为 trans_final.srt。
        适用于翻译已完成、只需调整拆分参数（如每行字符数）的场景。

        :param input_path: 单个 trans_full.srt 文件路径，或包含 trans_full.srt 文件的目录路径
        :param max_chars:  单条字幕最大字符数，默认 25
        """
        input_path = Path(input_path)

        if input_path.is_file():
            full_files = [input_path]
        elif input_path.is_dir():
            full_files = sorted(input_path.glob("*_trans_full.srt"))
            if not full_files:
                print(f"警告：目录 {input_path} 中未找到 *_trans_full.srt 文件")
                return
        else:
            raise FileNotFoundError(f"路径不存在: {input_path}")

        # 临时覆盖 MAX_CHARS（实例属性不影响类属性）
        original_max = SubtitleSplitter.MAX_CHARS
        SubtitleSplitter.MAX_CHARS = max_chars

        for full_path in full_files:
            print(f"正在拆分: {full_path.name}")

            with open(full_path, "r", encoding="utf-8") as f:
                subtitle_list = list(srt.parse(f.read()))

            final_subtitles = []
            for sub in subtitle_list:
                final_subtitles.extend(SubtitleSplitter.split(sub))
            final_subtitles = list(srt.sort_and_reindex(final_subtitles))

            final_path = full_path.with_name(
                full_path.stem.replace("_trans_full", "_trans_final") + ".srt"
            )
            with open(final_path, "w", encoding="utf-8") as f:
                f.write(srt.compose(final_subtitles))

            print(f"  拆分完成: {len(subtitle_list)} 句 → {len(final_subtitles)} 条 → {final_path.name}")

        SubtitleSplitter.MAX_CHARS = original_max

    def _process_single(self, ori_srt_path: Path):
        """处理单个 SRT 文件的完整流水线"""
        # 定义输出路径
        ori_full_path = ori_srt_path.with_name(
            ori_srt_path.stem.replace("_ori", "_ori_full") + ".srt"
        )
        trans_full_path = ori_srt_path.with_name(
            ori_srt_path.stem.replace("_ori", "_trans_full") + ".srt"
        )

        # 1. 读取原始字幕
        with open(ori_srt_path, "r", encoding="utf-8") as f:
            ori_subtitle_list = list(srt.parse(f.read()))
        print(f"原始字幕: {len(ori_subtitle_list)} 条片段")

        # 2. 合并为完整句子 → ori_full.srt
        merged_subtitles = SentenceMerger.merge(ori_subtitle_list)
        merged_subtitles = list(srt.sort_and_reindex(merged_subtitles))
        with open(ori_full_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(merged_subtitles))
        print(f"句子合并: {len(merged_subtitles)} 个完整句子 → {ori_full_path.name}")

        # 3. 批量翻译 → trans_full.srt
        translated_subtitles = self._batch_translate(
            merged_subtitles, log_dir=ori_srt_path.parent
        )
        translated_subtitles = list(srt.sort_and_reindex(translated_subtitles))
        with open(trans_full_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(translated_subtitles))
        print(f"翻译完成: {len(translated_subtitles)} 个句子 → {trans_full_path.name}")

        # 4. 拆分为短字幕 → trans_final.srt
        Translator.resplit(str(trans_full_path))

    def _batch_translate(
        self, subtitle_list: list, log_dir: Path = None
    ) -> list:
        """批量翻译字幕列表，返回翻译后的字幕列表"""
        texts = [s.content.strip() for s in subtitle_list]
        translated_texts = []
        batch_size = 10

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size

            print(
                f"  翻译批次 {batch_num}/{total_batches} "
                f"(第 {i+1}~{min(i+batch_size, len(texts))} 句) ..."
            )

            numbered_input = "\n".join(
                f"{idx + 1}. {text}" for idx, text in enumerate(batch)
            )

            try:
                response = self.client.chat.completions.create(
                    model=self.model_type,
                    messages=[
                        {"role": "system", "content": self.sys_prompt},
                        {
                            "role": "user",
                            "content": f"请翻译以下英文句子，保持顺序，输出合法JSON：\n\n{numbered_input}",
                        },
                    ],
                    temperature=0.3,
                    stream=False,
                    response_format={"type": "json_object"},
                    max_tokens=4096,
                )
                raw_result = response.choices[0].message.content.strip()
                batch_translations = self._parse_translation_result(
                    raw_result, expected_count=len(batch)
                )
                translated_texts.extend(batch_translations)

            except Exception as e:
                print(f"  ✗ 批次 {batch_num} 翻译失败: {e}")
                # 失败时保留原文
                translated_texts.extend(batch)
                # 写入失败日志
                if log_dir:
                    fail_log = log_dir / f"translate_failures_batch_{batch_num}.txt"
                    with open(fail_log, "w", encoding="utf-8") as f:
                        f.write(f"批次 {batch_num} 翻译失败: {e}\n\n原文:\n")
                        for idx, text in enumerate(batch):
                            f.write(f"{idx+1}. {text}\n")

        # 构建翻译后的字幕列表（保持原始时间戳）
        result = []
        for sub, trans_text in zip(subtitle_list, translated_texts):
            result.append(
                srt.Subtitle(
                    index=0,
                    start=sub.start,
                    end=sub.end,
                    content=trans_text,
                )
            )
        return result


if __name__ == "__main__":
    # ─── 完整流水线：ori.srt → ori_full.srt → trans_full.srt → trans_final.srt ───
    input_path = r"E:\CourseVideo\.NET内存专家\NET内存专家.3._作业.39355354830_ori.srt"
    term_file_path = r"E:\CourseVideo\.NET内存专家\terminology_translate.csv"
    translator = Translator(input_path=input_path, term_file_path=term_file_path)
    translator.run()

    # ─── 独立拆分：trans_full.srt → trans_final.srt（无需 API，不需术语表）───
    # 支持单文件或目录，可选 max_chars 参数（默认 25）
    # Translator.resplit(
    #     r"E:\CourseVideo\.NET内存专家\NET内存专家.7._类和结构体的数组列表.39355417046_trans_full.srt",
    #     max_chars=35,  # 可调整每行字符数
    # )
