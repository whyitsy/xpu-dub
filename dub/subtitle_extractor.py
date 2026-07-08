from pathlib import Path
import torch
# import intel_extension_for_pytorch as ipex


class Extractor:
    def __init__(self, video_path, output_dir=None, device="cpu", force=False):
        self.device = device
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir) if output_dir else self.video_path.parent if self.video_path.is_file() else self.video_path
        self.force = force

    def load_model(self, model_type="whisper"):
        if model_type == "whisper":
            import stable_whisper
            model = stable_whisper.load_model(
                name="medium.en",
                # name="turbo",
                device=self.device,
            )
            self.model = model
            return model
        
    def _transcribe_single(self, video_file: Path):
        print(f"Using device: {next(self.model.parameters()).device}")
        """单个文件的字幕提取逻辑"""
        # 创建与视频同名的子文件夹，所有生成文件放入其中
        output_sub_dir = self.output_dir / video_file.stem
        output_sub_dir.mkdir(parents=True, exist_ok=True)
        output_subtitle_path = output_sub_dir / (video_file.stem + "_ori.srt")
        if output_subtitle_path.exists():
            if not self.force:
                print(f"{video_file.stem}-字幕已存在，跳过提取！")
                return
            else:
                print(f"{video_file.stem}-字幕已存在，强制覆盖提取！")

        # 
        initial_prompt = "Method Table, sharplab.io"

        result = self.model.transcribe(
            str(video_file),
            language="en",
            fp16=False if self.device == "cpu" else True,
            initial_prompt=initial_prompt,
            # 优化重组规则：按句末标点拆分，控制单段时长，减少碎段
            regroup=(
                "cm_sp=.* /。/?/？/.! /。！？/,* /，_"  # 按句末标点切分
                "sg=.3_mg=.3+6_"  # 短片段合并，控制单段时长
                "sp=.* /。/?/？/.!"  # 强制句末标点为分割边界
            ),
            word_timestamps=True,
            verbose=False
        )

        result.to_srt_vtt(str(output_subtitle_path), word_level=False)
        print(f"{video_file.stem}-字幕提取完成！总句子数: {len(result.segments)}")
    
    def extract_subtitle(self):
        assert self.video_path.exists(), f"The video path \'{str(self.video_path)}\' does not exist."
        self.load_model()
        
        if self.video_path.is_file():
            self._transcribe_single(self.video_path)
        else:
            # 批量处理目录下 mp4 文件
            for video_file in sorted(self.video_path.glob("*.mp4")):
                self._transcribe_single(video_file)
 

if __name__ == "__main__":
    video_path = r"E:\CourseVideo\.NET内存专家" 
     
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"Using XPU device: {torch.xpu.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    extractor = Extractor(video_path=video_path, device=device)
    extractor.extract_subtitle()

        