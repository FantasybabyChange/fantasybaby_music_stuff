# music-stuff

一个从音频生成乐谱的实验项目。第一版先聚焦三件事：

- 提取主旋律
- 判断乐曲调性
- 推断核心和弦

当前代码是项目骨架，算法实现会逐步接入。预期流水线：

```text
audio input -> melody transcription -> rhythm quantization -> key/chord analysis -> MusicXML/MIDI/JSON output
```

## 开发

```bash
uv sync
uv run pytest
uv run music-stuff --help
```

## 高质量人声分离

人声/伴奏分离使用可选的 Demucs 后端。安装方式：

```bash
uv sync --extra separation
```

默认分离策略偏向质量而不是速度：

- 模型：`htdemucs_ft`
- shifts：`4`
- overlap：`0.5`
- 自动使用 CUDA（如果 PyTorch 能检测到 GPU）

生成目录会保留 `vocals.wav`、`no_vocals.wav`，以及 Demucs 识别出的单独 stem，例如 `drums.wav`、`bass.wav`、`other.wav`。这些文件可以用来听分离质量，判断人声残留或乐器串音来自哪一轨。

## 目标输出

后续 `transcribe` 命令会输出：

```text
output/
  melody.mid
  score.musicxml
  analysis.json
```

## 后续候选依赖

- `basic-pitch`: 音频转 MIDI / 主旋律提取
- `librosa`: 节拍、onset、chroma 等音频特征
- `pretty_midi`: MIDI 读写和时间处理
- `music21`: 乐理分析和 MusicXML 输出
- `essentia`: 更专业的音频特征、调性和和弦分析候选
