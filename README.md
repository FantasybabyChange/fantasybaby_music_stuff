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

启动本地页面：

```bash
uv run music-stuff ui
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

## 页面计算模式

网页上传区提供计算模式选项：

- `均衡`：页面默认值。Demucs 仍会优先使用 GPU，但把 `shifts` 降到 `2`、`overlap` 降到 `0.25`，减少重复推理，后续主旋律、节奏、调性和和弦分析继续在 CPU 上完成。
- `GPU`：高质量模式，强制使用 CUDA，参数为 `shifts=4`、`overlap=0.5`。如果 PyTorch 看不到可用 GPU，会直接报错，避免用户误以为跑在显卡上。
- `CPU`：强制 Demucs 在 CPU 上跑，显卡占用为 0，但大音频会明显更慢。
- `自动`：保持高质量参数，设备由 PyTorch 自动选择；有 CUDA 时通常走 GPU，没有 CUDA 时走 CPU。

说明：Demucs 的一次模型推理不能真正拆成“一半 GPU、一半 CPU”同时执行。当前的“均衡”模式采用的是低负载 GPU 分离 + CPU 后处理，目标是避免 GPU 长时间满负荷，同时保留比纯 CPU 更可用的速度。

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
