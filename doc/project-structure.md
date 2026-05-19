# Project Structure

```mermaid
flowchart TD
    user["User / Browser"] --> web["web.py\nLocal Web UI"]
    user --> cli["cli.py\nCommand Line"]

    web --> pipeline["pipeline.py\nTranscription Pipeline"]
    cli --> pipeline

    pipeline --> audio["audio.py\nWAV / MP3 / FLAC decode"]
    pipeline --> melody["melody.py\nMelody extraction"]
    pipeline --> rhythm["rhythm.py\nRhythm quantization"]
    pipeline --> harmony["harmony.py\nKey / chord analysis"]
    pipeline --> score["score.py\nJianpu + JSON export"]

    audio --> models["models.py\nShared data models"]
    melody --> models
    rhythm --> models
    harmony --> models
    score --> models

    score --> output["output/\nGenerated artifacts"]
```

## Main Folders

- `src/music_stuff/`: application source code
- `tests/`: unit and smoke tests
- `output/`: generated UI uploads, Jianpu files, analysis JSON, and logs
- `doc/`: project notes and diagrams

## Main Flow

```text
audio upload/input
  -> decode audio
  -> extract melody
  -> quantize rhythm
  -> estimate key
  -> export Jianpu and analysis JSON
  -> show/download in Web UI
```
