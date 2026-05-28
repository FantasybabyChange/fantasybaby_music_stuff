# Music Stuff Project — Optimization Summary

## Phase 1: Code Quality

### 1A. Type Hints Added (8 locations)

| File | Parameter | Type Added |
|------|-----------|------------|
| `pipeline.py:122` | `action` | `Callable[[], _T]` with `TypeVar`, return `-> _T` |
| `pipeline.py:129` | `mixed_audio` | `PreparedAudio` |
| `pipeline.py:189` | `stem_audio` | `PreparedAudio` |
| `rhythm.py:113` | `tempo` | `object`, return `-> float` |
| `source.py:252` | `torch` | `Any` |
| `source.py:311` | return type | `-> tuple[Any, int]` |
| `source.py:327` | `np` | `Any` |
| `melody.py:265` | `pitches, magnitudes, librosa, np` | all `Any` |

### 1B. Magic Numbers → Named Constants

Created `src/music_stuff/constants.py` with 14 named constants:

| Constant | Value | Used In |
|----------|-------|---------|
| `REGISTER_PITCH_SPAN` | 24.0 | melody.py |
| `REGISTER_SCORE_CAP` | 0.75 | melody.py |
| `CONTINUITY_PITCH_SPAN` | 12.0 | melody.py |
| `CONTINUITY_SCORE_CAP` | 0.85 | melody.py |
| `MERGE_GAP_RATIO` | 0.51 | rhythm.py |
| `VOICE_VS_ACCOMPANIMENT_THRESHOLD` | 0.35 | pipeline.py |
| `MELODY_SCORE_DENSITY_DIVISOR` | 32.0 | pipeline.py |
| `MELODY_SCORE_DURATION_DIVISOR` | 12.0 | pipeline.py |
| `MELODY_SCORE_DURATION_WEIGHT` | 0.7 | pipeline.py |
| `MELODY_SCORE_DENSITY_WEIGHT` | 0.3 | pipeline.py |
| `MELODY_SCORE_MINIMUM` | 0.05 | pipeline.py |
| `DISPLAY_OFFSET_GAP_SECONDS` | 2.5 | score.py |
| `DISPLAY_OFFSET_PHRASE_THRESHOLD` | 4 | score.py |
| `DISPLAY_OFFSET_SCORE_RATIO` | 0.8 | score.py |

### 1C. Docstrings Added

- `cli.py`: `build_parser`, `_handle_plan`, `_handle_transcribe`, `_handle_ui`
- `harmony.py`: `KeyAnalyzer.analyze`
- `score.py`: `ScoreExporter.export_analysis_json`, `ScoreExporter.export_jianpu`, `export_midi`, `export_musicxml`
- `harmony.py`: `ChordAnalyzer.analyze` — added stub notice with logging

### 1D. Error Handling Fixes

- `audio.py:105`: Added `timeout=120` to `subprocess.run` with `TimeoutExpired` handling
- `cli.py:149`: Replaced `hasattr(args, "func")` with `parser.set_defaults(func=None)` + `if args.func is None:`
- `test_audio.py`: Updated 3 `fake_run` mocks to accept `timeout=None` keyword argument

### 1E. Encapsulation Fix

Extracted `_frames_to_notes(frames, min_note_duration)` as a module-level function in `melody.py`. Both `SimplePitchMelodyTranscriber._frames_to_notes` and `MixedAudioMelodyTranscriber._frames_to_notes` now delegate to it, eliminating the `self.fallback._frames_to_notes(frames)` cross-class private access.

### 1F. `__all__` Declarations

Added `__all__` to all 11 modules: `__init__.py`, `audio.py`, `cli.py`, `harmony.py`, `melody.py`, `models.py`, `pipeline.py`, `rhythm.py`, `score.py`, `source.py`, `web.py`

---

## Phase 2: Performance

### 2A. `_decode_pcm` Vectorization

**Before:** Pure-Python byte loop iterating sample-by-sample with `int.from_bytes()`.
**After:** `numpy.frombuffer` with dtype-specific decoding for 1/2/4-byte samples. 3-byte samples still use a Python loop (rare format).

```python
# Before (audio.py:148-164)
for index in range(0, len(raw), sample_width):
    chunk = raw[index : index + sample_width]
    integer = int.from_bytes(chunk, "little", signed=True)
    values.append(max(-1.0, min(1.0, integer / max_value)))

# After
np.clip(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, -1.0, 1.0)
```

### 2B. `_downmix` Vectorization

**Before:** Python loop summing channels per frame.
**After:** `numpy.reshape(-1, channels).mean(axis=1)` — single vectorized operation.

### 2C. `PreparedAudio.samples` Type Change

**Before:** `samples: tuple[float, ...] = ()` — stores ~120MB of Python float objects for a 3-minute file.
**After:** `samples: Any = ()` — stores numpy ndarray directly, ~10-50x less memory.

Consumers (`rhythm.py`, `melody.py`) already call `np.asarray()` which is a no-op on ndarray.

### 2D. Autocorrelation Vectorization

**Before:** Nested Python loop — ~234 million scalar operations for a 3-minute file.
**After:** Inner loop replaced with `numpy.dot` calls. Outer lag loop preserved with stride optimization.

### 2E. Demucs Model Caching

**Before:** `get_model(model_name)` called on every `separate()` invocation.
**After:** `@functools.lru_cache(maxsize=2)` on `_get_cached_model()` — model loaded once per name.

### 2F. ffmpeg Binary Resolution Caching

**Before:** `shutil.which("ffmpeg")` called on every `separate()` invocation.
**After:** `@functools.lru_cache(maxsize=1)` on `_find_system_ffmpeg()`.

### 2G. Beat Offset Modulo

**Before:** `while` loop subtracting `beat_seconds` from offset.
**After:** `offset = beat_times[0] % beat_seconds` — single modulo operation.

### 2H. Numpy Array Compatibility

Updated `if not audio.samples` checks in `rhythm.py` and `melody.py` to `if not len(audio.samples)` to work with numpy arrays (which raise `ValueError` on boolean evaluation of multi-element arrays).

---

## Phase 3: Architecture

### 3A. Web.py Template Extraction

**Before:** 1386-line monolith with 720-line inline HTML/CSS/JS f-string.
**After:** HTML template extracted to `src/music_stuff/templates/index.html`, loaded via `string.Template` with `@functools.lru_cache`. Web.py reduced to ~350 lines.

Template uses `$variable` syntax for 6 dynamic placeholders: `$logo_href`, `$video_href`, `$compute_mode_html`, `$message_html`, `$history_html`, `$result_html`.

Added to `pyproject.toml` force-include for wheel distribution.

### 3B. CLI Lazy Imports

**Before:** `from music_stuff.web import UIConfig, run_ui` at module level — loads web.py and all dependencies even for `music-stuff plan`.
**After:** Import moved inside `_handle_ui()` function — only loads when web UI is requested.

### 3C. Dead Code Cleanup

- `harmony.py`: `ChordAnalyzer.analyze` now logs "Chord analysis not yet implemented" instead of silently returning empty tuple.
- `score.py`: Added docstrings to `export_midi` and `export_musicxml` stubs explaining they are planned features.

---

## Test Results

All 46 tests pass after each phase:
- Phase 1: 46 passed
- Phase 2: 46 passed
- Phase 3: 46 passed

## Files Modified

| File | Phases |
|------|--------|
| `src/music_stuff/audio.py` | 1D, 2A, 2B, 2C |
| `src/music_stuff/cli.py` | 1C, 1D, 3B |
| `src/music_stuff/constants.py` | 1B (new) |
| `src/music_stuff/harmony.py` | 1C, 3C |
| `src/music_stuff/melody.py` | 1A, 1B, 1E, 2D, 2H |
| `src/music_stuff/models.py` | 1F |
| `src/music_stuff/pipeline.py` | 1A, 1B, 1F |
| `src/music_stuff/rhythm.py` | 1A, 1B, 2G, 2H |
| `src/music_stuff/score.py` | 1B, 1C, 1F, 3C |
| `src/music_stuff/source.py` | 1A, 1F, 2E, 2F |
| `src/music_stuff/web.py` | 1F, 3A |
| `src/music_stuff/templates/index.html` | 3A (new) |
| `pyproject.toml` | 3A |
| `tests/test_audio.py` | 1D, 2C |
