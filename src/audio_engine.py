"""PCM playback with a piecewise DAC presentation clock, not a wall-clock timer.

Only the audio callback owns the render cursor/voices. The UI submits bounded
commands; PortAudio stream APIs are called only from the UI thread. Python is
soft real-time: tiny object/slice allocations remain, but no decoding, file I/O,
ML, sleeps or blocking locks occur in the callback.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .storage import ROOT


class AudioError(RuntimeError):
    """An actionable audio setup or stream error (never silently use fake time)."""


@dataclass
class AudioTrack:
    pcm: np.ndarray
    sample_rate: int
    name: str = "test_track"
    source: str | None = None
    track_id: str = ""
    kind: str = "synthetic"
    bpm: float | None = None
    lead_in: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError("sample_rate должен быть положительным целым числом.")
        pcm = np.asarray(self.pcm, dtype=np.float32)
        if pcm.ndim == 1:
            pcm = np.repeat(pcm[:, None], 2, axis=1)
        elif pcm.ndim == 2 and pcm.shape[1] == 1:
            pcm = np.repeat(pcm, 2, axis=1)
        if pcm.ndim != 2 or pcm.shape[1] != 2 or len(pcm) == 0 or not np.isfinite(pcm).all():
            raise ValueError("Нужен непустой конечный mono/stereo PCM-массив.")
        self.pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        self.pcm.flags.writeable = False
        if not self.track_id:
            self.track_id = hashlib.sha256(self.pcm.data).hexdigest()

    @property
    def duration(self) -> float:
        return len(self.pcm) / self.sample_rate

    def reference(self) -> dict:
        source = self.source
        if source:
            try:
                source = str(Path(source).resolve().relative_to(ROOT))
            except ValueError:
                source = str(Path(source).resolve())
        return {"kind": self.kind, "id": self.track_id, "name": self.name,
                "audio_path": source, "sample_rate": self.sample_rate,
                "duration": self.duration, "bpm": self.bpm,
                "lead_in": self.lead_in, "generator_version": 1 if self.kind == "synthetic" else None}

    @classmethod
    def from_file(cls, path: str | Path, sample_rate: int = 48000) -> "AudioTrack":
        """Decode the complete file and resample before opening the output stream."""
        import soundfile as sf

        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise AudioError(f"Аудиофайл не найден: {path}")
        try:
            pcm, original_rate = sf.read(path, dtype="float32", always_2d=True)
            if pcm.shape[1] > 2:
                pcm = pcm.mean(axis=1, keepdims=True)
            if original_rate != sample_rate:
                import librosa
                pcm = librosa.resample(pcm.T, orig_sr=original_rate, target_sr=sample_rate).T
            digest = hashlib.sha256()
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
            return cls(pcm, sample_rate, path.stem, str(path), digest.hexdigest(), "file", None, 0)
        except (OSError, ValueError, RuntimeError) as error:
            raise AudioError(f"Не удалось прочитать {path.name}: {error}") from error


def synth_click(sample_rate: int = 48000, *, hit: bool = False) -> np.ndarray:
    """Pre-rendered, short, stereo tick; reused by every metronome/hit voice."""
    duration = 0.035 if hit else 0.045
    t = np.arange(round(duration * sample_rate), dtype=np.float64) / sample_rate
    frequency = 1700 if hit else 1100
    signal = (np.sin(2 * np.pi * frequency * t) + 0.3 * np.sin(2 * np.pi * frequency * 2.1 * t))
    signal *= (0.27 if hit else 0.38) * np.exp(-t * 130)
    signal *= np.minimum(1.0, t * sample_rate / 8)
    return np.ascontiguousarray(np.repeat(signal.astype(np.float32)[:, None], 2, axis=1))


def generate_test_track(duration: float = 48.0, sample_rate: int = 48000,
                        bpm: float = 120.0, lead_in: float = 2.0) -> AudioTrack:
    """Deterministic 120-BPM default with obvious beat transients and a soft melody."""
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("duration и bpm должны быть положительными.")
    if not isinstance(sample_rate, int) or sample_rate <= 0 or lead_in < 0:
        raise ValueError("Некорректный sample_rate или lead_in.")
    frames = round(duration * sample_rate)
    if frames < 1:
        raise ValueError("Трек должен содержать хотя бы один frame.")
    pcm = np.zeros((frames, 2), dtype=np.float32)
    interval = 60.0 / bpm
    tick = synth_click(sample_rate)
    scale = [261.6256, 329.6276, 391.9954, 523.2511, 440.0, 391.9954, 329.6276, 293.6648]
    beat = 0
    while (beat_time := lead_in + beat * interval) < duration:
        start = round(beat_time * sample_rate)
        count = min(len(tick), frames - start)
        if count > 0:
            pcm[start:start + count] += tick[:count] * (1.2 if beat % 4 == 0 else 0.8)
        length = min(round(interval * 0.8 * sample_rate), frames - start)
        if length > 0:
            t = np.arange(length) / sample_rate
            envelope = np.minimum(1.0, t / 0.012) * np.exp(-t * 5)
            melody = 0.09 * np.sin(2 * np.pi * scale[beat % len(scale)] * t) * envelope
            pcm[start:start + length] += melody.astype(np.float32)[:, None]
        beat += 1
    fade = min(frames, round(sample_rate * 0.025))
    pcm[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
    return AudioTrack(pcm, sample_rate, bpm=bpm, lead_in=lead_in,
                      track_id=f"synthetic-v1-{sample_rate}-{frames}-{bpm:g}-{lead_in:g}")


def metronome_track(duration: float = 90.0, sample_rate: int = 48000) -> AudioTrack:
    return AudioTrack(np.zeros((round(duration * sample_rate), 2), np.float32),
                      sample_rate, "calibration", track_id="calibration-silence")


@dataclass(frozen=True)
class ClockAnchor:
    dac_time: float
    song_time: float
    state: str
    epoch: int


def position_at(anchors: tuple[ClockAnchor, ...], now: float, duration: float,
                initial: float = 0.0) -> tuple[float, str]:
    """Do not use a future seek/pause anchor to describe the audio still playing."""
    for anchor in reversed(anchors):
        if anchor.dac_time <= now:
            position = anchor.song_time
            if anchor.state == "playing":
                position += now - anchor.dac_time
            return max(0.0, min(duration, position)), anchor.state
    return initial, "starting"


def get_sounddevice():
    try:
        import sounddevice
        return sounddevice
    except (ImportError, OSError) as error:
        raise AudioError("PortAudio недоступен. Linux: установите libportaudio2 "
                         "(и при необходимости libsndfile1). Активируйте .venv. "
                         f"Исходная ошибка: {error}") from error


class AudioEngine:
    """One continuously running output stream, including silence while paused.

    Pause/seek/stop become effective at the next *presentation boundary*, not at
    the UI request time. The DAC timeline retains old segments until they play.
    Closing aborts queued audio; normal transport controls do not restart clocks.
    """
    MAX_VOICES = 32

    def __init__(self, track: AudioTrack, *, device=None, latency="low", blocksize=0,
                 volume: float = 0.65, stream_factory: Callable | None = None):
        self.track = track
        self.device, self.latency, self.blocksize = device, latency, blocksize
        self.volume = min(1.0, max(0.0, float(volume)))
        self._stream_factory = stream_factory
        self._stream = None
        self._commands: deque[tuple[str, float | None, int]] = deque(maxlen=64)
        self._command_version = 0
        self._applied_version = 0
        self._command_dac_time = 0.0
        # Position requests are tracked separately from pause/play commands.
        # Relative seeks must stack against a queued/not-yet-heard seek, but
        # must not reuse an old seek when a later pause is pending.
        self._position_request: tuple[int, float] | None = None
        self._applied_position: tuple[int, float, float] | None = None
        self._hits: deque[int] = deque(maxlen=64)
        self._anchors: deque[ClockAnchor] = deque(maxlen=4096)
        self._render_state = "stopped"
        self._desired_state = "stopped"
        self._render_frame = 0
        self._initial_position = 0.0
        self._closed_position = 0.0
        self._epoch = 0
        self._voice_samples: list[np.ndarray | None] = [None] * self.MAX_VOICES
        self._voice_positions = [0] * self.MAX_VOICES
        self._click = synth_click(track.sample_rate)
        self._hit = synth_click(track.sample_rate, hit=True)
        self._metronome: tuple[bool, float, float] = (False, 120.0, 2.0)
        self.underruns = 0
        self.voice_drops = 0
        self.callback_error: str | None = None
        self.clock_error: str | None = None

    @property
    def requested_state(self) -> str:
        return self._desired_state

    @property
    def transport_pending(self) -> bool:
        if self._stream is None:
            return False
        return self._applied_version < self._command_version or float(self._stream.time) < self._command_dac_time

    def _queue(self, state: str, position: float | None = None) -> None:
        self._command_version += 1
        if position is not None:
            self._position_request = (self._command_version, position)
        self._commands.append((state, position, self._command_version))

    def snapshot(self, seconds_ago: float = 0.0) -> tuple[float, str]:
        if self._stream is None:
            return self._closed_position, self._desired_state
        try:
            # Stream APIs must never be called from _callback.
            now = float(self._stream.time) - max(0.0, seconds_ago)
        except Exception as error:
            raise AudioError(f"Не удалось прочитать аудиочасы: {error}") from error
        # CPython's deque append/popleft and this snapshot are protected by GIL;
        # there is no blocking Python mutex on the audio thread.
        position, state = position_at(tuple(self._anchors), now, self.track.duration, self._initial_position)
        if seconds_ago > 0 and now < self._command_dac_time:
            state = "transition"  # stale input from before the current transport epoch
        return position, state

    @property
    def song_time(self) -> float:
        return self.snapshot()[0]

    @property
    def state(self) -> str:
        return self.snapshot()[1]

    @property
    def output_latency(self) -> float | None:
        return float(self._stream.latency) if self._stream is not None else None

    def check_health(self) -> None:
        if self.callback_error or self.clock_error:
            raise AudioError(self.callback_error or self.clock_error)
        if self._stream is not None and not self._stream.active:
            raise AudioError("Аудиопоток остановился. Проверьте устройство и его настройки.")

    def _open(self) -> None:
        if self._stream is not None:
            return
        factory = self._stream_factory or get_sounddevice().OutputStream
        stream = None
        try:
            stream = factory(samplerate=self.track.sample_rate, channels=2, dtype="float32",
                             device=self.device, latency=self.latency, blocksize=self.blocksize,
                             callback=self._callback)
            self._stream = stream
            stream.start()
        except Exception as error:
            self._stream = None
            if stream is not None:
                stream.close()
            raise AudioError("Не удалось открыть аудиовыход. Проверьте --diagnose, "
                             "output_device и sample_rate в config.json. "
                             f"Подробности: {error}") from error

    def play(self) -> None:
        self._desired_state = "playing"
        self._queue("playing")
        self._open()

    def pause(self) -> None:
        if self._stream is not None:
            self._desired_state = "paused"
            self._queue("paused")

    def toggle_pause(self) -> None:
        if self._desired_state == "playing":
            self.pause()
        else:
            self.play()

    def stop(self) -> None:
        self._desired_state = "stopped"
        if self._stream is None:
            # seek() is also allowed before opening a device. Reset its render
            # cursor and initial anchor, not just the position exposed by UI.
            self._render_frame = 0
            self._initial_position = self._closed_position = 0.0
            self._commands.clear()
            self._position_request = self._applied_position = None
        else:
            self._queue("stopped", 0.0)

    def seek(self, seconds: float) -> None:
        if not math.isfinite(seconds):
            raise ValueError("Позиция seek должна быть конечной.")
        seconds = max(0.0, min(self.track.duration, seconds))
        if self._stream is None:
            self._render_frame = round(seconds * self.track.sample_rate)
            self._initial_position = self._closed_position = seconds
        else:
            self._queue(self._desired_state, seconds)

    def rewind(self, seconds: float = 5.0) -> float:
        """Queue a sample-aligned backwards seek, retaining play/pause state.

        Consecutive presses stack even before the callback consumes commands or
        the DAC presents them. Once a seek is heard, base the next rewind on the
        current audible position, not that old target (including during pause).
        Returns the raw track target; chart/input correction belongs to the UI.
        """
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Длительность отката должна быть положительной и конечной.")
        if self._stream is None:
            base = self._closed_position
        else:
            try:
                now = float(self._stream.time)
            except Exception as error:
                raise AudioError(f"Не удалось прочитать аудиочасы: {error}") from error
            if not math.isfinite(now):
                raise AudioError("Backend вернул неконечные аудиочасы.")
            request, applied = self._position_request, self._applied_position
            if request is not None and (applied is None or applied[0] < request[0] or now < applied[2]):
                base = request[1]
            else:
                base, _ = position_at(tuple(self._anchors), now, self.track.duration, self._initial_position)
        target = round(max(0.0, base - seconds) * self.track.sample_rate) / self.track.sample_rate
        self.seek(target)
        return target

    def set_metronome(self, enabled: bool, bpm: float = 120.0, first_beat: float = 2.0) -> None:
        if not math.isfinite(bpm) or not 20 <= bpm <= 400 or not math.isfinite(first_beat) or first_beat < 0:
            raise ValueError("Некорректные параметры метронома.")
        self._metronome = (bool(enabled), bpm, first_beat)

    def trigger_hit(self) -> None:
        """Earliest unrendered output buffer, NOT zero physical output latency."""
        if self._desired_state == "playing":
            self._hits.append(self._epoch)

    def _clear_voices(self) -> None:
        for i in range(self.MAX_VOICES):
            self._voice_samples[i] = None
            self._voice_positions[i] = 0
        self._hits.clear()

    def _spill_voice(self, sample: np.ndarray, position: int) -> None:
        for i in range(self.MAX_VOICES):
            if self._voice_samples[i] is None:
                self._voice_samples[i] = sample
                self._voice_positions[i] = position
                return
        self.voice_drops += 1

    def _mix_at(self, out: np.ndarray, sample: np.ndarray, offset: int) -> None:
        count = min(len(sample), len(out) - offset)
        if count > 0:
            np.add(out[offset:offset + count], sample[:count], out=out[offset:offset + count])
        if count < len(sample):
            self._spill_voice(sample, count)

    def _callback(self, outdata, frames, time_info, status) -> None:
        outdata.fill(0)
        try:
            dac_time = float(time_info.outputBufferDacTime)
            if not math.isfinite(dac_time) or dac_time <= 0:
                self.clock_error = "Backend не предоставляет пригодный outputBufferDacTime; таймер не подменён wall clock."
                return
            if self._anchors and dac_time < self._anchors[-1].dac_time - 1e-6:
                self.clock_error = "Аудиочасы backend пошли назад. Нужен перезапуск потока."
                return
            if getattr(status, "output_underflow", False):
                self.underruns += 1
                self._epoch += 1
            changed = False
            position_change = None
            while self._commands:
                state, position, version = self._commands.popleft()
                self._applied_version = version
                if position is not None:
                    self._render_frame = min(len(self.track.pcm), round(position * self.track.sample_rate))
                    position_change = (version, self._render_frame / self.track.sample_rate, dac_time)
                if self._render_state == "finished" and state == "playing" and position is None:
                    self._render_frame = 0
                self._render_state = state
                changed = True
            request = self._position_request
            if (position_change is None and request is not None and self._applied_version >= request[0]
                    and (self._applied_position is None or self._applied_position[0] < request[0])):
                # A burst of >64 state-only commands may evict a seek from the
                # bounded deque. Retain its latest position intent, not a ghost
                # pending target that never reaches the DAC.
                self._render_frame = min(len(self.track.pcm), round(request[1] * self.track.sample_rate))
                position_change = (request[0], self._render_frame / self.track.sample_rate, dac_time)
                changed = True
            if changed:
                self._command_dac_time = dac_time
                self._epoch += 1
                self._clear_voices()
            if self._render_state != "playing":
                if changed or not self._anchors:
                    self._anchors.append(ClockAnchor(dac_time, self._render_frame / self.track.sample_rate,
                                                     self._render_state, self._epoch))
                if position_change is not None:
                    self._applied_position = position_change
                return

            start = self._render_frame
            count = min(frames, len(self.track.pcm) - start)
            self._anchors.append(ClockAnchor(dac_time, start / self.track.sample_rate, "playing", self._epoch))
            # Publish only after the matching DAC anchor is visible to the UI.
            if position_change is not None:
                self._applied_position = position_change
            if count:
                outdata[:count] = self.track.pcm[start:start + count]
            for i in range(self.MAX_VOICES):
                sample = self._voice_samples[i]
                if sample is not None:
                    position = self._voice_positions[i]
                    length = min(frames, len(sample) - position)
                    np.add(outdata[:length], sample[position:position + length], out=outdata[:length])
                    self._voice_positions[i] += length
                    if self._voice_positions[i] >= len(sample):
                        self._voice_samples[i] = None
            # Bounded queue: a stalled UI cannot create an unbounded callback workload.
            for _ in range(64):
                try:
                    epoch = self._hits.popleft()
                except IndexError:
                    break
                if epoch == self._epoch:
                    self._mix_at(outdata, self._hit, 0)
            enabled, bpm, first = self._metronome
            if enabled and count:
                interval = 60.0 / bpm
                beat = max(0, math.floor((start / self.track.sample_rate - first) / interval))
                while (frame := round((first + beat * interval) * self.track.sample_rate)) < start + count:
                    if frame >= start:
                        self._mix_at(outdata, self._click, frame - start)
                    beat += 1
            np.multiply(outdata, self.volume, out=outdata)
            np.clip(outdata, -1.0, 1.0, out=outdata)
            self._render_frame += count
            if self._render_frame >= len(self.track.pcm):
                self._render_state = "finished"
                self._anchors.append(ClockAnchor(dac_time + count / self.track.sample_rate,
                                                 self.track.duration, "finished", self._epoch))
                self._clear_voices()
        except Exception as error:
            # Exceptions in a PortAudio callback otherwise don't propagate to the UI.
            outdata.fill(0)
            self.callback_error = f"Ошибка audio callback: {type(error).__name__}: {error}"

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._closed_position = self.song_time
            finally:
                stream, self._stream = self._stream, None
                try:
                    stream.abort()
                finally:
                    stream.close()
        self._desired_state = "stopped"
        self._commands.clear()
        self._position_request = self._applied_position = None
        self._anchors.clear()
        self._clear_voices()

    def __enter__(self) -> "AudioEngine":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
