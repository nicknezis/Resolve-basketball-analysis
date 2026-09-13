"""FFmpeg helpers: binary discovery, encoder probing, and a raw-frame video writer.

The pip ``opencv-python`` wheels ship without an H.264/H.265 encoder, so
``cv2.VideoWriter`` can only produce MPEG-4 Part 2 (``mp4v``).  This module
pipes BGR frames straight into an ``ffmpeg`` subprocess instead, which lets us
pick a real codec (hardware VideoToolbox on macOS, libx264/libx265 elsewhere)
and mux the source audio in the same pass.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Preference order per requested codec family.  First available encoder wins.
_ENCODER_CANDIDATES: dict[str, tuple[str, ...]] = {
    "h264": ("h264_videotoolbox", "h264_nvenc", "libx264"),
    "hevc": ("hevc_videotoolbox", "hevc_nvenc", "libx265"),
}

# Bitrate used for hardware encoders (which don't expose a CRF-style knob),
# expressed in kbit/s per megapixel of output.  1280x720 ≈ 0.92 MP → ~11 Mbit/s
# for H.264; HEVC gets ~40% less for the same visual quality.
_DEFAULT_KBPS_PER_MEGAPIXEL = {"h264": 12_000, "hevc": 7_000}


@lru_cache(maxsize=1)
def find_ffmpeg() -> str | None:
    """Return the ffmpeg executable path, or None if not on PATH."""
    return shutil.which("ffmpeg")


@lru_cache(maxsize=1)
def find_ffprobe() -> str | None:
    """Return the ffprobe executable path, or None if not on PATH."""
    return shutil.which("ffprobe")


@lru_cache(maxsize=1)
def available_encoders() -> frozenset[str]:
    """Parse ``ffmpeg -encoders`` once and return the set of encoder names."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return frozenset()
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.warning("Could not query ffmpeg encoders: %s", exc)
        return frozenset()

    names: set[str] = set()
    for line in out.splitlines():
        # Encoder lines look like " V....D libx264  libx264 H.264 / AVC ..."
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            names.add(parts[1])
    return frozenset(names)


def resolve_encoder(codec: str, encoders: frozenset[str] | None = None) -> str | None:
    """Pick the best available ffmpeg encoder for a codec family.

    Args:
        codec: "h264" or "hevc".
        encoders: Override for the available encoder set (used by tests).

    Returns:
        Encoder name, or None if no candidate is available.
    """
    candidates = _ENCODER_CANDIDATES.get(codec)
    if candidates is None:
        raise ValueError(f"Unknown codec {codec!r}; expected one of {list(_ENCODER_CANDIDATES)}")
    encoders = available_encoders() if encoders is None else encoders
    for name in candidates:
        if name in encoders:
            return name
    return None


@lru_cache(maxsize=32)
def encoder_works(encoder: str, width: int, height: int) -> bool:
    """Encode one synthetic frame to check the encoder accepts this size.

    Hardware encoders can be present in ``ffmpeg -encoders`` yet refuse to
    open a session (e.g. ``hevc_videotoolbox`` on tiny frames or under
    Rosetta).  A one-frame probe costs ~100 ms and lets us fall through to a
    software encoder instead of failing after rendering a whole clip.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return False
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=size={width}x{height}:rate=30",
        "-frames:v", "1",
        *encoder_args(encoder, width, height, None),
        "-f", "null", "-",
    ]
    try:
        ok = subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    if not ok:
        logger.debug("Encoder %s rejected %dx%d in probe", encoder, width, height)
    return ok


def pick_encoder(codec: str, width: int, height: int) -> str | None:
    """First encoder for ``codec`` that is installed *and* passes a probe."""
    candidates = _ENCODER_CANDIDATES.get(codec)
    if candidates is None:
        raise ValueError(f"Unknown codec {codec!r}; expected one of {list(_ENCODER_CANDIDATES)}")
    installed = available_encoders()
    for name in candidates:
        if name in installed and encoder_works(name, width, height):
            return name
    return None


def encoder_args(encoder: str, width: int, height: int, quality: int | None) -> list[str]:
    """Build the ``-c:v ...`` argument list for an encoder.

    ``quality`` is a CRF value for software encoders and a bitrate in kbit/s
    for hardware encoders.  None picks a sensible default for each.
    """
    args = ["-c:v", encoder]
    if encoder.endswith("_videotoolbox") or encoder.endswith("_nvenc"):
        family = "hevc" if encoder.startswith("hevc") else "h264"
        megapixels = (width * height) / 1_000_000
        kbps = quality if quality is not None else int(_DEFAULT_KBPS_PER_MEGAPIXEL[family] * megapixels)
        args += ["-b:v", f"{kbps}k"]
        if encoder.endswith("_videotoolbox"):
            # Without this VideoToolbox may emit keyframes only every ~10 s,
            # which makes scrubbing in an NLE sluggish.  -allow_sw lets it use
            # Apple's software path when the hardware session is unavailable.
            args += ["-g", "60", "-allow_sw", "1"]
    elif encoder == "libx264":
        args += ["-crf", str(quality if quality is not None else 18), "-preset", "fast"]
    elif encoder == "libx265":
        args += ["-crf", str(quality if quality is not None else 20), "-preset", "fast"]
    if encoder.startswith("hevc") or encoder == "libx265":
        # QuickTime / Resolve only recognise HEVC-in-MP4 with the hvc1 tag.
        args += ["-tag:v", "hvc1"]
    return args


def run_ffmpeg(args: list[str], description: str = "ffmpeg") -> subprocess.CompletedProcess:
    """Run ffmpeg with the given arguments, raising with stderr on failure."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg not found on PATH")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", *args]
    logger.debug("%s: %s", description, " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{description} failed (exit {proc.returncode}): {proc.stderr.strip()[-2000:]}"
        )
    return proc


class FfmpegVideoWriter:
    """Encode BGR uint8 frames to a video file via an ffmpeg subprocess.

    Usage::

        with FfmpegVideoWriter(path, fps, (w, h), codec="h264",
                               audio_source=src, audio_start_sec=1.5,
                               audio_duration_sec=42.0) as writer:
            writer.write(frame)

    Audio, if given, is transcoded to AAC so PCM sources (Sony camera
    originals) produce a valid MP4.
    """

    def __init__(
        self,
        path: Path | str,
        fps: float,
        size: tuple[int, int],
        codec: str = "hevc",
        quality: int | None = None,
        audio_source: Path | str | None = None,
        audio_start_sec: float | None = None,
        audio_duration_sec: float | None = None,
    ) -> None:
        self.path = Path(path)
        self.fps = fps
        self.width, self.height = size
        self.codec = codec
        self.frames_written = 0

        encoder = pick_encoder(codec, self.width, self.height)
        if encoder is None:
            raise RuntimeError(
                f"No working ffmpeg encoder for {codec!r} at {self.width}x{self.height} "
                f"(installed: {sorted(available_encoders()) or 'none'})"
            )
        self.encoder = encoder

        cmd = [
            find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
            # Video from stdin as raw BGR frames
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}", "-r", f"{fps:.6f}",
            "-i", "pipe:0",
        ]
        if audio_source is not None:
            if audio_start_sec is not None:
                cmd += ["-ss", f"{audio_start_sec:.6f}"]
            if audio_duration_sec is not None:
                cmd += ["-t", f"{audio_duration_sec:.6f}"]
            cmd += ["-i", str(audio_source)]
        cmd += ["-map", "0:v:0"]
        if audio_source is not None:
            # No -shortest: the audio is already cut to the clip with -ss/-t,
            # and -shortest would drop the last video frame whenever the
            # audio stream ends a few ms early.
            cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"]
        cmd += encoder_args(encoder, self.width, self.height, quality)
        cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]

        logger.debug("FfmpegVideoWriter: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        """Write one BGR uint8 frame of the configured size."""
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            raise ValueError(
                f"Frame size {w}x{h} does not match writer size {self.width}x{self.height}"
            )
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            # ffmpeg died — surface its stderr instead of a bare pipe error
            self._raise_from_process("ffmpeg exited early")
        self.frames_written += 1

    def close(self) -> None:
        """Flush, wait for ffmpeg to finish, and raise if it failed."""
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except BrokenPipeError:
            pass
        self._proc.wait()
        proc, self._proc = self._proc, None
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            raise RuntimeError(
                f"ffmpeg encode to {self.path} failed (exit {proc.returncode}): "
                f"{stderr.strip()[-2000:]}"
            )

    def _raise_from_process(self, message: str) -> None:
        proc, self._proc = self._proc, None
        proc.wait()
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise RuntimeError(f"{message}: {stderr.strip()[-2000:]}")

    def __enter__(self) -> "FfmpegVideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None and self._proc is not None:
            # Don't mask the original exception with an ffmpeg error
            try:
                self._proc.kill()
            finally:
                self._proc = None
            return
        self.close()
