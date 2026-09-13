"""Tests for the ffmpeg-backed video writer and encoder selection."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from src.analysis import video_io
from src.analysis.video_io import FfmpegVideoWriter, encoder_args, resolve_encoder


class TestEncoderSelection:
    def test_prefers_videotoolbox_for_h264(self):
        enc = frozenset({"libx264", "h264_videotoolbox", "mpeg4"})
        assert resolve_encoder("h264", enc) == "h264_videotoolbox"

    def test_falls_back_to_libx264(self):
        assert resolve_encoder("h264", frozenset({"libx264", "mpeg4"})) == "libx264"

    def test_hevc_software_fallback(self):
        assert resolve_encoder("hevc", frozenset({"libx265"})) == "libx265"

    def test_none_when_unavailable(self):
        assert resolve_encoder("h264", frozenset({"mpeg4"})) is None

    def test_unknown_codec_raises(self):
        with pytest.raises(ValueError):
            resolve_encoder("av1", frozenset())

    def test_hardware_args_use_bitrate(self):
        args = encoder_args("h264_videotoolbox", 1280, 720, None)
        assert "-b:v" in args and "-crf" not in args

    def test_software_args_use_crf(self):
        args = encoder_args("libx264", 1280, 720, 20)
        assert args[args.index("-crf") + 1] == "20"

    def test_hevc_gets_hvc1_tag(self):
        args = encoder_args("hevc_videotoolbox", 1280, 720, None)
        assert args[args.index("-tag:v") + 1] == "hvc1"


@pytest.mark.skipif(video_io.find_ffmpeg() is None, reason="ffmpeg not installed")
class TestFfmpegVideoWriter:
    @pytest.mark.parametrize("codec", ["h264", "hevc"])
    def test_writes_playable_file(self, tmp_path, codec):
        w, h, n = 320, 180, 12
        if video_io.pick_encoder(codec, w, h) is None:
            pytest.skip(f"no working {codec} encoder in this ffmpeg build")
        out = tmp_path / f"clip.{codec}.mp4"
        with FfmpegVideoWriter(out, fps=30.0, size=(w, h), codec=codec) as writer:
            for i in range(n):
                frame = np.full((h, w, 3), i * 20, dtype=np.uint8)
                writer.write(frame)
        assert writer.frames_written == n
        assert out.exists() and out.stat().st_size > 0

        ffprobe = video_io.find_ffprobe()
        if ffprobe:
            probe = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name,nb_frames", "-of", "csv=p=0", str(out)],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            assert codec in probe
            assert str(n) in probe

    def test_wrong_frame_size_raises(self, tmp_path):
        if video_io.pick_encoder("h264", 64, 64) is None:
            pytest.skip("no working h264 encoder")
        writer = FfmpegVideoWriter(tmp_path / "x.mp4", fps=30.0, size=(64, 64), codec="h264")
        try:
            with pytest.raises(ValueError):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
        finally:
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.close()
