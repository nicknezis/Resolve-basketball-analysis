"""Tests for src.analysis.color — 3D LUT parsing and frame application."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src.analysis.color import FrameLUT, parse_cube_file


def _write_cube(path: Path, size: int, values: list[list[float]]) -> None:
    """Write a minimal .cube file."""
    lines = [f"LUT_3D_SIZE {size}\n"]
    for r, g, b in values:
        lines.append(f"{r:.6f} {g:.6f} {b:.6f}\n")
    path.write_text("".join(lines))


def _identity_values(size: int) -> list[list[float]]:
    """Generate identity LUT values in .cube format order.

    .cube format: R varies fastest (innermost), G middle, B slowest.
    Output values are (R, G, B) matching the input coordinates.
    """
    values = []
    for b in range(size):
        for g in range(size):
            for r in range(size):
                values.append([
                    r / (size - 1),
                    g / (size - 1),
                    b / (size - 1),
                ])
    return values


# ---- parse_cube_file ----


def test_parse_cube_file():
    """Parse a minimal 2x2x2 .cube file, verify shape and values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cube_path = Path(tmpdir) / "test.cube"
        # 2x2x2 = 8 entries in .cube order (R fastest, B slowest):
        # (B=0,G=0,R=0), (B=0,G=0,R=1), (B=0,G=1,R=0), (B=0,G=1,R=1),
        # (B=1,G=0,R=0), (B=1,G=0,R=1), (B=1,G=1,R=0), (B=1,G=1,R=1)
        values = [
            [0.0, 0.0, 0.0],  # B=0,G=0,R=0 → black
            [1.0, 0.0, 0.0],  # B=0,G=0,R=1 → red
            [0.0, 1.0, 0.0],  # B=0,G=1,R=0 → green
            [1.0, 1.0, 0.0],  # B=0,G=1,R=1 → yellow
            [0.0, 0.0, 1.0],  # B=1,G=0,R=0 → blue
            [1.0, 0.0, 1.0],  # B=1,G=0,R=1 → magenta
            [0.0, 1.0, 1.0],  # B=1,G=1,R=0 → cyan
            [1.0, 1.0, 1.0],  # B=1,G=1,R=1 → white
        ]
        _write_cube(cube_path, 2, values)

        lut, size = parse_cube_file(cube_path)
        assert size == 2
        assert lut.shape == (2, 2, 2, 3)
        assert lut.dtype == np.float32

        # After transpose, lut[r, g, b] = output for input (R, G, B)
        np.testing.assert_allclose(lut[0, 0, 0], [0.0, 0.0, 0.0])  # black
        np.testing.assert_allclose(lut[1, 1, 1], [1.0, 1.0, 1.0])  # white
        np.testing.assert_allclose(lut[1, 0, 0], [1.0, 0.0, 0.0])  # red
        np.testing.assert_allclose(lut[0, 1, 0], [0.0, 1.0, 0.0])  # green
        np.testing.assert_allclose(lut[0, 0, 1], [0.0, 0.0, 1.0])  # blue


def test_parse_cube_from_zip():
    """Parse a .cube file extracted from a ZIP archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a .cube file
        cube_path = Path(tmpdir) / "inner.cube"
        values = _identity_values(2)
        _write_cube(cube_path, 2, values)

        # Wrap it in a ZIP
        zip_path = Path(tmpdir) / "test.lut"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(cube_path, "SomeFolder/my_lut.cube")

        lut, size = parse_cube_file(zip_path)
        assert size == 2
        assert lut.shape == (2, 2, 2, 3)


def test_parse_cube_with_comments_and_headers():
    """Parser should skip comments and header keywords."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cube_path = Path(tmpdir) / "test.cube"
        content = (
            "# This is a comment\n"
            'TITLE "My LUT"\n'
            "DOMAIN_MIN 0.0 0.0 0.0\n"
            "DOMAIN_MAX 1.0 1.0 1.0\n"
            "LUT_3D_SIZE 2\n"
            "0.0 0.0 0.0\n"
            "0.0 0.0 1.0\n"
            "0.0 1.0 0.0\n"
            "0.0 1.0 1.0\n"
            "1.0 0.0 0.0\n"
            "1.0 0.0 1.0\n"
            "1.0 1.0 0.0\n"
            "1.0 1.0 1.0\n"
        )
        cube_path.write_text(content)

        lut, size = parse_cube_file(cube_path)
        assert size == 2
        assert lut.shape == (2, 2, 2, 3)


def test_parse_cube_missing_size():
    """Should raise ValueError if LUT_3D_SIZE is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cube_path = Path(tmpdir) / "bad.cube"
        cube_path.write_text("0.0 0.0 0.0\n1.0 1.0 1.0\n")

        with pytest.raises(ValueError, match="LUT_3D_SIZE"):
            parse_cube_file(cube_path)


def test_parse_cube_wrong_count():
    """Should raise ValueError if entry count doesn't match size^3."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cube_path = Path(tmpdir) / "bad.cube"
        cube_path.write_text("LUT_3D_SIZE 2\n0.0 0.0 0.0\n")

        with pytest.raises(ValueError, match="Expected 8"):
            parse_cube_file(cube_path)


# ---- FrameLUT ----


def test_lut_passthrough():
    """FrameLUT(None).apply(frame) returns the exact same array object."""
    lut = FrameLUT(None)
    frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    result = lut.apply(frame)
    assert result is frame


def test_lut_identity():
    """A .cube file with identity mapping should produce (nearly) unchanged output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cube_path = Path(tmpdir) / "identity.cube"
        size = 17  # typical small identity LUT
        values = _identity_values(size)
        _write_cube(cube_path, size, values)

        lut = FrameLUT(cube_path)

        # Create a test frame with known colors
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        frame[0:10, :] = [255, 0, 0]    # blue (BGR)
        frame[10:20, :] = [0, 255, 0]   # green
        frame[20:30, :] = [0, 0, 255]   # red
        frame[30:40, :] = [128, 128, 128]  # gray
        frame[40:50, :] = [0, 0, 0]     # black

        result = lut.apply(frame)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8
        # Identity LUT should preserve values within rounding tolerance
        np.testing.assert_allclose(result.astype(float), frame.astype(float), atol=2)


def test_lut_transforms():
    """A .cube file that maps all inputs to white should produce all-white output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cube_path = Path(tmpdir) / "white.cube"
        size = 2
        # Every LUT entry maps to white
        values = [[1.0, 1.0, 1.0]] * (size ** 3)
        _write_cube(cube_path, size, values)

        lut = FrameLUT(cube_path)
        frame = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        result = lut.apply(frame)

        # Everything should be white (255, 255, 255)
        assert np.all(result == 255)


def test_trilinear_interpolation():
    """Verify interpolation between LUT grid points is linear.

    With a 2x2x2 LUT where corner values are the coordinates themselves,
    a mid-gray input (0.5, 0.5, 0.5) should interpolate to (0.5, 0.5, 0.5).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cube_path = Path(tmpdir) / "interp.cube"
        values = _identity_values(2)
        _write_cube(cube_path, 2, values)

        lut = FrameLUT(cube_path)

        # Mid-gray pixel: RGB (128, 128, 128) -> BGR (128, 128, 128)
        frame = np.full((1, 1, 3), 128, dtype=np.uint8)
        result = lut.apply(frame)

        # Should be close to 128 (exact mid-point interpolation)
        np.testing.assert_allclose(result[0, 0].astype(float), [128, 128, 128], atol=1)


def test_lut_output_is_contiguous():
    """Output array should be C-contiguous for OpenCV compatibility."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cube_path = Path(tmpdir) / "identity.cube"
        values = _identity_values(2)
        _write_cube(cube_path, 2, values)

        lut = FrameLUT(cube_path)
        frame = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = lut.apply(frame)
        assert result.flags["C_CONTIGUOUS"]
