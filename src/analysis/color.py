"""3D LUT parsing and per-frame color space conversion.

Supports .cube files directly, or .zip/.lut archives containing .cube files.
Uses vectorized numpy trilinear interpolation — no extra dependencies needed.
"""

from __future__ import annotations

import logging
import zipfile
from io import StringIO
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def parse_cube_file(path: Path) -> tuple[np.ndarray, int]:
    """Parse a .cube 3D LUT file (or extract one from a ZIP archive).

    Args:
        path: Path to a .cube file, or a .zip/.lut archive containing one.

    Returns:
        Tuple of (lut_array with shape (N, N, N, 3), size N).

    Raises:
        ValueError: If the file cannot be parsed or no .cube found in archive.
    """
    path = Path(path)

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            cube_names = [n for n in zf.namelist() if n.lower().endswith(".cube")]
            if not cube_names:
                raise ValueError(f"No .cube file found inside archive: {path}")
            cube_name = cube_names[0]
            logger.info("Extracting '%s' from archive '%s'", cube_name, path.name)
            cube_text = zf.read(cube_name).decode("utf-8")
    else:
        cube_text = path.read_text()

    return _parse_cube_text(cube_text)


def _parse_cube_text(text: str) -> tuple[np.ndarray, int]:
    """Parse .cube text content into a 3D LUT array."""
    size = 0
    values: list[list[float]] = []

    for line in StringIO(text):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.upper().startswith("LUT_3D_SIZE"):
            size = int(line.split()[-1])
            continue

        # Skip other header keywords
        if line.upper().startswith(("TITLE", "DOMAIN_MIN", "DOMAIN_MAX", "LUT_1D_SIZE")):
            continue

        # Data line: three floats
        parts = line.split()
        if len(parts) >= 3:
            try:
                values.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    if size == 0:
        raise ValueError("LUT_3D_SIZE not found in .cube file")

    expected = size ** 3
    if len(values) != expected:
        raise ValueError(
            f"Expected {expected} LUT entries for size {size}, got {len(values)}"
        )

    lut = np.array(values, dtype=np.float32).reshape(size, size, size, 3)
    # .cube format orders data with R varying fastest (innermost) and B slowest
    # (outermost). After reshape, axes are (B, G, R, 3). Transpose so that
    # lut[r, g, b] gives the output for input (R, G, B).
    lut = np.ascontiguousarray(lut.transpose(2, 1, 0, 3))
    logger.info("Parsed 3D LUT: %dx%dx%d", size, size, size)
    return lut, size


class FrameLUT:
    """Applies a 3D LUT to video frames via trilinear interpolation.

    When initialized with ``None``, all operations are no-ops with zero overhead.
    """

    def __init__(self, lut_path: Path | None) -> None:
        if lut_path is None:
            self._table_flat = None
            return

        lut, size = parse_cube_file(lut_path)

        # Pre-compute a full 256x256x256 lookup table via trilinear interpolation.
        # ~48MB one-time cost, but makes apply() a single flat array index per frame.
        logger.info("Building 256^3 lookup table from %dx%dx%d LUT...", size, size, size)
        table = self._build_full_table(lut, size)
        # Flatten to (256^3, 3) for fast single-index lookup
        self._table_flat = table.reshape(-1, 3)
        logger.info("LUT loaded from %s (size %d)", lut_path, size)

    @staticmethod
    def _build_full_table(lut: np.ndarray, size: int) -> np.ndarray:
        """Pre-interpolate the 3D LUT into a full 256x256x256 uint8 table."""
        max_idx = size - 1
        lut_flat = lut.reshape(-1, 3)  # (N^3, 3)
        stride_r = size * size
        stride_g = size

        # Generate all 256 values per channel, scaled to LUT coordinates
        vals = np.arange(256, dtype=np.float32) * (max_idx / 255.0)
        lo = np.floor(vals).astype(np.intp)
        np.clip(lo, 0, max_idx - 1, out=lo)
        frac = vals - lo

        # Build the table one R-slice at a time to limit peak memory
        table = np.empty((256, 256, 256, 3), dtype=np.uint8)

        for ri in range(256):
            r0 = lo[ri]
            fr = frac[ri]

            # For this R value, compute all G x B combinations
            g0 = lo[np.newaxis, :]       # (1, 256)
            b0 = lo[np.newaxis, :]       # (1, 256)
            fg = frac[np.newaxis, :]     # (1, 256)
            fb = frac[np.newaxis, :]     # (1, 256)

            # Base indices for all (G, B) pairs: shape (256, 256)
            base = (r0 * stride_r
                    + lo[:, np.newaxis] * stride_g   # G dim -> rows
                    + lo[np.newaxis, :])              # B dim -> cols

            # 8 corner lookups: each (256, 256, 3)
            c000 = lut_flat[base.ravel()].reshape(256, 256, 3).astype(np.float32)
            c001 = lut_flat[(base + 1).ravel()].reshape(256, 256, 3).astype(np.float32)
            c010 = lut_flat[(base + stride_g).ravel()].reshape(256, 256, 3).astype(np.float32)
            c011 = lut_flat[(base + stride_g + 1).ravel()].reshape(256, 256, 3).astype(np.float32)
            c100 = lut_flat[(base + stride_r).ravel()].reshape(256, 256, 3).astype(np.float32)
            c101 = lut_flat[(base + stride_r + 1).ravel()].reshape(256, 256, 3).astype(np.float32)
            c110 = lut_flat[(base + stride_r + stride_g).ravel()].reshape(256, 256, 3).astype(np.float32)
            c111 = lut_flat[(base + stride_r + stride_g + 1).ravel()].reshape(256, 256, 3).astype(np.float32)

            # Trilinear interpolation
            fb3 = frac[np.newaxis, :, np.newaxis]  # (1, 256, 1) for B axis
            fg3 = frac[:, np.newaxis, np.newaxis]   # (256, 1, 1) for G axis

            c00 = c000 + (c001 - c000) * fb3
            c01 = c010 + (c011 - c010) * fb3
            c10 = c100 + (c101 - c100) * fb3
            c11 = c110 + (c111 - c110) * fb3

            c0 = c00 + (c01 - c00) * fg3
            c1 = c10 + (c11 - c10) * fg3

            result = c0 + (c1 - c0) * fr

            np.clip(result * 255.0, 0, 255, out=result)
            table[ri] = result.astype(np.uint8)

        return table

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply the 3D LUT to a BGR uint8 frame via pre-computed table lookup.

        Returns the input unchanged (same object) when no LUT is loaded.
        """
        if self._table_flat is None:
            return frame

        h, w = frame.shape[:2]
        flat = frame.reshape(-1, 3)

        # Compute flat index: R * 65536 + G * 256 + B (BGR input → RGB index)
        idx = (flat[:, 2].astype(np.int32) << 16) | (flat[:, 1].astype(np.int32) << 8) | flat[:, 0]
        result = self._table_flat[idx]

        # Result is RGB, convert back to BGR
        return np.ascontiguousarray(result.reshape(h, w, 3)[:, :, ::-1])
