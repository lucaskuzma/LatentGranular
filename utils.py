"""Audio preprocessing: chunking at zero crossings and augmentation.

Both steps write WAV files into a ``<stem>_chunks/`` folder next to each
source file so they only run once.  The codebook builder then just encodes
whatever is in those folders.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def _find_zero_crossing_before(audio: np.ndarray, pos: int) -> int:
    """Walk backwards from *pos* to find the most recent zero crossing.
    Returns *pos* unchanged if none found (hard cut)."""
    for i in range(pos, max(pos - 4096, 0), -1):
        if i > 0 and np.sign(audio[i]) != np.sign(audio[i - 1]):
            return i
    return pos


def _chunk_audio(
    audio: np.ndarray,
    sr: int,
    max_length_ms: float,
    max_count: int,
) -> list[np.ndarray]:
    """Split *audio* into chunks of at most *max_length_ms*, trimmed at zero
    crossings.  Returns up to *max_count* chunks."""
    max_samples = int(sr * max_length_ms / 1000.0)
    chunks: list[np.ndarray] = []
    pos = 0

    while pos < len(audio) and len(chunks) < max_count:
        end = min(pos + max_samples, len(audio))
        if end < len(audio):
            end = _find_zero_crossing_before(audio, end)
        chunk = audio[pos:end]
        if len(chunk) > 0:
            chunks.append(chunk)
        pos = end

    return chunks


def _augment_chunk(
    chunk: np.ndarray,
    sr: int,
    pitch_shifts: list[int],
    volume_scales: list[float],
) -> list[tuple[str, np.ndarray]]:
    """Return a list of (suffix, audio) pairs for augmented variants."""
    variants: list[tuple[str, np.ndarray]] = []
    for vol in volume_scales:
        variants.append((f"_vol{vol:.1f}", chunk * vol))
    for steps in pitch_shifts:
        sign = "p" if steps >= 0 else "m"
        variants.append(
            (f"_pitch{sign}{abs(steps)}", librosa.effects.pitch_shift(chunk, sr=sr, n_steps=steps))
        )
    return variants


def prepare_source_files(
    paths: list[str | Path],
    sr: int = 44_100,
    chunk_enabled: bool = False,
    max_length_ms: float = 4000.0,
    max_count: int = 64,
    augment_enabled: bool = False,
    pitch_shifts: list[int] | None = None,
    volume_scales: list[float] | None = None,
) -> list[Path]:
    """Preprocess source files (chunk + augment) and return paths to encode.

    For each source file, a ``<stem>_chunks/`` folder is created alongside it.
    If the folder already contains WAV files the step is skipped.

    Returns a flat list of all WAV paths the codebook builder should encode.
    """
    if pitch_shifts is None:
        pitch_shifts = [-5, -2, 2, 5]
    if volume_scales is None:
        volume_scales = [0.3, 0.7]

    all_paths: list[Path] = []

    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  WARNING: {p} not found, skipping")
            continue

        if not chunk_enabled and not augment_enabled:
            all_paths.append(p)
            continue

        chunk_dir = p.parent / f"{p.stem}_chunks"
        existing = sorted(chunk_dir.glob("*.wav")) if chunk_dir.exists() else []

        if existing:
            print(f"  {p.name}: {len(existing)} prepared files already in {chunk_dir.name}/")
            all_paths.extend(existing)
            continue

        chunk_dir.mkdir(exist_ok=True)
        y, _ = librosa.load(str(p), sr=sr, mono=True)
        y, _ = librosa.effects.trim(y, top_db=40)
        y = librosa.util.normalize(y)

        if chunk_enabled:
            chunks = _chunk_audio(y, sr, max_length_ms, max_count)
        else:
            chunks = [y]

        written = 0
        for ci, chunk in enumerate(chunks):
            # original chunk
            name = f"{p.stem}_{ci:03d}.wav"
            out = chunk_dir / name
            sf.write(str(out), chunk, sr)
            all_paths.append(out)
            written += 1

            # augmented variants
            if augment_enabled:
                for suffix, variant in _augment_chunk(chunk, sr, pitch_shifts, volume_scales):
                    aug_name = f"{p.stem}_{ci:03d}{suffix}.wav"
                    aug_out = chunk_dir / aug_name
                    sf.write(str(aug_out), variant, sr)
                    all_paths.append(aug_out)
                    written += 1

        print(f"  {p.name}: wrote {written} files to {chunk_dir.name}/")

    return all_paths
