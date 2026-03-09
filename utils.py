"""Audio preprocessing: silence stripping and augmentation.

Writes results into a cache directory next to each source file so they
only run once.  The codebook builder then just encodes whatever is there.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def _augment(
    audio: np.ndarray,
    sr: int,
    pitch_shifts: list[int],
    volume_scales: list[float],
) -> list[tuple[str, np.ndarray]]:
    """Return (suffix, audio) pairs for the full n^2 augmentation grid.

    Every combination of pitch and volume is produced, plus pitch-only
    and volume-only variants.
    """
    variants: list[tuple[str, np.ndarray]] = []

    for steps in pitch_shifts:
        sign = "+" if steps >= 0 else ""
        pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
        # pitch-only at original volume
        variants.append((f"_p{sign}{steps}", pitched))
        # pitch x volume
        for vol in volume_scales:
            v_pct = int(vol * 100)
            variants.append((f"_p{sign}{steps}_v{v_pct}", pitched * vol))

    for vol in volume_scales:
        v_pct = int(vol * 100)
        # volume-only at original pitch
        variants.append((f"_v{v_pct}", audio * vol))

    return variants


def prepare_source_files(
    paths: list[str | Path],
    sr: int = 44_100,
    strip_silence: bool = True,
    augment_enabled: bool = False,
    pitch_shifts: list[int] | None = None,
    volume_scales: list[float] | None = None,
) -> list[Path]:
    """Preprocess source files and return paths to encode.

    For each source file, a cache directory is created alongside it using
    the source file's name (without extension).  If the directory already
    contains WAV files the step is skipped.

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

        if not strip_silence and not augment_enabled:
            all_paths.append(p)
            continue

        cache_dir = p.parent / p.stem
        existing = sorted(cache_dir.glob("*.wav")) if cache_dir.exists() else []

        if existing:
            print(f"  {p.name}: {len(existing)} prepared files in {cache_dir.name}/")
            all_paths.extend(existing)
            continue

        cache_dir.mkdir(exist_ok=True)
        y, _ = librosa.load(str(p), sr=sr, mono=True)

        if strip_silence:
            intervals = librosa.effects.split(y, top_db=40)
            y = np.concatenate([y[start:end] for start, end in intervals])

        y = librosa.util.normalize(y)

        # save the processed original
        out = cache_dir / f"{p.stem}.wav"
        sf.write(str(out), y, sr)
        all_paths.append(out)
        written = 1

        if augment_enabled:
            for suffix, variant in _augment(y, sr, pitch_shifts, volume_scales):
                aug_out = cache_dir / f"{p.stem}{suffix}.wav"
                sf.write(str(aug_out), variant, sr)
                all_paths.append(aug_out)
                written += 1

        print(f"  {p.name}: wrote {written} files to {cache_dir.name}/")

    return all_paths
