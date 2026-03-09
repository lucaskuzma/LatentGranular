# %% Setup — Latent Granular Resynthesis
# Implementation of Tokui & Baker (2025) — "Latent Granular Resynthesis using
# Neural Audio Codecs". Operates granular synthesis at the latent-vector level
# of a pretrained neural audio codec, enabling training-free timbre transfer.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from IPython.display import Audio, display

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# %% Codec Abstraction Layer
# A thin wrapper so we can swap Music2Latent, DAC, or any future codec
# without touching the granular logic.


class Codec(ABC):
    """Minimal interface every codec wrapper must satisfy."""

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def latent_rate(self) -> float:
        """Approximate number of latent vectors per second of audio."""
        ...

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Channel dimension of each latent vector."""
        ...

    @abstractmethod
    def encode(self, audio: np.ndarray) -> torch.Tensor:
        """Encode a mono waveform (float32, self.sample_rate) into latents.

        Returns shape ``(1, latent_dim, seq_len)``.
        """
        ...

    @abstractmethod
    def decode(self, latents: torch.Tensor) -> np.ndarray:
        """Decode latents back to a mono waveform ``(samples,)``."""
        ...

    # ── helpers ──────────────────────────────────────────────────────────

    def ms_to_vectors(self, ms: float) -> int:
        return max(1, round(ms / 1000.0 * self.latent_rate))

    def vectors_to_ms(self, n: int) -> float:
        return n / self.latent_rate * 1000.0


# ── Music2Latent ─────────────────────────────────────────────────────────


class Music2LatentCodec(Codec):
    """Wraps ``music2latent.EncoderDecoder``.

    Continuous consistency-autoencoder, ~10 Hz, 64-dim latents at 44.1 kHz.
    License: CC BY-NC 4.0.
    """

    def __init__(self, device: Optional[torch.device] = None):
        from music2latent import EncoderDecoder

        self._device = device or DEVICE
        self._enc_dec = EncoderDecoder(device=self._device)

    @property
    def sample_rate(self) -> int:
        return 44_100

    @property
    def latent_rate(self) -> float:
        return 10.75  # 44100 / 4096

    @property
    def latent_dim(self) -> int:
        return 64

    def encode(self, audio: np.ndarray) -> torch.Tensor:
        audio = librosa.util.normalize(audio)
        latent = self._enc_dec.encode(audio)  # (ch, 64, T)
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
        return latent[:1]  # mono — first channel

    def decode(self, latents: torch.Tensor) -> np.ndarray:
        wav = self._enc_dec.decode(latents)
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        return wav.squeeze()


# ── DAC (Descript Audio Codec) ───────────────────────────────────────────


class DACCodec(Codec):
    """Wraps ``dac.DAC``, bypassing RVQ to use continuous encoder output.

    ~86 Hz, 1024-dim latents at 44.1 kHz.  License: MIT.
    """

    def __init__(
        self, model_type: str = "44khz", device: Optional[torch.device] = None
    ):
        import dac

        self._device = device or DEVICE
        model_path = dac.utils.download(model_type=model_type)
        self._model = dac.DAC.load(model_path).to(self._device).eval()

    @property
    def sample_rate(self) -> int:
        return self._model.sample_rate

    @property
    def latent_rate(self) -> float:
        return self.sample_rate / self._model.hop_length  # 44100/512 ≈ 86.13

    @property
    def latent_dim(self) -> int:
        return self._model.latent_dim  # 1024

    @torch.no_grad()
    def encode(self, audio: np.ndarray) -> torch.Tensor:
        audio = librosa.util.normalize(audio)
        t = torch.from_numpy(audio).float().to(self._device)
        if t.dim() == 1:
            t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        t = self._model.preprocess(t, self.sample_rate)
        z = self._model.encoder(t)  # (1, 1024, T') — continuous, pre-RVQ
        return z

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> np.ndarray:
        latents = latents.to(self._device)
        wav = self._model.decoder(latents)  # (1, 1, T)
        return wav.squeeze().cpu().numpy()


# %% Granular Codebook
# Encodes a source audio corpus into latent vectors, segments them into
# overlapping grains, and stores them for efficient nearest-neighbour lookup.


@dataclass
class Augmentation:
    """Simple pre-encoding augmentation config."""

    pitch_shifts: list[int] = field(default_factory=lambda: [-5, -2, 2, 5])
    volume_scales: list[float] = field(default_factory=lambda: [0.3, 0.7])
    enabled: bool = False


class GranularCodebook:
    """A codebook of latent grains extracted from a source audio corpus."""

    def __init__(
        self,
        codec: Codec,
        grain_size: int = 2,
        stride: int = 1,
        augmentation: Optional[Augmentation] = None,
    ):
        self.codec = codec
        self.grain_size = grain_size
        self.stride = stride
        self.aug = augmentation or Augmentation()

        self.grains: Optional[torch.Tensor] = None  # (N, dim, grain_size)
        self._source_latents: Optional[torch.Tensor] = None

    # ── public API ───────────────────────────────────────────────────────

    def build(self, paths: Sequence[str | Path], max_duration: Optional[float] = None):
        """Encode source files and segment into grains.

        Parameters
        ----------
        paths : sequence of file paths
        max_duration : if set, truncate each file to this many seconds
        """
        all_latents: list[torch.Tensor] = []
        sr = self.codec.sample_rate

        for p in paths:
            print(f"  Encoding {p} ...")
            y, _ = librosa.load(str(p), sr=sr, mono=True)
            if max_duration is not None:
                y = y[: int(sr * max_duration)]
            variants = self._augmented_variants(y, sr)
            for v in variants:
                lat = self.codec.encode(v)  # (1, dim, T)
                all_latents.append(lat.cpu())

        full = torch.cat(all_latents, dim=-1)  # (1, dim, total_T)
        self._source_latents = full
        self.grains = self._segment(full)
        print(
            f"Codebook: {self.grains.shape[0]} grains  "
            f"(grain_size={self.grain_size}, stride={self.stride}, "
            f"latent_dim={self.codec.latent_dim})"
        )

    def save(self, path: str | Path):
        torch.save(
            {
                "grains": self.grains,
                "grain_size": self.grain_size,
                "stride": self.stride,
                "codec_class": type(self.codec).__name__,
                "latent_dim": self.codec.latent_dim,
            },
            path,
        )

    def load(self, path: str | Path):
        data = torch.load(path, weights_only=False)
        self.grains = data["grains"]
        self.grain_size = data["grain_size"]
        self.stride = data["stride"]
        print(f"Loaded codebook: {self.grains.shape[0]} grains")

    # ── internals ────────────────────────────────────────────────────────

    def _segment(self, latents: torch.Tensor) -> torch.Tensor:
        """Unfold latents into overlapping grains.

        Input:  (1, dim, T)
        Output: (N, dim, grain_size)
        """
        _, dim, T = latents.shape
        grains = []
        for i in range(0, T - self.grain_size + 1, self.stride):
            grains.append(latents[0, :, i : i + self.grain_size])
        return torch.stack(grains)  # (N, dim, grain_size)

    def _augmented_variants(self, audio: np.ndarray, sr: int) -> list[np.ndarray]:
        variants = [audio]
        if not self.aug.enabled:
            return variants
        for vol in self.aug.volume_scales:
            variants.append(audio * vol)
        for steps in self.aug.pitch_shifts:
            variants.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps))
        return variants


# %% Target Matcher
# Encodes the target audio, segments it the same way, and for each target
# grain finds the best match from the codebook via cosine similarity with
# temperature-controlled softmax sampling.


@dataclass
class MatchResult:
    """Everything returned by the matching step."""

    hybrid_latents: torch.Tensor  # (1, dim, T') — the reassembled sequence
    target_latents: torch.Tensor  # (1, dim, T_target)
    distances: np.ndarray  # (n_target_grains, n_codebook_grains)
    selected_indices: list[int]  # which codebook grain was chosen per step


def match_target(
    target_path: str | Path,
    codebook: GranularCodebook,
    temperature: float = 0.01,
    threshold: float = 1.0,
    max_duration: Optional[float] = None,
) -> MatchResult:
    """Match a target audio file against a codebook.

    Parameters
    ----------
    target_path : path to the target audio file
    codebook : a built GranularCodebook
    temperature : softmax temperature — lower = more faithful match
    threshold : if best cosine distance exceeds this, keep the target grain
    max_duration : optionally truncate target to this many seconds
    """
    codec = codebook.codec
    sr = codec.sample_rate

    y, _ = librosa.load(str(target_path), sr=sr, mono=True)
    if max_duration is not None:
        y = y[: int(sr * max_duration)]

    target_latents = codec.encode(y)  # (1, dim, T)
    target_grains = codebook._segment(target_latents)  # (M, dim, gs)

    db = codebook.grains.to(DEVICE)  # (N, dim, gs)
    N = db.shape[0]

    db_flat = db.reshape(N, -1)  # (N, dim*gs)
    db_flat_norm = F.normalize(db_flat, dim=-1)

    selected_indices: list[int] = []
    all_distances: list[np.ndarray] = []
    result_grains: list[torch.Tensor] = []

    gs = codebook.grain_size
    stride = codebook.stride

    for i in range(target_grains.shape[0]):
        tg = target_grains[i : i + 1].to(DEVICE)  # (1, dim, gs)
        tg_flat = tg.reshape(1, -1)
        tg_flat_norm = F.normalize(tg_flat, dim=-1)

        # cosine distance = 1 - cosine_similarity
        cos_sim = (tg_flat_norm @ db_flat_norm.T).squeeze(0)  # (N,)
        cos_dist = 1.0 - cos_sim

        dists_np = cos_dist.cpu().numpy()
        all_distances.append(dists_np)

        # softmax over negative distances / temperature
        logits = -cos_dist / (temperature + 1e-8)
        logits = logits - logits.max()  # numerical stability
        probs = torch.softmax(logits, dim=0).cpu().numpy()

        idx = np.random.choice(N, p=probs)

        if dists_np.min() > threshold:
            result_grains.append(tg.squeeze(0).cpu())
        else:
            result_grains.append(db[idx].cpu())
        selected_indices.append(idx)

    # Reassemble: place grains at their stride positions, averaging overlaps
    total_len = (len(result_grains) - 1) * stride + gs
    _, dim, _ = target_latents.shape
    hybrid = torch.zeros(dim, total_len)
    counts = torch.zeros(1, total_len)

    for i, grain in enumerate(result_grains):
        start = i * stride
        hybrid[:, start : start + gs] += grain
        counts[:, start : start + gs] += 1.0

    counts = counts.clamp(min=1.0)
    hybrid = hybrid / counts
    hybrid = hybrid.unsqueeze(0)  # (1, dim, T')

    # Trim or pad to match original target length
    T_target = target_latents.shape[-1]
    if hybrid.shape[-1] > T_target:
        hybrid = hybrid[:, :, :T_target]
    elif hybrid.shape[-1] < T_target:
        pad = T_target - hybrid.shape[-1]
        hybrid = F.pad(hybrid, (0, pad))

    return MatchResult(
        hybrid_latents=hybrid,
        target_latents=target_latents.cpu(),
        distances=np.stack(all_distances),
        selected_indices=selected_indices,
    )


# %% Reconstruction and Output
# Decode the hybrid latent sequence, normalize, and play back.


def reconstruct(
    result: MatchResult,
    codec: Codec,
    output_path: Optional[str | Path] = None,
) -> np.ndarray:
    """Decode hybrid latents and optionally write to disk."""
    wav = codec.decode(result.hybrid_latents)
    wav = wav / (np.abs(wav).max() + 1e-8)  # peak-normalize

    if output_path is not None:
        sf.write(str(output_path), wav, codec.sample_rate)
        print(f"Wrote {output_path}")

    return wav


def play_comparison(
    source_path: str | Path,
    target_path: str | Path,
    output_wav: np.ndarray,
    codec: Codec,
):
    """Display side-by-side audio players in a notebook."""
    sr = codec.sample_rate
    src, _ = librosa.load(str(source_path), sr=sr, mono=True)
    tgt, _ = librosa.load(str(target_path), sr=sr, mono=True)

    print("── Source ──")
    display(Audio(src, rate=sr))
    print("── Target ──")
    display(Audio(tgt, rate=sr))
    print("── Output (latent granular resynthesis) ──")
    display(Audio(output_wav, rate=sr))


# %% Visualization
# Distance heatmaps, grain selection patterns, and spectrogram comparisons.

import matplotlib.pyplot as plt


def plot_distance_heatmap(result: MatchResult, max_codebook_grains: int = 500):
    """Heatmap of cosine distances: target grains (y) vs codebook grains (x)."""
    dists = result.distances[:, :max_codebook_grains]
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(dists, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Codebook grain index")
    ax.set_ylabel("Target grain index")
    ax.set_title("Cosine distance: target ↔ codebook")
    fig.colorbar(im, ax=ax, label="cosine distance")
    plt.tight_layout()
    plt.show()


def plot_grain_selection(result: MatchResult):
    """Which codebook grain was selected at each target step."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(
        range(len(result.selected_indices)),
        result.selected_indices,
        s=4,
        alpha=0.6,
    )
    ax.set_xlabel("Target grain step")
    ax.set_ylabel("Selected codebook grain")
    ax.set_title("Grain selection over time")
    plt.tight_layout()
    plt.show()


def plot_spectrograms(
    source_path: str | Path,
    target_path: str | Path,
    output_wav: np.ndarray,
    sr: int,
):
    """Side-by-side mel spectrograms of source, target, and output."""
    src, _ = librosa.load(str(source_path), sr=sr, mono=True)
    tgt, _ = librosa.load(str(target_path), sr=sr, mono=True)

    signals = {"Source": src, "Target": tgt, "Output": output_wav}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    for ax, (label, wav) in zip(axes, signals.items()):
        S = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title(label)

    plt.tight_layout()
    plt.show()


def plot_min_distances(result: MatchResult):
    """Per-grain minimum cosine distance — shows how well each target grain
    was matched by the codebook."""
    mins = result.distances.min(axis=1)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(mins, linewidth=0.8)
    ax.set_xlabel("Target grain step")
    ax.set_ylabel("Min cosine distance")
    ax.set_title("Best codebook match quality per target grain")
    plt.tight_layout()
    plt.show()


# %% Configuration — reads from config.toml (gitignored)
# Copy config.example.toml to get started:
#     cp config.example.toml config.toml

import tomllib

_cfg_path = Path("config.toml")
if not _cfg_path.exists():
    raise FileNotFoundError(
        "config.toml not found — copy the example:\n"
        "  cp config.example.toml config.toml"
    )

with open(_cfg_path, "rb") as f:
    CFG = tomllib.load(f)

SOURCE_FILES: list[str] = CFG["source"]["files"]
TARGET_FILE: str = CFG["target"]["file"]
GRAIN_SIZE: int = CFG["grains"]["size"]
STRIDE: int = CFG["grains"]["stride"]
TEMPERATURE: float = CFG["matching"]["temperature"]
THRESHOLD: float = CFG["matching"]["threshold"]
AUGMENT: bool = CFG.get("augmentation", {}).get("enabled", False)

print(f"Source: {SOURCE_FILES}")
print(f"Target: {TARGET_FILE}")
print(f"Grains: size={GRAIN_SIZE}, stride={STRIDE}")
print(f"Matching: temperature={TEMPERATURE}, threshold={THRESHOLD}")
print(f"Augment: {AUGMENT}")

# %% Run: Music2Latent — Build codebook

if SOURCE_FILES and TARGET_FILE:
    codec_m2l = Music2LatentCodec(device=DEVICE)
    _aug_cfg = CFG.get("augmentation", {})
    aug = Augmentation(
        enabled=AUGMENT,
        **{k: _aug_cfg[k] for k in ("pitch_shifts", "volume_scales") if k in _aug_cfg},
    )
    codebook_m2l = GranularCodebook(
        codec_m2l, grain_size=GRAIN_SIZE, stride=STRIDE, augmentation=aug
    )
    codebook_m2l.build(SOURCE_FILES)
else:
    print("Set SOURCE_FILES and TARGET_FILE in config.toml, then re-run.")

# %% Match and reconstruct

if SOURCE_FILES and TARGET_FILE:
    result_m2l = match_target(
        TARGET_FILE,
        codebook_m2l,
        temperature=TEMPERATURE,
        threshold=THRESHOLD,
    )
    output_m2l = reconstruct(result_m2l, codec_m2l, output_path="audio/output_m2l.wav")
    play_comparison(SOURCE_FILES[0], TARGET_FILE, output_m2l, codec_m2l)
else:
    print("Set SOURCE_FILES and TARGET_FILE in config.toml, then re-run.")

# %% Visualize

if SOURCE_FILES and TARGET_FILE:
    plot_distance_heatmap(result_m2l)
    plot_grain_selection(result_m2l)
    plot_min_distances(result_m2l)
    plot_spectrograms(SOURCE_FILES[0], TARGET_FILE, output_m2l, codec_m2l.sample_rate)

# %% Run: DAC (MIT licensed) — same source/target for comparison

if SOURCE_FILES and TARGET_FILE:
    codec_dac = DACCodec(device=DEVICE)

    # DAC latents are ~86 Hz vs Music2Latent's ~10 Hz, so we scale grain_size
    # to cover roughly the same time window.
    dac_grain_size = codec_dac.ms_to_vectors(codec_m2l.vectors_to_ms(GRAIN_SIZE))
    dac_stride = max(1, dac_grain_size // 2)
    print(
        f"DAC grain_size={dac_grain_size} vectors "
        f"(≈{codec_dac.vectors_to_ms(dac_grain_size):.0f} ms), "
        f"stride={dac_stride}"
    )

    codebook_dac = GranularCodebook(
        codec_dac, grain_size=dac_grain_size, stride=dac_stride
    )
    codebook_dac.build(SOURCE_FILES)

    result_dac = match_target(
        TARGET_FILE,
        codebook_dac,
        temperature=TEMPERATURE,
        threshold=THRESHOLD,
    )
    output_dac = reconstruct(result_dac, codec_dac, output_path="audio/output_dac.wav")

    print("\n── DAC output ──")
    display(Audio(output_dac, rate=codec_dac.sample_rate))
    print("\n── Music2Latent output (for comparison) ──")
    display(Audio(output_m2l, rate=codec_m2l.sample_rate))

    plot_spectrograms(SOURCE_FILES[0], TARGET_FILE, output_dac, codec_dac.sample_rate)
else:
    print("Set SOURCE_FILES and TARGET_FILE in config.toml, then re-run.")
