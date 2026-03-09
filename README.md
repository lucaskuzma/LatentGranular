# Latent Granular Resynthesis

Implementation of **"Latent Granular Resynthesis using Neural Audio Codecs"**
by Nao Tokui and Tom Baker (ISMIR 2025 Late-Breaking Demo).

- Paper: <https://arxiv.org/abs/2507.19202>
- Original demo: <https://huggingface.co/spaces/naotokui/latentgranular>
- Original repo: <https://github.com/naotokui/latentgranular>

## What this does

Encodes a **source** audio corpus and a **target** audio signal into the latent
space of a neural audio codec, then replaces each latent "grain" of the target
with the closest match from the source codebook. The decoder reconstructs
continuous audio that preserves the target's temporal structure while adopting
the source's timbral character — no model training required.

## Deviations from the original

| Area | Original (HF demo) | This implementation |
|---|---|---|
| **Codec support** | Music2Latent only | Codec-agnostic abstraction — ships with Music2Latent *and* DAC (MIT licensed) |
| **Matching** | Per-grain loop with scipy `cdist` on CPU | Vectorized cosine similarity via PyTorch on GPU |
| **Output normalization** | Hardcoded `* 31000` int16 | Proper peak normalization, float32 WAV output |
| **Stereo** | Hardcoded duplicate-to-stereo | Mono throughout (clean foundation for future stereo) |
| **Stride** | Single stride parameter, used inconsistently | Separate concerns: dense codebook stride for coverage, target always non-overlapping (stride=grain_size) — decoder handles interpolation |
| **Persistence** | None | Codebooks can be saved/loaded with `torch.save` |
| **Visualization** | None | Distance heatmaps, grain selection plots, source breakdown, spectrograms |
| **Preprocessing** | None (augmentation in-memory, re-computed every run) | Silence stripping + full n^2 pitch x volume augmentation grid, cached to disk in `<stem>/` folders |
| **DAC comparison** | N/A | Automatic grain-size scaling to match time windows across codecs |

## Setup

Requires Python 3.11+ (uses `tomllib`).

```bash
# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

Model weights for both Music2Latent and DAC download automatically on first
use (from HuggingFace Hub).

## Usage

Copy the example config and edit it with your audio paths and parameters:

```bash
cp config.example.toml config.toml
```

`config.toml` is gitignored so your local paths and experiments don't dirty
the repo. Then open `latent_granular.py` in Cursor / VS Code, drop your audio
files in `audio/`, and run the `# %%` cells top-to-bottom.

## License notes

- **Music2Latent** (Sony CSL Paris): CC BY-NC 4.0 — non-commercial use only.
- **DAC** (Descript): MIT — fully permissive.
- This implementation code: choose whichever codec fits your use case.
