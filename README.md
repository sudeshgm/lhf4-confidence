# LHF-4 confidence smoke

Frozen VLM smoke for the driving-confidence kill test. Not Experiment −1. nuScenes was not downloaded.

- 20 Unsplash frames, login-free. Gold assigned only where the clean frame has an obvious answer.
- Fog: `scripts/fog.py`. Severity 0 is the original. Severities 1–4 blend toward grey `(128,128,128)` with strengths `0.25, 0.45, 0.65, 0.85`.
- Pre-rendered severity 0 and 4 images are in `data/images/`.
- GPU steps: [GPU_STEPS.md](GPU_STEPS.md).

Images are Unsplash photos under the [Unsplash License](https://unsplash.com/license). Do not commit model weights.
