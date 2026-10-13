# LHF-4 GPU smoke

No training. First 8 manifest rows (`i01`–`i08`). Severities 0 and 4 only.

Fog is already rendered. Gold is the clean-frame label. Do not relabel the fogged image.

## On the GPU machine

```bash
git clone https://github.com/sudeshgm/lhf4-confidence.git
cd lhf4-confidence
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If `cu121` does not match the machine, install the PyTorch build that matches `nvidia-smi`.

```bash
python scripts/smoke_vlm.py
```

Copy the block that starts with `MODEL:` through the `ROWS` lines and paste it back. Do not edit the numbers.

Private repo. If clone asks for a login, use a GitHub personal access token as the password, or `gh auth login` first.
