# audio-melody-extraction
## 1. create virtual environment and install dependencies
```bash
uv venv
uv pip install -r requirements.txt
```

## 2. download the datasets
```bash
chmod +x get_data.sh
./get_data.sh
```
datasets will be placed in 
```
datasets/
├── MIR-1K/
└── mirex05TrainFiles/
```

## 3. run preprocessing
```
uv run preprocessing.py
```
the resulting processed dataset will be saved under:
```
processed_data/
├── ah1_training_data.npz
└── config.pkl
```
### ```ah1_training_data.npz``` contains: 
| key                     | shape            | description                       |
| ----------------------- | ---------------- | --------------------------------- |
| `X`                     | (N, 1, 365, 517) | normalized log-CQT input features |
| `y_pitch_idx`           | (N, 517)         | AH1 pitch classes (0–48)          |
| `y_chroma`              | (N, 517)         | chroma labels (0–11)              |
| `y_octave`              | (N, 517)         | octave labels (0–3)               |
| `y_voicing`             | (N, 517)         | binary melody presence            |
| `y_pitch_hz`            | (N, 517)         | approximate Hz (from pitch class) |
| `song_ids`              | (N,)             | source clip + pitch-shift tag     |
| `norm_mean`, `norm_std` | broadcastable    | per-frequency normalization stats |


this pipeline follows the AH1 label specification:
- CQT: 365 bins, fmin=65 Hz, hop=256 samples
- Pitch classes: C2 (65.406 Hz) -> C6 (1046.5 Hz) = 48 bins
- Multi-task outputs: chroma (12), octave (4), voicing (2)
- Pitch-shift augmentation: +/-2 semitones
- this ensures compatibility with the original MIREX 2020 AH1 CRNN approach.

## NOTE: due to the ambiguity of the paper, this implementation uses only MIR-1K and MIREX05 for data preprocessing. 
