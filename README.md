# audio-melody-extraction
Re-implementation of a Multi-Task CRNN Architecture[^1] submission for the MIREX 2020 Audio Melody Extraction Challenge.

### 1. uv Virtual Environment and Dependency Installation
Install [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) before continuing.
```bash
uv venv
uv pip install -r requirements.txt
```

### 2. Dataset Download
```bash
chmod +x get_data.sh
./get_data.sh
```
Acquires MIR-1K[^2] and MIREX05[^3] datasets as in [^1], which are stored as follows:
```
datasets/
├── MIR-1K/
└── mirex05TrainFiles/
```

## 3. Preprocessing
```
uv run preprocessing.py
```
Joins MIR-1K and MIREX05 to creates training, validation and test splits.  
Augments training data and windows all data splits as in [^1].  
Saves data splits as the following .npz zipfiles:
```
processed_data/
├── ah1_train_set.npz
├── ah1_val_set.npz
└── ah1_test_set.npz
```
### Each ```ah1_*_set.npz``` contains: 
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


This pipeline follows only the AH1 label specification mentioned in [^1], not the incompatible HL1 specification.
- CQT: 365 bins, fmin=65 Hz, hop=256 samples
- Pitch classes: C2 (65.406 Hz) -> C6 (1046.5 Hz) = 48 bins
- Multi-task outputs: chroma (12), octave (4), voicing (2)
- Training data pitch-shift augmentation: ±2 semitones
- See [config.preproc_config](config.py) for AH1, train/val/test split configuration details

## 4. Training
```
uv run training.py
```
Trains a MelodyCRNN model in accordance with [config.train_config](config.py).  
The iteration of the model with the lowest validation loss after an epoch is saved to as:
```
models/
└── melody_crnn_[TIMESTAMP].pt
```
where `[TIMESTAMP]` takes the format `YYYY-MM-DD_HH-MM-SS` and stores when the training script was called, not when the model iteration was validated.

## 5. Evaluation
```
wip
```
Because a testing set is unavailable, we produce a test split of our unified MIR-1K-MIREX05 database for evaluation.

## 6. Inference
```
wip
```

[^1]: A. Huang and H. Liu, *MIREX2020: AUDIO MELODY EXTRACTION USING NEW MULTI‐TASK CONVOLUTIONAL RECURRENT NEURAL NETWORK*. Accessed: 2025. [Online]. Available: https://www.music-ir.org/mirex/abstracts/2020/AH1.pdf
[^2]: R. Jang, “MIR Corpora,” Multimedia Information Retrieval LAB, http://mirlab.org/dataset/public/ (accessed Nov. 23, 2025).
[^3]: G. Poliner, “Polyphonic Melody Extraction,” LabROSA, https://labrosa.ee.columbia.edu/projects/melody/ (accessed Nov. 23, 2025).
