"""
AH1 Melody Extraction Preprocessing Pipeline

Processes MIR-1K and MIREX05 datasets for training a multi-task CRNN,
following the AH1 version description from MIREX 2020 Audio Melody Extraction.
"""

import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import warnings
import random

from config import preproc_config as config

warnings.filterwarnings("ignore")

# AH1 pitch utilities

def hz_to_ah1_index(f0_hz: float, c2_freq: float, num_pitches: int) -> int:
    if f0_hz <= 0:
        return 0
    semitone = 12.0 * np.log2(f0_hz / c2_freq)
    idx = int(round(semitone)) + 1  
    if 1 <= idx <= num_pitches:
        return idx
    return 0


def hz_array_to_ah1_indices(f0_hz: np.ndarray) -> np.ndarray:
    func = np.vectorize(lambda x: hz_to_ah1_index(x, config['c2_freq'], config['num_pitches']))
    return func(f0_hz).astype(np.int32)


def indices_to_chroma_octave(pitch_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = pitch_idx.astype(np.int32) - 1
    chroma = np.zeros_like(p, dtype=np.int32)
    octave = np.zeros_like(p, dtype=np.int32)

    mask = pitch_idx > 0
    chroma[mask] = p[mask] % 12
    octave[mask] = p[mask] // 12

    chroma[~mask] = 0
    octave[~mask] = 0
    return chroma, octave


def indices_to_hz(pitch_idx: np.ndarray) -> np.ndarray:
    f0_hz = np.zeros_like(pitch_idx, dtype=np.float32)
    mask = pitch_idx > 0
    semitone = (pitch_idx[mask].astype(np.float32) - 1.0)
    f0_hz[mask] = config['c2_freq'] * (2.0 ** (semitone / 12.0))
    return f0_hz


def shift_pitch_classes(pitch_idx: np.ndarray, k: int, num_pitches: int) -> np.ndarray:
    if k == 0:
        return pitch_idx.copy()

    shifted = pitch_idx.astype(np.int32).copy()
    mask = shifted > 0
    shifted[mask] = shifted[mask] + k
    shifted[(shifted < 1) | (shifted > num_pitches)] = 0
    return shifted

# CQT computation

def compute_log_cqt(audio: np.ndarray) -> np.ndarray:
    cqt = librosa.cqt(
        audio,
        sr=config['sample_rate'],
        hop_length=config['hop_size'],
        fmin=config['cqt_fmin'],
        n_bins=config['cqt_bins'],
        bins_per_octave=config['bins_per_octave'],
        window=config['window'],
    )
    mag = np.abs(cqt)
    log_cqt = np.log1p(mag)  
    return log_cqt.astype(np.float32)

# annotation loaders

def load_pv_annotation(pv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(pv_path, "r") as f:
        lines = f.readlines()

    f0_list = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            val = float(line)
        except ValueError:
            val = 0.0
        if val <= 0:
            val = 0.0
        f0_list.append(val)

    f0_10ms = np.array(f0_list, dtype=np.float32)
    times_10ms = np.arange(len(f0_10ms), dtype=np.float32) * 0.01  # every 10 ms
    return times_10ms, f0_10ms


def load_ref_annotation(ref_path: Path) -> tuple[np.ndarray, np.ndarray]:
    times = []
    f0s = []
    with open(ref_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
            f0 = float(parts[1])
        except ValueError:
            continue
        if f0 <= 0:
            f0 = 0.0
        times.append(t)
        f0s.append(f0)

    if len(times) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    return np.array(times, dtype=np.float32), np.array(f0s, dtype=np.float32)


def interpolate_f0_to_frames(times: np.ndarray, f0: np.ndarray, cqt_frames: int) -> np.ndarray:
    if len(times) == 0 or len(f0) == 0:
        return np.zeros(cqt_frames, dtype=np.float32)

    frame_times = (np.arange(cqt_frames, dtype=np.float32) * config['hop_size']) / float(config['sample_rate'])
    f0_per_frame = np.interp(frame_times, times, f0, left=0.0, right=0.0)
    f0_per_frame[f0_per_frame < 0] = 0.0
    return f0_per_frame.astype(np.float32)

# dataset loading (MIR-1K, MIREX05, ADC2004)

def collect_mir1k_samples():
    print("Processing MIR-1K dataset...")

    wav_dir = Path(config['mir1k_path']) / "Wavfile"
    pitch_dir = Path(config['mir1k_path']) / "PitchLabel"

    if not wav_dir.exists():
        print(f"Warning: MIR-1K wavfile directory not found at {wav_dir}")
        return []

    samples = []
    wav_files = sorted(wav_dir.glob("*.wav"))

    for wav_path in tqdm(wav_files, desc="MIR-1K"):
        pv_path = pitch_dir / (wav_path.stem + ".pv")
        if not pv_path.exists():
            print(f"Warning: No pitch label for {wav_path.name}")
            continue

        audio, _ = librosa.load(wav_path, sr=config['sample_rate'], mono=True)
        times_10ms, f0_10ms = load_pv_annotation(pv_path)

        samples.append({
            "name": wav_path.stem,
            "source": "mir1k",
            "audio": audio,
            "times": times_10ms,
            "f0": f0_10ms,
        })

    print(f"Loaded {len(samples)} MIR-1K samples.")
    return samples


def collect_mirex05_samples():
    print("Processing MIREX05 dataset...")

    mirex_dir = Path(config['mirex05_path'])
    if not mirex_dir.exists():
        print(f"Warning: MIREX05 directory not found at {mirex_dir}")
        return []

    samples = []
    wav_files = sorted(mirex_dir.glob("*.wav"))

    for wav_path in tqdm(wav_files, desc="MIREX05"):
        base_name = wav_path.stem.replace("MIDI", "")
        ref_path = mirex_dir / f"{base_name}REF.txt"

        if not ref_path.exists():
            print(f"Warning: No REF file for {wav_path.name}")
            continue

        audio, _ = librosa.load(wav_path, sr=44100, mono=True)
        audio = librosa.resample(audio, orig_sr=44100, target_sr=config['sample_rate'])
        times, f0 = load_ref_annotation(ref_path)

        samples.append({
            "name": wav_path.stem,
            "source": "mirex05",
            "audio": audio,
            "times": times,
            "f0": f0,
        })

    print(f"Loaded {len(samples)} MIREX05 samples.")
    return samples

def collect_adc2004_samples():
    print("Processing ADC2004 dataset...")

    adc_dir = Path(config['adc2004_path'])
    if not adc_dir.exists():
        print(f"Warning: ADC2004 directory not found at {adc_dir}")
        return []

    samples = []
    wav_files = sorted(adc_dir.glob("*.wav"))

    for wav_path in tqdm(wav_files, desc="ADC2004"):
        ref_path = adc_dir / f"{wav_path.stem}REF.txt"

        if not ref_path.exists():
            print(f"Warning: No REF file for {wav_path.name}")
            continue

        audio, _ = librosa.load(wav_path, sr=44100, mono=True)
        audio = librosa.resample(audio, orig_sr=44100, target_sr=config['sample_rate'])
        times, f0 = load_ref_annotation(ref_path)

        samples.append({
            "name": wav_path.stem,
            "source": "adc2004",
            "audio": audio,
            "times": times,
            "f0": f0,
        })

    print(f"Loaded {len(samples)} ADC2004 samples.")
    return samples

# feature + label generation with optional augmentation

def create_segments(samples, augmentation: bool):

    X_segments = []
    pitch_idx_segments = []
    chroma_segments = []
    octave_segments = []
    voicing_segments = []
    pitch_hz_segments = []
    song_id_segments = []

    pitch_shifts = config['pitch_shifts'] if augmentation else [0]

    for sample in tqdm(samples, desc="Segmenting"):
        audio = sample["audio"]
        times = sample["times"]
        f0_base = sample["f0"]
        base_name = sample["name"]

        log_cqt_orig = compute_log_cqt(audio)
        cqt_frames_orig = log_cqt_orig.shape[1]
        f0_per_frame = interpolate_f0_to_frames(times, f0_base, cqt_frames_orig)

        for k in pitch_shifts:
            if k == 0:
                audio_shift = audio
            else:
                audio_shift = librosa.effects.pitch_shift(
                    audio, sr=config['sample_rate'], n_steps=k
                )
            
            # pad so all data enters segments
            log_cqt = compute_log_cqt(audio_shift)
            cqt_frames = log_cqt.shape[1]
            audio_pad_len = config['hop_size'] * (config['num_frames'] - (cqt_frames % config['num_frames']))
            padded_audio = np.concatenate((audio_shift, np.zeros((audio_pad_len))))

            log_cqt = compute_log_cqt(padded_audio)
            cqt_frames = log_cqt.shape[1]

            if k != 0:
                voiced_mask = f0_per_frame > 0
                f0_per_frame_shifted = f0_per_frame.copy()
                f0_per_frame_shifted[voiced_mask] *= (2.0 ** (k / 12.0))
            else:
                f0_per_frame_shifted = f0_per_frame

            pitch_idx = hz_array_to_ah1_indices(f0_per_frame_shifted)
            pitch_idx = np.concatenate((pitch_idx, np.zeros((cqt_frames - cqt_frames_orig))))

            chroma, octave = indices_to_chroma_octave(pitch_idx)
            voicing = (pitch_idx > 0).astype(np.int32)

            pitch_hz = indices_to_hz(pitch_idx)

            win = config['num_frames']

            for start in range(0, cqt_frames - win + 1, win):
                end = start + win

                X_seg = log_cqt[:, start:end]          
                p_seg = pitch_idx[start:end]           
                c_seg = chroma[start:end]
                o_seg = octave[start:end]
                v_seg = voicing[start:end]
                hz_seg = pitch_hz[start:end]

                X_segments.append(X_seg)
                pitch_idx_segments.append(p_seg)
                chroma_segments.append(c_seg)
                octave_segments.append(o_seg)
                voicing_segments.append(v_seg)
                pitch_hz_segments.append(hz_seg)
                song_id_segments.append(f"{base_name}_shift{k:+d}")

    print(f"Total segments created: {len(X_segments)}")
    return (
        X_segments,
        pitch_idx_segments,
        chroma_segments,
        octave_segments,
        voicing_segments,
        pitch_hz_segments,
        song_id_segments,
    )

def create_segmented_dataset(
    X_segments,
    pitch_idx_segments,
    chroma_segments,
    octave_segments,
    voicing_segments,
    pitch_hz_segments,
    song_id_segments
):
    X = np.stack(X_segments, axis=0).astype(np.float32)  # [N, 365, T]
    y_pitch_idx = np.stack(pitch_idx_segments, axis=0).astype(np.int32)  # [N, T]
    y_chroma = np.stack(chroma_segments, axis=0).astype(np.int32)
    y_octave = np.stack(octave_segments, axis=0).astype(np.int32)
    y_voicing = np.stack(voicing_segments, axis=0).astype(np.int32)
    y_pitch_hz = np.stack(pitch_hz_segments, axis=0).astype(np.float32)
    song_ids = np.array(song_id_segments)

    X = np.expand_dims(X, axis=1)

    # optional per-frequency normalization over training set
    # mean/std computed over (N, time) for each freq
    mean = X.mean(axis=(0, 3), keepdims=True)
    std = X.std(axis=(0, 3), keepdims=True) + 1e-8
    X_norm = (X - mean) / std

    dataset = {
        "X": X_norm,
        "y_pitch_idx": y_pitch_idx,
        "y_chroma": y_chroma,
        "y_octave": y_octave,
        "y_voicing": y_voicing,
        "y_pitch_hz": y_pitch_hz,
        "song_ids": song_ids,
        "norm_stats": {"mean": mean.astype(np.float32), "std": std.astype(np.float32)},
    }
    return dataset

# main pipeline

def run_preprocessing():

    # fixes train/val/test split for reproducibility
    random.seed(config['random_seed'])

    output_dir = Path(config['dataset_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    mir1k_samples = collect_mir1k_samples()
    mirex05_samples = collect_mirex05_samples()
    train_val_samples = mir1k_samples + mirex05_samples
    test_split = collect_adc2004_samples()

    if not train_val_samples:
        print("Error: no train/validation samples loaded. Check your dataset paths.")
        return None
    if not test_split:
        print("Error: no testing samples loaded. Check your dataset paths.")
        return None
    sample_count = len(train_val_samples)
    print(f"\nTotal training/validation samples: {sample_count}")
    print(f"\nTotal testing samples: {len(test_split)}")

    # make train, val, test splits according to config ratios
    random.shuffle(train_val_samples)
    train_split = train_val_samples[:int(sample_count * config['train_ratio'])]
    val_split = train_val_samples[int(sample_count * config['train_ratio']):]

    print("Creating training segments with augmentation...")
    train_dataset = create_segmented_dataset(
        *create_segments(train_split, True)
    )

    print("Creating validation segments...")
    val_dataset = create_segmented_dataset(
        *create_segments(val_split, False)
    )

    print("Creating testing segments...")
    test_dataset = create_segmented_dataset(
        *create_segments(test_split, False)
    )
    
    print("\nDataset shapes:   train / val / test")
    print(f"  X (log-CQT):    {train_dataset['X'].shape} / {val_dataset['X'].shape} / {test_dataset['X'].shape}")           
    print(f"  y_pitch_idx:    {train_dataset['y_pitch_idx'].shape} / {val_dataset['y_pitch_idx'].shape} / {test_dataset['y_pitch_idx'].shape}") 
    print(f"  y_chroma:       {train_dataset['y_chroma'].shape} / {val_dataset['y_chroma'].shape} / {test_dataset['y_chroma'].shape}")    
    print(f"  y_octave:       {train_dataset['y_octave'].shape} / {val_dataset['y_octave'].shape} / {test_dataset['y_octave'].shape}")    
    print(f"  y_voicing:      {train_dataset['y_voicing'].shape} / {val_dataset['y_voicing'].shape} / {test_dataset['y_voicing'].shape}")   
    print(f"  y_pitch_hz:     {train_dataset['y_pitch_hz'].shape} / {val_dataset['y_pitch_hz'].shape} / {test_dataset['y_pitch_hz'].shape}")  
    print(f"  song_ids:       {train_dataset['song_ids'].shape} / {val_dataset['song_ids'].shape} / {test_dataset['song_ids'].shape}")

    def save_compressed(file, dataset):
        return np.savez_compressed(
            file=file,
            X=dataset["X"],
            y_pitch_idx=dataset["y_pitch_idx"],
            y_chroma=dataset["y_chroma"],
            y_octave=dataset["y_octave"],
            y_voicing=dataset["y_voicing"],
            y_pitch_hz=dataset["y_pitch_hz"],
            song_ids=dataset["song_ids"],
            norm_mean=dataset["norm_stats"]["mean"],
            norm_std=dataset["norm_stats"]["std"]
        )
    
    save_compressed(config['train_set_path'], train_dataset)
    save_compressed(config['val_set_path'], val_dataset)
    save_compressed(config['test_set_path'], test_dataset)

    print(f"\nSaved processed data to {config['dataset_dir']}")

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    run_preprocessing()
