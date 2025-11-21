"""
AH1 Melody Extraction Preprocessing Pipeline

Processes MIR-1K and MIREX05 datasets for training a multi-task CRNN,
following the AH1 version description from MIREX 2020 Audio Melody Extraction.
"""

import numpy as np
import librosa
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# configuration

class Config:
    SAMPLE_RATE = 16000
    HOP_SIZE = 256  

    CQT_FMIN = 65.0    
    CQT_BINS = 365      
    BINS_PER_OCTAVE = 57  

    NUM_FRAMES = 517         
    FRAME_STEP = NUM_FRAMES // 2  

    
    C2_FREQ = 65.406   
    NUM_PITCHES = 48   


    PITCH_SHIFTS = [-2, -1, 0, 1, 2]

    MIR1K_PATH = "./datasets/MIR-1K"             
    MIREX05_PATH = "./datasets/mirex05TrainFiles"
    OUTPUT_PATH = "./processed_data"


# AH1 pitch utilities

def hz_to_ah1_index(f0_hz: float, c2_freq: float, num_pitches: int) -> int:
    if f0_hz <= 0:
        return 0
    semitone = 12.0 * np.log2(f0_hz / c2_freq)
    idx = int(round(semitone)) + 1  
    if 1 <= idx <= num_pitches:
        return idx
    return 0


def hz_array_to_ah1_indices(f0_hz: np.ndarray, config: Config) -> np.ndarray:
    func = np.vectorize(lambda x: hz_to_ah1_index(x, config.C2_FREQ, config.NUM_PITCHES))
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


def indices_to_hz(pitch_idx: np.ndarray, config: Config) -> np.ndarray:
    f0_hz = np.zeros_like(pitch_idx, dtype=np.float32)
    mask = pitch_idx > 0
    semitone = (pitch_idx[mask].astype(np.float32) - 1.0)
    f0_hz[mask] = config.C2_FREQ * (2.0 ** (semitone / 12.0))
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

def compute_log_cqt(audio: np.ndarray, config: Config) -> np.ndarray:
    """
    Compute log-magnitude CQT.

    Returns:
        log_cqt: [freq_bins=365, time_frames=T]
    """
    cqt = librosa.cqt(
        audio,
        sr=config.SAMPLE_RATE,
        hop_length=config.HOP_SIZE,
        fmin=config.CQT_FMIN,
        n_bins=config.CQT_BINS,
        bins_per_octave=config.BINS_PER_OCTAVE,
        window="blackmanharris",
    )
    mag = np.abs(cqt)
    log_cqt = np.log1p(mag)  
    return log_cqt.astype(np.float32)

# annotation loaders

def load_mir1k_annotation(pv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    MIR-1K .pv files contain f0 values at 10 ms intervals (Hz).
    Returns:
        times_10ms: [N] in seconds
        f0_10ms: [N] in Hz
    """
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


def load_mirex05_annotation(ref_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    MIREX05 REF files: "timestamp_sec pitch_hz" per line.
    Returns:
        times: [N] seconds
        f0:    [N] Hz
    """
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


def interpolate_f0_to_frames(times: np.ndarray, f0: np.ndarray, n_frames: int, config: Config) -> np.ndarray:
    """
    Given sparse f0 annotation (times, f0), interpolate to per-frame f0
    for a CQT with 'n_frames' frames and hop size config.HOP_SIZE.

    Returns:
        f0_per_frame: [n_frames] in Hz
    """
    if len(times) == 0 or len(f0) == 0:
        return np.zeros(n_frames, dtype=np.float32)

    frame_times = (np.arange(n_frames, dtype=np.float32) * config.HOP_SIZE) / float(config.SAMPLE_RATE)
    f0_per_frame = np.interp(frame_times, times, f0, left=0.0, right=0.0)
    f0_per_frame[f0_per_frame < 0] = 0.0
    return f0_per_frame.astype(np.float32)

# dataset loading (MIR-1K + MIREX05)

def collect_mir1k_samples(config: Config):
    """
    Collect MIR-1K samples: each item includes:
        name, source, audio, times, f0
    where times/f0 are 10ms annotation (we'll interpolate later).
    """
    print("Processing MIR-1K dataset...")

    wav_dir = Path(config.MIR1K_PATH) / "Wavfile"
    pitch_dir = Path(config.MIR1K_PATH) / "PitchLabel"

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

        audio, _ = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
        times_10ms, f0_10ms = load_mir1k_annotation(pv_path)

        samples.append({
            "name": wav_path.stem,
            "source": "mir1k",
            "audio": audio,
            "times": times_10ms,
            "f0": f0_10ms,
        })

    print(f"Loaded {len(samples)} MIR-1K samples.")
    return samples


def collect_mirex05_samples(config: Config):
    """
    Collect MIREX05 samples: each item includes:
        name, source, audio, times, f0
    """
    print("Processing MIREX05 dataset...")

    mirex_dir = Path(config.MIREX05_PATH)
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

        audio, _ = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
        times, f0 = load_mirex05_annotation(ref_path)

        samples.append({
            "name": wav_path.stem,
            "source": "mirex05",
            "audio": audio,
            "times": times,
            "f0": f0,
        })

    print(f"Loaded {len(samples)} MIREX05 samples.")
    return samples

# feature + label generation with augmentation

def create_training_segments(samples, config: Config):
    """
    For each sample:
        - For each pitch shift k ∈ PITCH_SHIFTS:
            - pitch-shift audio by k semitones
            - compute CQT
            - interpolate base f0 to frames
            - shift f0 by k semitones in Hz
            - convert to AH1 pitch_idx, chroma, octave, voicing
            - cut into segments of NUM_FRAMES

    Returns:
        X_segments:      list of [365, NUM_FRAMES]
        pitch_idx_segs:  list of [NUM_FRAMES]
        chroma_segs:     list of [NUM_FRAMES]
        octave_segs:     list of [NUM_FRAMES]
        voicing_segs:    list of [NUM_FRAMES]
        pitch_hz_segs:   list of [NUM_FRAMES]
        song_id_segs:    list of str
    """
    print("Creating training segments with augmentation...")

    X_segments = []
    pitch_idx_segments = []
    chroma_segments = []
    octave_segments = []
    voicing_segments = []
    pitch_hz_segments = []
    song_id_segments = []

    for sample in tqdm(samples, desc="Augmenting"):
        audio = sample["audio"]
        times = sample["times"]
        f0_base = sample["f0"]
        base_name = sample["name"]

        for k in config.PITCH_SHIFTS:
            if k == 0:
                audio_shift = audio
            else:
                audio_shift = librosa.effects.pitch_shift(
                    audio, sr=config.SAMPLE_RATE, n_steps=k
                )

            log_cqt = compute_log_cqt(audio_shift, config)  
            n_frames = log_cqt.shape[1]

            f0_per_frame = interpolate_f0_to_frames(times, f0_base, n_frames, config)

            if k != 0:
                voiced_mask = f0_per_frame > 0
                f0_per_frame_shifted = f0_per_frame.copy()
                f0_per_frame_shifted[voiced_mask] *= (2.0 ** (k / 12.0))
            else:
                f0_per_frame_shifted = f0_per_frame

            pitch_idx = hz_array_to_ah1_indices(f0_per_frame_shifted, config)

            chroma, octave = indices_to_chroma_octave(pitch_idx)
            voicing = (pitch_idx > 0).astype(np.int32)

            pitch_hz = indices_to_hz(pitch_idx, config)

            step = config.FRAME_STEP
            win = config.NUM_FRAMES

            for start in range(0, n_frames - win + 1, step):
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


def create_windowed_dataset(
    X_segments,
    pitch_idx_segments,
    chroma_segments,
    octave_segments,
    voicing_segments,
    pitch_hz_segments,
    song_id_segments,
    config: Config,
):
    """
    Stack lists into arrays and add channel dimension:
    X: [N, 1, 365, NUM_FRAMES]
    """
    X = np.stack(X_segments, axis=0).astype(np.float32)  # [N, 365, T]
    y_pitch_idx = np.stack(pitch_idx_segments, axis=0).astype(np.int32)  # [N, T]
    y_chroma = np.stack(chroma_segments, axis=0).astype(np.int32)
    y_octave = np.stack(octave_segments, axis=0).astype(np.int32)
    y_voicing = np.stack(voicing_segments, axis=0).astype(np.int32)
    y_pitch_hz = np.stack(pitch_hz_segments, axis=0).astype(np.float32)
    song_ids = np.array(song_id_segments)

    X = np.expand_dims(X, axis=1)

    # Optional per-frequency normalization over training set
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

def run_preprocessing(config: Config = Config()):
    output_dir = Path(config.OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    mir1k_samples = collect_mir1k_samples(config)
    mirex05_samples = collect_mirex05_samples(config)
    all_samples = mir1k_samples + mirex05_samples

    if not all_samples:
        print("Error: no samples loaded. Check your dataset paths.")
        return None

    print(f"\nTotal base samples: {len(all_samples)}")

    (
        X_segments,
        pitch_idx_segments,
        chroma_segments,
        octave_segments,
        voicing_segments,
        pitch_hz_segments,
        song_id_segments,
    ) = create_training_segments(all_samples, config)

    dataset = create_windowed_dataset(
        X_segments,
        pitch_idx_segments,
        chroma_segments,
        octave_segments,
        voicing_segments,
        pitch_hz_segments,
        song_id_segments,
        config,
    )

    print("\nDataset shapes:")
    print(f"  X (log-CQT):    {dataset['X'].shape}")           
    print(f"  y_pitch_idx:    {dataset['y_pitch_idx'].shape}") 
    print(f"  y_chroma:       {dataset['y_chroma'].shape}")    
    print(f"  y_octave:       {dataset['y_octave'].shape}")    
    print(f"  y_voicing:      {dataset['y_voicing'].shape}")   
    print(f"  y_pitch_hz:     {dataset['y_pitch_hz'].shape}")  
    print(f"  song_ids:       {dataset['song_ids'].shape}")

    out_file = output_dir / "ah1_training_data.npz"
    np.savez_compressed(
        out_file,
        X=dataset["X"],
        y_pitch_idx=dataset["y_pitch_idx"],
        y_chroma=dataset["y_chroma"],
        y_octave=dataset["y_octave"],
        y_voicing=dataset["y_voicing"],
        y_pitch_hz=dataset["y_pitch_hz"],
        song_ids=dataset["song_ids"],
        norm_mean=dataset["norm_stats"]["mean"],
        norm_std=dataset["norm_stats"]["std"],
    )
    print(f"\nSaved processed data to {out_file}")

    config_file = output_dir / "config.pkl"
    with open(config_file, "wb") as f:
        pickle.dump(
            {
                "sample_rate": config.SAMPLE_RATE,
                "hop_size": config.HOP_SIZE,
                "cqt_fmin": config.CQT_FMIN,
                "cqt_bins": config.CQT_BINS,
                "bins_per_octave": config.BINS_PER_OCTAVE,
                "num_frames": config.NUM_FRAMES,
                "frame_step": config.FRAME_STEP,
                "num_pitches": config.NUM_PITCHES,
                "pitch_shifts": config.PITCH_SHIFTS,
            },
            f,
        )
    print(f"Saved config to {config_file}")

    return dataset


if __name__ == "__main__":
    run_preprocessing()
