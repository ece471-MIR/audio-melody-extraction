# import numpy as np
# import torch
# import librosa
# import sys
# from typing import Tuple

# import preprocessing
# from train import calculate_accuracies
# from config import preproc_config as config

# def process_test_sample(audio, times, f0):

#     log_cqt = preprocessing.compute_log_cqt(audio)
#     n_frames = log_cqt.shape[1]

#     f0_per_frame = preprocessing.interpolate_f0_to_frames(times, f0, n_frames)
#     pitch_idx = preprocessing.hz_array_to_ah1_indices(f0_per_frame)

#     chroma, octave = preprocessing.indices_to_chroma_octave(pitch_idx)
#     voicing = (pitch_idx > 0).astype(np.int32) 
#     pitch_hz = preprocessing.indices_to_hz(pitch_idx)

#     step = config['frame_step']
#     win = config['num_frames']

#     X_segments = []
#     pitch_idx_segments = []
#     chroma_segments = []
#     octave_segments = []
#     voicing_segments = []
#     pitch_hz_segments = []

#     for start in range(0, n_frames - win + 1, step):
#         end = start + win

#         X_segments.append(log_cqt[:, start:end])
#         pitch_idx_segments.append(pitch_idx[start:end])
#         chroma_segments.append(chroma[start:end])
#         octave_segments.append(octave[start:end])
#         voicing_segments.append(voicing[start:end])
#         pitch_hz_segments.append(pitch_hz[start:end])
    
#     return (
#         X_segments,
#         pitch_idx_segments,
#         chroma_segments,
#         octave_segments,
#         voicing_segments,
#         pitch_hz_segments
#     )

# def window_inference(model, x,
#                      y_chroma, y_octave, y_voicing):
#     for i in range(0, len(x)):
#         chroma_logits, octave_logits, voicing_logits = model(x[i])
#         c_cor, o_cor, v_cor, v_rec, v_far, v_true, v_false, frames = calculate_accuracies(
#             chroma_logits, octave_logits, voicing_logits,
#             y_chroma, y_octave, y_voicing
#         )


#     return 1

# def infer():
#     if len(sys.argv) != 3:
#         print('Usage: python infer.py MODEL_PATH TEST_SAMPLE_STEM')
#         exit()
    
#     name = sys.argv[2]
#     wav_path = config['adc2004_path'] + '/' + name + '.wav'
#     ref_path = config['adc2004_path'] + '/' + name + 'REF.txt'
#     if not wav_path.exists():
#         print(f"{wav_path} does not exist.")
#         exit()
#     if not ref_path.exists():
#         print(f"{ref_path} does not exist.")
#         exit()
    
#     audio, _ = librosa.load(wav_path, sr=44100, mono=True)
#     audio = librosa.resample(audio, orig_sr=44100, target_sr=config['sample_rate'])
#     times, f0 = preprocessing.load_ref_annotation(ref_path)

#     (   X_segments,
#         pitch_idx_segments,
#         chroma_segments,
#         octave_segments,
#         voicing_segments,
#         pitch_hz_segments
#     ) = process_test_sample(audio, times, f0)

#     model = torch.load(sys.argv[1])

#     window_inference(
#         model,
#         X_segments,
#         chroma_segments,
#         octave_segments,
#         voicing_segments
#     )

# if __name__ == "__main__":
#     infer()