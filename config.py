common_config = {
    'cqt_bins': 365,
    'num_frames': 517,

    'chroma_classes': 12,
    'octave_classes': 4,
    'voicing_classes': 2,
    'train_set_path': './processed_data/ah1_train_set.npz',
    'val_set_path': './processed_data/ah1_val_set.npz',
    'test_set_path': './processed_data/ah1_test_set.npz'
}

preproc_config = {
    'sample_rate': 16000,
    'hop_size': 256,

    'cqt_fmin': 65.0,
    'bins_per_octave': 57,
    'window': 'blackmanharris',

    'frame_step': common_config['num_frames'] // 2,
    'c2_freq': 65.406,
    
    'num_pitches': common_config['chroma_classes']
                 * common_config['octave_classes'],
    'pitch_shifts': [-2, -1, 0, 1, 2],

    'mir1k_path': './datasets/MIR-1K',
    'mirex05_path': './datasets/mirex05TrainFiles',
    'dataset_dir': './processed_data',

    'random_seed': 37,
    'train_ratio': 0.50,
    'val_ratio': 0.25,
    'test_ratio': 0.25
}
preproc_config.update(common_config)

model_config = {
    'lstm_hidden_size': 128,
    'fc_hidden_size': 256,

    'dropout_rate': 0.6
}
model_config.update(common_config)

train_config = {
    'model_dir': './models',
    'epochs': 20,
    'batch_size': 16,
    'learning_rate': 5e-5,
    'unvoiced_weight': 1,
    'voiced_weight': 8
}
train_config.update(common_config)