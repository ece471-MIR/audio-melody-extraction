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

    'c2_freq': 65.406,
    'num_pitches': common_config['chroma_classes']
                 * common_config['octave_classes'],
    'pitch_shifts': [-2, -1, 0, 1, 2],

    'mir1k_path': './datasets/MIR-1K',
    'mirex05_path': './datasets/mirex05TrainFiles',
    'adc2004_path': './datasets/adc2004_full_set',
    'dataset_dir': './processed_data',

    'random_seed': 42,
    'train_ratio': 0.8,
    'val_ratio': 0.2,
}
preproc_config.update(common_config)

model_config = {
    'lstm_hidden_size': 128,
    'fc_hidden_size': 256,

    'dropout_rate': 0.5
}
model_config.update(common_config)

train_config = {
    'model_dir': './models',
    'epochs': 15,
    'batch_size': 8,
    'learning_rate': 2.5e-5,
    'weight_decay': 2e-4,
    'unvoiced_weight': 1,
    'voiced_weight': 9,
    'voicing_loss_weight': 2
}
train_config.update(common_config)
