OPTIONS = [
    ('train-batch', 64, 'Batch size at training.'),
    ('train-crop', 224, 'Input image size at training.'),
    ('train-epoch', 300, 'Number of epochs at training.'),
    ('train-warmup', 5, 'Number of epochs for warmup at training.'),
    ('train-lr', 0.025, 'Initial learning rate at training'),
    ('train-wdecay', 0.0001, 'Weight decay  (L2 penalty) at training.'),
    ('train-bdecay', False, 'Adaptation of decay to bias parameters.'),
    ('tune-batch', 256, 'Batch size at tuning.'),
    ('tune-epoch', 60, 'Number of epochs at tuning.'),
    ('tune-lr', 0.004, 'Learning rate at tuning.'),
    ('tune-wdecay', 0.0001, 'Weight decay  (L2 penalty) at tuning.'),
    ('valid-crop', 224, 'Input image size at validation.'),
    ('cutmix-beta', 1.0, 'Beta parameter of CutMix at training.'),
    ('cutmix-prob', 0.0, 'Probability of CutMix at training.'),
    ('mixup-beta', 1.0, 'Beta parameter of Mixup at training.'),
    ('mixup-prob', 0.0, 'Probability of Mixup at training.'),
    ('auto-augment', False, 'Use of auto augumentation.'),
    ('label-smooth', 0.0, 'Factor "k" of label smoothing.'),
    ('dropout-prob', 0.0, 'Probability of dropout.'),
    ('shakedrop-prob', 0.0, 'Probability of shake-drop.'),
    ('dropblock-prob', 0.0, 'Drop probability of DropBlock.'),
    ('dropblock-size', 7, 'Drop block size of DropBlock.'),
    ('stochdepth-prob', 0.0, 'Drop probability of stochastic depth.'),
    ('signal-augment', 0.0, 'Standard deviation of signal augmentation.'),
    ('semodule-reduction', 16, 'Reduction ratio of Squeeze-and-Excitation modules.'),
    ('gate-reduction', 8, 'reduction rate of gate modules in DenseResNets.'),
    ('gate-connections', 4, 'number of connections of gate modules in DenseResNets.'),
    ('seed', 2020, 'Seed of random libraries.'),
]


def arg_type_bool(x):
    return x.lower() in ('true', 'yes', '1')
