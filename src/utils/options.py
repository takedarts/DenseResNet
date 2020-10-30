OPTIONS = [
    ('train-batch', 64, 'Batch size at training.'),
    ('train-crop', 224, 'Input image size at training.'),
    ('train-epoch', 300, 'Number of epochs at training.'),
    ('train-warmup', 5, 'Number of epochs for warmup at training.'),
    ('train-lr', 0.025, 'Initial learning rate at training'),
    ('train-wdecay', 0.0001, 'Weight decay (L2 penalty) at training.'),
    ('train-bdecay', False, 'Adaptation of decay (L2 penalty) to bias parameters.'),
    ('tune-batch', 256, 'Batch size at tuning.'),
    ('tune-epoch', 60, 'Number of epochs at tuning.'),
    ('tune-lr', 0.004, 'Learning rate at tuning.'),
    ('tune-wdecay', 0.0001, 'Weight decay (L2 penalty) at tuning.'),
    ('valid-crop', 224, 'Input image size at validation.'),
    ('cutmix-alpha', 1.0, 'Distribution parameter Alpha of CutMix at training.'),
    ('cutmix-prob', 0.0, 'Probability of CutMix at training.'),
    ('mixup-alpha', 1.0, 'Distribution parameter Alpha of Mixup at training.'),
    ('mixup-prob', 0.0, 'Probability of Mixup at training.'),
    ('randomerasing-prob', 0.0, 'Probability of RandomErasing.'),
    ('randomerasing-type', ['random', 'zero'], 'Type of RandomErasing.'),
    ('autoaugment', False, 'Use of auto augumentation.'),
    ('labelsmooth', 0.0, 'Factor "k" of label smoothing.'),
    ('dropout-prob', 0.0, 'Probability of dropout.'),
    ('shakedrop-prob', 0.0, 'Probability of shake-drop.'),
    ('dropblock-prob', 0.0, 'Drop probability of DropBlock.'),
    ('dropblock-size', 7, 'Drop block size of DropBlock.'),
    ('stochdepth-prob', 0.0, 'Drop probability of stochastic depth.'),
    ('signalaugment', 0.0, 'Standard deviation of signal augmentation.'),
    ('semodule-reduction', 16, 'Reduction ratio of "Squeeze and Excitation" modules.'),
    ('affmodule-reduction', 16, 'Reduction ratio of "Attentional Feature Fusion" modules.'),
    ('gate-reduction', 8, 'reduction rate of gate modules in DenseResNets or SkipResNets.'),
    ('dense-connections', 4, 'number of connections of gate modules in DenseResNets.'),
    ('skip-connections', 16, 'number of connections of gate modules in SkipResNets.'),
    ('seed', 2020, 'Seed of random libraries.'),
]


def arg_type_bool(x):
    return x.lower() in ('true', 'yes', '1')
