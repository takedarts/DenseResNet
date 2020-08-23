import models

import utils.pthflops
import torch.jit

import logging
import argparse
import os
import functools
import warnings

warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)

LOGGER = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data')

parser = argparse.ArgumentParser(
    description='Calculate flops.', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('file', help='File name.')
parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')


def count_python_op(node):
    if node.pyname() == 'AdjustedStackFunction':
        return 0
    elif node.pyname() == 'GatedFunction':
        inputs = list(node.inputs())

        size = utils.pthflops.string_to_shape(inputs[-1])
        count = functools.reduce(lambda x, y: x * y, size)

        return 2 * (len(inputs) - 1) * count
    else:
        raise Exception('unsupported python operation: {}'.format(node.pyname()))


def count_div(node):
    inp = utils.pthflops.string_to_shape(list(node.inputs())[0])

    if inp is not None:
        return functools.reduce(lambda x, y: x * y, inp)
    else:
        return 0


def count_exp(node):
    inp = utils.pthflops.string_to_shape(list(node.inputs())[0])

    return 1 * functools.reduce(lambda x, y: x * y, inp)


def count_sigmoid(node):
    inp = utils.pthflops.string_to_shape(list(node.inputs())[0])

    return 4 * functools.reduce(lambda x, y: x * y, inp)


def count_sum(node):
    inp = utils.pthflops.string_to_shape(list(node.inputs())[0])

    if inp is not None:
        return functools.reduce(lambda x, y: x * y, inp)
    else:
        return 0


def count_mean(node):
    inp = utils.pthflops.string_to_shape(list(node.inputs())[0])

    if inp is not None:
        return functools.reduce(lambda x, y: x * y, inp)
    else:
        return 0


def zero_op(node):  # @UnusedVariable
    return 0


custom_ops = {
    'prim::PythonOp': count_python_op,
    'onnx::Div': count_div,
    'onnx::Exp': count_exp,
    'onnx::Sigmoid': count_sigmoid,
    'onnx::ReduceSum': count_sum,
    'onnx::ReduceMean': count_mean,
    'onnx::Concat': zero_op,
    'onnx::Constant': zero_op,
    'onnx::Gather': zero_op,
    'onnx::Pad': zero_op,
    'onnx::Split': zero_op,
    'onnx::Reshape': zero_op,
    'onnx::Shape': zero_op,
    'onnx::Unsqueeze': zero_op,
    'onnx::Squeeze': zero_op,
}


def main():
    args = parser.parse_args()
    utils.setup_logging(args.debug)

    # load file
    snapshot = torch.load(args.file, map_location=lambda s, _: s)
    params = snapshot['params']

    # model
    models.CONFIG.semodule_reduction = params.semodule_reduction
    models.CONFIG.gate_reduction = params.gate_reduction
    models.CONFIG.gate_connections = params.gate_connections

    model = models.create_model(
        params.dataset, params.model,
        shakedrop=params.shakedrop_prob)
    print('parameters={:,d}'.format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # flops
    image = utils.load_dataset(
        params.dataset, DATA_DIR, params.valid_crop,
        train=False, stdaug=False, autoaug=False)[0][0].unsqueeze(0)
    flops, groups = utils.pthflops.count_ops(
        model, image, custom_ops=custom_ops, verbose=args.debug, print_readable=False)

    for name, value in groups:
        print(f'{value:8d} - {name}')

    print(f'total: {flops:,d} [flops]')


if __name__ == '__main__':
    main()
