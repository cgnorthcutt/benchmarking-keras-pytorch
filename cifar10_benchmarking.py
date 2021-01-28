# These imports enhance Python2/3 compatibility.
from __future__ import (
    print_function, absolute_import, division,
    unicode_literals, with_statement,
)

import numpy as np
import os
import argparse
from torchvision.datasets import CIFAR10
import pandas as pd

VAL_SIZE = 10000
DATASET = 'cifar10'
# The following list is only used for debugging purposes.
label2name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
              'horse', 'ship', 'truck']

models = {
    "pytorch": [
        'vgg11_bn',
        'vgg13_bn',
        'vgg16_bn',
        'vgg19_bn',
        'resnet18',
        'resnet34',
        'resnet50',
        'densenet121',
        'densenet161',
        'densenet169',
        'mobilenet_v2',
        'googlenet',
        'inception_v3',
    ],
}

# Set up argument parser
parser = argparse.ArgumentParser(description='PyTorch and Keras Benchmarking')
parser.add_argument('val_dir', metavar='DIR',
                    help='path to  val dataset folder')
parser.add_argument('-k', '--keras-dir', metavar='KERASDIR',
                    default="keras_imagenet/",
                    help='directory where Keras model outputs are stored.')
parser.add_argument('-p', '--pytorch-dir', metavar='PYTORCHDIR',
                    default="pytorch_imagenet/",
                    help='directory where PyTorch model outputs are stored.')
parser.add_argument('-o', '--output-dir', metavar='OUTPUTFN',
                    default="benchmarking_results.csv",
                    help='csv filename to store the benchmarking csv results.')
parser.add_argument('-i', '--indices_to_omit', metavar='INDICES_TO_OMIT',
                    default=None,
                    help='path of numpy .npy file of storing val indices to omit.'
                         'ONLY ONE of --indices_to_omit and --indices_to_keep'
                         'may be used. DO NOT USE BOTH.')
parser.add_argument('-j', '--indices_to_keep', default=None,
                    help='path of numpy .npy file of storing val indices to use.'
                         'ONLY ONE of --indices_to_omit and --indices_to_keep'
                         'may be used. DO NOT USE BOTH.')
parser.add_argument('-l', '--custom_labels', default=None,
                    help='path of numpy .npy file storing custom labels.'
                         'Must match same length as sum(indices_to_keep)'
                         'if indices_to_keep is specfied (similarly to '
                         'indices_to_omit)')


# Important notes about how this works in terms of 
#  --indices_to_omit, --indices_to_keep, and --custom_labels
# The algorithm:
# if no custom labels and mask1 or mask2 given
# - compute acc1, acc5 using full
# - compute cacc1, cacc5 using mask
# if custom labels and mask1 or mask2 given
# - compute acc1, acc5 using mask
# - compute cacc1, cacc5 using mask + custom labels
# if no custome labels and no mask given
# - compute acc1, acc5
# if custome labels and no mask given
# - compute acc1, acc5
# -compute cacc1, cacc5 using custom labels


def compute_val_acc(top5preds, top5probs, labels, indices_to_ignore=None,
                    indices_to_keep=None, custom_labels=None):
    # Create a mask of having True for each example we want to include in scoring.
    bool_mask = np.ones(VAL_SIZE).astype(bool)
    if indices_to_ignore is not None:
        bool_mask[indices_to_ignore] = False
    if indices_to_keep is not None:
        bool_mask = np.zeros(VAL_SIZE).astype(bool)
        bool_mask[indices_to_keep] = True
    if custom_labels is not None:
        true = np.asarray(custom_labels)
    else:
        true = labels[bool_mask]
    pred = top5preds[range(len(top5preds)), np.argmax(top5probs, axis=1)]
    pred = pred[bool_mask]
    # print('hey!')
    # print(len(pred), pred)
    # print(len(true), true)
    # print(indices_to_keep)
    # print("sum(bool_mask)", sum(bool_mask))
    # print("len(pred)", len(pred))
    # print("len(true)", len(true))
    # print("len(indices_to_keep)", len(indices_to_keep))
    # # print("len(indices_to_ignore)", len(indices_to_ignore))
    acc1 = sum(pred == true) / float(len(true))
    acc5 = sum(
        [true[i] in row for i, row in enumerate(top5preds[bool_mask])]) / float(
        len(true))
    return acc1, acc5


def main(args=parser.parse_args()):
    '''if no custom labels and mask1 or mask2 given
    - compute acc1, acc5 using full
    - compute cacc1, cacc5 using mask
    if custom labels and mask1 or mask2 given
    - compute acc1, acc5 using mask
    - compute cacc1, cacc5 using mask + custom labels
    if no custome labels and no mask given
    - compute acc1, acc5
    if custome labels and no mask given
    - compute acc1, acc5
    -compute cacc1, cacc5 using custom labels'''

    # Grab dataset data
    val_dataset = CIFAR10(root=args.val_dir, train=False, download=True, )
    labels = np.asarray(val_dataset.targets)

    if args.indices_to_omit:
        indices_to_omit = np.load(args.indices_to_omit)
        print('Only computing scores for {} of {} labels.'.format(
            len(labels) - len(indices_to_omit), len(labels),
        ))
    else:
        indices_to_omit = None
    if args.indices_to_keep:
        indices_to_keep = np.load(args.indices_to_keep)
        print('Only computing scores for {} of {} labels.'.format(
            len(indices_to_keep), len(labels),
        ))
    else:
        indices_to_keep = None
    if args.custom_labels:
        custom_labels = np.load(args.custom_labels, allow_pickle=True)
    else:
        custom_labels = None



    print(indices_to_keep)
    print(custom_labels)


    dirs = {
        "keras": args.keras_dir,
        "pytorch": args.pytorch_dir,
    }

    data = []
    for platform_name in ["Keras 2.2.4", "PyTorch 1.0"]:
        platform = platform_name.split()[0].lower()
        if platform not in models:
            continue  # No models on that platform.

        # Read in data and compute accuracies
        pred_suffix = "_{}_{}_top5preds.npy".format(platform, DATASET)
        prob_suffix = "_{}_{}_top5probs.npy".format(platform, DATASET)
        for model in models[platform]:
            top5preds = np.load(
                os.path.join(dirs[platform], model + pred_suffix))
            top5probs = np.load(
                os.path.join(dirs[platform], model + prob_suffix))
            if 'nasnet' in model:
                pred = top5preds[
                    range(len(top5preds)), np.argmax(top5probs, axis=1)]
                np.save('nasnet_correct_mask', pred == labels)
            # This code is just computing accuracy first with as few
            # modifications to the original val set as possible and then again
            # will all modifications.
            if args.custom_labels is None and (
                    args.indices_to_omit or args.indices_to_keep):
                acc1, acc5 = compute_val_acc(top5preds, top5probs, labels)
                acc1c, acc5c = compute_val_acc(top5preds, top5probs, labels,
                                               indices_to_omit, indices_to_keep)
            elif args.custom_labels and (
                    args.indices_to_omit or args.indices_to_keep):
                acc1, acc5 = compute_val_acc(top5preds, top5probs, labels,
                                             indices_to_omit, indices_to_keep)
                acc1c, acc5c = compute_val_acc(top5preds, top5probs, labels,
                                               indices_to_omit, indices_to_keep,
                                               custom_labels)
            elif args.custom_labels and not (
                    args.indices_to_omit or args.indices_to_keep):
                acc1, acc5 = compute_val_acc(top5preds, top5probs, labels)
                acc1c, acc5c = compute_val_acc(top5preds, top5probs, labels,
                                               custom_labels)
            else:  # args.custom_labels is None and not (args.indices_to_omit or args.indices_to_keep):
                acc1, acc5 = compute_val_acc(top5preds, top5probs, labels,
                                             indices_to_omit, indices_to_keep)
            if args.custom_labels or args.indices_to_omit or args.indices_to_keep:
                data.append((platform_name, model, acc1, acc1c, acc5, acc5c))
            else:
                data.append((platform_name, model, acc1, acc5))
    if args.custom_labels or args.indices_to_omit or args.indices_to_keep:
        cols = ["Platform", "Model", "Acc@1", "cAcc@1", "Acc@5", "cAcc@5", ]
    else:
        cols = ["Platform", "Model", "Acc@1", "Acc@5", ]
    df = pd.DataFrame(data, columns=cols)
    # Add rankings for each accuracy metric
    for col in [c for c in df.columns if 'Acc' in c]:
        df.sort_values(by=col, ascending=False, inplace=True)
        df[col.replace('Acc', 'Rank')] = np.arange(len(df)) + 1
    # Final sort by Acc@1 column
    df.sort_values(by="Acc@1", ascending=False, inplace=True)
    df = df.reset_index(drop=True)
    df.to_csv(args.output_dir, index=False)
    print(df)


if __name__ == '__main__':
    main()
