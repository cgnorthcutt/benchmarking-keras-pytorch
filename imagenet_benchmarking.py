
# These imports enhance Python2/3 compatibility.
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement

import numpy as np
import os
import argparse
from torchvision import datasets
import pandas as pd

VAL_SIZE = 50000

models = {
    "keras" : [
        "densenet121",
        "densenet169",
        "densenet201",
        "mobilenet",
        "mobilenetV2",
        "nasnetmobile",
        "resnet50",
        "vgg16",
        "vgg19",
        "xception",
        "inceptionresnetv2",
        "inceptionv3",
        "nasnetlarge",
    ],
    "pytorch" : [
        "inception_v3",
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
        "alexnet",
        "squeezenet1_0",
        "squeezenet1_1",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        ],
}


# Set up argument parser
parser = argparse.ArgumentParser(description='PyTorch and Keras ImageNet Benchmarking')
parser.add_argument('val_dir', metavar='DIR',
                    help='path to imagenet val dataset folder')
parser.add_argument('-k', '--keras-dir', metavar='MODEL', default="keras_imagenet/",
                    help='directory where Keras model outputs are stored.')
parser.add_argument('-p', '--pytorch-dir', metavar='MODEL', default="pytorch_imagenet/",
                    help='directory where PyTorch model outputs are stored.')
parser.add_argument('-o', '--output-dir', metavar='MODEL',
                    default="benchmarking_results.csv",
                    help='csv filename to store the benchmarking csv results.')
parser.add_argument('-i', '--indices_to_omit', metavar='INDICES_TO_OMIT', default=None,
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
    pred = top5preds[range(len(top5preds)), np.argmax(top5probs, axis = 1)]
    pred = pred[bool_mask]
    acc1 = sum(pred == true) / float(len(true))
    acc5 = sum([true[i] in row for i, row in enumerate(top5preds[bool_mask])]) / float(len(true))
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
    
    # Grab imagenet data
    val_dataset = datasets.ImageFolder(args.val_dir)
    img_paths, labels = (list(t) for t in zip(*val_dataset.imgs))
    labels = np.asarray(labels)
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
        custom_labels = np.load(args.custom_labels)
    else:
        custom_labels = None
        
    dirs = {
        "keras" : args.keras_dir,
        "pytorch" : args.pytorch_dir,
    }
            
    data = []
    for platform_name in ["Keras 2.2.4", "PyTorch 1.0"]:
        platform = platform_name.split()[0].lower()
    
        # Read in data and compute accuracies
        pred_suffix = "_{}_imagenet_top5preds.npy".format(platform)
        prob_suffix = "_{}_imagenet_top5probs.npy".format(platform)
        for model in models[platform]:
            top5preds = np.load(os.path.join(dirs[platform], model + pred_suffix))
            top5probs = np.load(os.path.join(dirs[platform], model + prob_suffix))
            if 'nasnet' in model:
                pred = top5preds[range(len(top5preds)), np.argmax(top5probs, axis = 1)]
                np.save('nasnet_correct_mask', pred == labels)
            # This code is just computing accuracy first with as few
            # modifications to the original val set as possible and then again
            # will all modifications.
            if args.custom_labels is None and (args.indices_to_omit or args.indices_to_keep):
                acc1, acc5 = compute_val_acc(top5preds, top5probs, labels)                
                acc1c, acc5c = compute_val_acc(top5preds, top5probs, labels,
                        indices_to_omit, indices_to_keep)
            elif args.custom_labels and (args.indices_to_omit or args.indices_to_keep):
                acc1, acc5 = compute_val_acc(top5preds, top5probs, labels,
                        indices_to_omit, indices_to_keep)                
                acc1c, acc5c = compute_val_acc(top5preds, top5probs, labels,
                        indices_to_omit, indices_to_keep, custom_labels)
            elif args.custom_labels and not (args.indices_to_omit or args.indices_to_keep):
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
        cols = ["Platform", "Model", "Acc@1", "cAcc@1", "Acc@5", "cAcc@5",]
    else:
        cols = ["Platform", "Model", "Acc@1", "Acc@5",]
    df = pd.DataFrame(data, columns=cols)
    # Add rankings for each accuracy mettric
    for col in [c for c in df.columns if 'Acc' in c]:
        df.sort_values(by=col, ascending=False, inplace=True)
        df[col.replace('Acc', 'Rank')] = np.arange(len(df)) + 1
    # Final sort by Acc@1 column
    if args.custom_labels:
        df.sort_values(by="cAcc@1", ascending=False, inplace=True)
    else:
        df.sort_values(by="Acc@1", ascending=False, inplace=True)
    df = df.reset_index(drop=True)
    df.to_csv(args.output_dir, index=False)
    print(df)
    
# Old version of printing and storing results,
#   temporarily kept here for longetivity. Now we use pandas to print the csv.

#     # Order final accuracies
#     data.sort(key=lambda x: x[2], reverse=True)

#     # Compute ranking for each column in data
#     ranks = [list(np.argsort(z)[::-1] + 1) for z in [list(z) for z in zip(*data)][2:]]
#     data = list(zip(*[list(z) for z in zip(*data)] + ranks))
#     print(pd.DataFrame(data))

#     #### Print results
#     if args.indices_to_omit is None and args.indices_to_keep is None:        
#         header_row = ("{:<13}{:<19}{:<8}{:<8}{:<8}{:<8}")
#         print(header_row.format("Platform", "Model", "Acc@1", "Acc@5", "Rank@1", "Rank@5"))
#         print(header_row.format("-----", "-----", "-----", "------", "------", "-------"))
#         data_row = ("{:<13}{:<19}{:<8.2%}{:<8.2%}{:<8}{:<8}")
#     else:
#         header_row = ("{:<13}{:<19}{:<8}{:<8}{:<8}{:<8}{:<8}{:<8}{:<8}{:<8}")
#         print(header_row.format("Platform", "Model", "Acc@1", "cAcc@1", "Acc@5", "cAcc@5", "Rank@1", "cRank@5", "Rank@5", "cRank@5"))
#         print(header_row.format("-----", "-----", "-----", "------", "-----", "------", "------", "-------", "------", "-------"))
#         data_row = ("{:<13}{:<19}{:<8.2%}{:<8.2%}{:<8.2%}{:<8.2%}{:<8}{:<8}{:<8}{:<8}")
        
#     for d in data:
#         print(data_row.format(*d))


if __name__ == '__main__':
    main()


# Example output:

# Platform     Model              Acc@1   Acc@5   Rank@1  Rank@5
# -----        -----              -----   ------  ------  -------
# Keras 2.2.4  nasnetlarge        80.83%  95.27%  1       1
# Keras 2.2.4  inceptionresnetv2  78.93%  94.45%  2       2
# PyTorch 1.0  resnet152          77.62%  93.81%  3       3
# Keras 2.2.4  xception           77.18%  93.49%  4       4
# PyTorch 1.0  densenet161        76.92%  93.49%  5       5
# PyTorch 1.0  resnet101          76.64%  93.30%  6       6
# PyTorch 1.0  densenet201        76.41%  93.18%  7       7
# Keras 2.2.4  inceptionv3        76.02%  92.90%  8       8
# PyTorch 1.0  densenet169        75.59%  92.69%  9       9
# PyTorch 1.0  resnet50           75.06%  92.48%  10      10
# Keras 2.2.4  densenet201        74.77%  92.32%  11      11
# PyTorch 1.0  densenet121        74.07%  91.94%  12      12
# Keras 2.2.4  densenet169        73.92%  91.76%  13      13
# PyTorch 1.0  vgg19_bn           72.90%  91.35%  14      14
# PyTorch 1.0  resnet34           72.34%  90.84%  15      16
# PyTorch 1.0  vgg16_bn           72.29%  91.01%  16      15
# Keras 2.2.4  densenet121        72.09%  90.70%  17      17
# Keras 2.2.4  nasnetmobile       71.59%  90.19%  18      19
# PyTorch 1.0  vgg19              71.19%  90.40%  19      18
# PyTorch 1.0  vgg16              70.66%  89.93%  20      20
# Keras 2.2.4  resnet50           70.35%  89.55%  21      22
# PyTorch 1.0  vgg13_bn           70.12%  89.56%  22      21
# Keras 2.2.4  mobilenetV2        69.98%  89.49%  23      23
# PyTorch 1.0  vgg11_bn           69.36%  89.06%  24      24
# PyTorch 1.0  inception_v3       69.25%  88.69%  25      25
# Keras 2.2.4  mobilenet          69.02%  88.48%  26      27
# PyTorch 1.0  vgg13              68.69%  88.65%  27      28
# PyTorch 1.0  resnet18           68.37%  88.56%  28      26
# PyTorch 1.0  vgg11              67.85%  88.11%  29      29
# Keras 2.2.4  vgg19              65.58%  86.54%  30      30
# Keras 2.2.4  vgg16              65.24%  86.20%  31      31
# PyTorch 1.0  squeezenet1_0      56.49%  79.06%  32      33
# PyTorch 1.0  squeezenet1_1      56.42%  79.21%  33      32
# PyTorch 1.0  alexnet            54.50%  77.64%  34      34

