#!/usr/bin/env python
# coding: utf-8

# In[1]:


# These imports enhance Python2/3 compatibility.
from __future__ import (
    print_function, absolute_import, division, unicode_literals, with_statement
)


# In[13]:


from torchvision import models, datasets, transforms
import torch
import multiprocessing
import numpy as np
import os
import sys
import argparse

# This helps with dataloading for inference
torch.multiprocessing.set_sharing_strategy('file_system')
VAL_SIZE = 50000


# In[3]:


pytorch_models = [
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
]


# In[22]:


# Set up argument parser
parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('val_dir', metavar='DIR',
                    help='path to imagenet val dataset folder')
parser.add_argument('-m', '--model', metavar='MODEL', default=None,
                    choices=pytorch_models,
                    help='model architecture: ' +
                        ' | '.join(pytorch_models) +
                        ' (example: resnet50)' +
                        ' (default: Runs across all PyTorch models)')
parser.add_argument('-g', '--gpu', metavar='MODEL', default=0,
                    help='int of GPU to use. Only uses single GPU.')
parser.add_argument('-b', '--batch-size', metavar='BATCHSIZE', default=32,
                    help='Number of examples to run forward pass on at once.')
parser.add_argument('-o', '--output-dir', metavar='OUTPUTDIR',
                    default="pytorch_imagenet/",
                    help='directory folder to store output results in.')
parser.add_argument('--save-all-probs',  action='store_true', default = False, 
                    help='Store entire softmax output of all examples (100 MB)')
parser.add_argument('--save-labels', action='store_true', default = False, 
                    help='Store labels')


# In[23]:


def main(args = parser.parse_args()):
    '''Select GPU and set up data loader for ImageNet val set.'''
    global device
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    dataloaders = {}
    for img_size in [224, 299]:
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_dataset = datasets.ImageFolder(args.val_dir, val_transform)
        dataloaders[img_size] = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=args.batch_size,                                             
            shuffle=False, 
            num_workers=max(1, multiprocessing.cpu_count() - 2),
        )
        
    # Run forward pass inference on all models for all examples in val set.
    models = pytorch_models if args.model is None else [args.model]
    for model in models:
        process_model(model, dataloaders, args.output_dir,
                      args.save_all_probs, args.save_labels)


# In[ ]:


def process_model(
    model_name,
    dataloaders,
    out_dir = "pytorch_imagenet/",
    save_all_probs = False,
    save_labels = False,
):
    '''Actual work is done here. This runs inference on a pyTorch model,
    using the pyTorch batch loader.
    
    Top5 predictions and probabilities for each example for the model
    are stored in the pytorch_imagenet/ output directory.'''
    
    # Load PyTorch model pre-trained on ImageNet
    model = eval("models.{}(pretrained=True)".format(model_name))
    # Send the model to GPU/CPU
    model = model.to(device)
    wfn_base = os.path.join(out_dir, model_name + "_pytorch_imagenet_")
    probs, labels = [], []
    if model_name is "inception_v3":
        loader = dataloaders[299]
    else:
        loader = dataloaders[224]
    
    # Inference, with no gradient changing
    model.eval() # set model to inference mode (not train mode)
    with torch.set_grad_enabled(False):
        for i, (x_val, y_val) in enumerate(loader):
            print("\r{} completed: {:.2%}".format(
                model_name, i / len(loader)), end="")
            sys.stdout.flush()
            out = torch.nn.functional.softmax(model(x_val.to(device)), dim=1)
            probs.append(out.numpy() if device == 'cpu' else out.cpu().numpy())
            labels.append(y_val)
            
    # Convert batches to single numpy arrays    
    probs = np.stack([p for l in probs for p in l])
    labels = np.array([t for l in labels for t in l])
    if save_labels:
        np.save(wfn_base + "labels.npy", labels.astype(int))
    if save_all_probs:
        np.save(wfn_base + "probs.npy", probs.astype(np.float16))
    
    # Extract top 5 predictions for each example
    n = 5
    top = np.argpartition(-probs, n, axis = 1)[:,:n]
    top_probs = probs[np.arange(probs.shape[0])[:, None], top]
    right1 = sum(top[range(len(top)), np.argmax(top_probs, axis = 1)] == labels)
    acc1 = right1 / float(len(labels))
    count5 = sum([labels[i] in row for i, row in enumerate(top)])
    acc5 = count5 / float(len(labels))
    print('\n{}: acc1: {:.2%}, acc5: {:.2%}'.format(model_name, acc1, acc5))
    
    # Save top 5 predictions and associated probabilities
    np.save(wfn_base + "top5preds.npy", top)
    np.save(wfn_base + "top5probs.npy", top_probs.astype(np.float16))


# In[ ]:


if __name__ == '__main__':
    main()

