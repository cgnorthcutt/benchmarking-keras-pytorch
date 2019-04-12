#!/usr/bin/env python
# coding: utf-8

# In[2]:


# These imports enhance Python2/3 compatibility.
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


# In[3]:


# General imports
import argparse
import numpy as np
import os
import sys
from PIL import Image

# Use PyTorch/torchvision for dataloading (more reliable/faster)
from torchvision import datasets


# In[1]:


# Keras modules
from keras.preprocessing import image

# Keras models
from keras.applications import (
    DenseNet121,
    DenseNet169,
    DenseNet201,
    InceptionResNetV2,
    InceptionV3,
    MobileNet,
    MobileNetV2,
    NASNetLarge,
    NASNetMobile,
    ResNet50,
    VGG16,
    VGG19,
    Xception,
)

# Import preprocess_inputs parent modules
from keras.applications import (
    densenet,
    inception_resnet_v2,
    inception_v3,
    mobilenet,
    mobilenet_v2,
    nasnet,
    resnet50,
    vgg16,
    vgg19,
    xception,
)

# # Helps with compatibilty with CUDA version 10 and the RTX 2080 GPU line. Uncomment if relevant.
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# gpu_options = tf.GPUOptions2(allow_growth=True)
# config = tf.ConfigProto(gpu_options=gpu_options)
# set_session(tf.Session(config=config))


# In[ ]:


keras_models = {
    "densenet121" : DenseNet121,
    "densenet169" : DenseNet169,
    "densenet201" : DenseNet201,
    "mobilenet" : MobileNet,
    "mobilenetV2" : MobileNetV2,
    "nasnetmobile" : NASNetMobile,
    "resnet50" : ResNet50,
    "vgg16" : VGG16,
    "vgg19" : VGG19,
    "xception" : Xception,
    "inceptionresnetv2" : InceptionResNetV2,
    "inceptionv3" : InceptionV3,
    "nasnetlarge" : NASNetLarge,
}

models_preprocessing = {
    "densenet121" : densenet.preprocess_input,
    "densenet169" : densenet.preprocess_input,
    "densenet201" : densenet.preprocess_input,
    "mobilenet" : mobilenet.preprocess_input,
    "mobilenetV2" : mobilenet_v2.preprocess_input,
    "nasnetmobile" : nasnet.preprocess_input,
    "resnet50" : resnet50.preprocess_input,
    "vgg16" : vgg16.preprocess_input,
    "vgg19" : vgg19.preprocess_input,
    "xception" : xception.preprocess_input,
    "inceptionresnetv2" : inception_resnet_v2.preprocess_input,
    "inceptionv3" : inception_v3.preprocess_input,
    "nasnetlarge" : nasnet.preprocess_input,
}

models_img_size = {
    "densenet121" : (224, 224),
    "densenet169" : (224, 224),
    "densenet201" : (224, 224),
    "mobilenet" : (224, 224),
    "mobilenetV2" : (224, 224),
    "nasnetmobile" : (224, 224),
    "resnet50" : (224, 224),
    "vgg16" : (224, 224),
    "vgg19" : (224, 224),
    "xception" : (299, 299),
    "inceptionresnetv2" : (299, 299),
    "inceptionv3" : (299, 299),
    "nasnetlarge" : (331, 331),
}


# In[ ]:


# Set up argument parser
parser = argparse.ArgumentParser(description='Keras ImageNet Inference')
parser.add_argument('val_dir', metavar='DIR',
                    help='path to imagenet val dataset folder')
parser.add_argument('-m', '--model', metavar='MODEL', default=None,
                    choices=keras_models.keys(),
                    help='model architecture: ' +
                        ' | '.join(keras_models.keys()) +
                        ' (example: resnet50)' +
                        ' (default: Runs across all Keras models)')
parser.add_argument('-g', '--gpu', metavar='MODEL', default=0,
                    help='int of GPU to use. Only uses single GPU.')
parser.add_argument('-o', '--output-dir', metavar='OUTPUT_DIR', default="keras_imagenet/",
                    help='directory folder to store output results in.')
parser.add_argument('--save-all-probs',  action='store_true', default = False, 
                    help='Store entire softmax output for all examples (100 MB)')


# In[ ]:


def main(args = parser.parse_args()):
    '''Select GPU and set up data loader for ImageNet val set.'''
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    
    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Grab imagenet data
    val_dataset = datasets.ImageFolder(args.val_dir)
    img_paths, labels = (list(t) for t in zip(*val_dataset.imgs))

    # Run forward pass inference on all models for all examples in val set.
    models = keras_models if args.model is None else [args.model]
    for model in models:
        process_model(model, img_paths, labels, args.output_dir, args.save_all_probs)


# In[6]:


# def images2data(img_paths, img_size = (224, 224)):
#     result = []
#     for i, img_path in enumerate(img_paths):
#         if i % 50 == 0:
#             print("\rComplete: {:.1%}".format(i / len(img_paths)), end = "")
#         img = image.load_img(img_path, target_size=img_size)
#         result.append(np.expand_dims(image.img_to_array(img), axis=0))
#     print()
#     return result


# In[9]:

def crop_center(img, target_size):
    # target_size is assumed to be in network's order (H, W)
    w, h = img.size
    cx = w // 2
    cy = h // 2
    left = cx - target_size[1] // 2
    top = cy - target_size[0] // 2
    return img.crop((left, top, left + target_size[1], top + target_size[0]))


def shortest_edge_scale(img, target_size, scale):
    # target_size is assumed to be in network's order (H, W)
    w, h = img.size
    nw = int(w * target_size[1] / scale) // min((w, h))
    nh = int(h * target_size[0] / scale) // min((w, h))
    return img.resize((nw, nh), resample=Image.BILINEAR)


def process_model(
    model_name, 
    img_paths, 
    labels, 
    out_dir="keras_imagenet/",
    save_all_probs = False,
):
    '''Actual work is done here. This runs inference on a Keras model,
    by computing the output of a forward pass, individually for each example.
    Running examples in batches or using vectorized operations results in
    random outputs and lack of reproducibility in Keras (this is a bug).
    This method will avoid those issues by running the forward pass on each
    example, one at a time. This is slower, but accurate.
    
    Top5 predictions and probabilities for each example for the model
    are stored in the keras_imagenet/ output directory.'''
    
    preprocess_model = models_preprocessing[model_name]  
    img_size = models_img_size[model_name]
    Model = keras_models[model_name] 
    wfn_base = os.path.join(out_dir, model_name + "_keras_imagenet_")

    # Create Keras model
    model = Model(weights='imagenet')

    # Preprocessing and Forward pass through validation set.
    probs = []
    inputs = []
    batch_size = 64
    for i, img_path in enumerate(img_paths):
        img = image.load_img(img_path, target_size=None)
        img = shortest_edge_scale(img, img_size, 0.875)
        img = crop_center(img, img_size)
        img = np.expand_dims(image.img_to_array(img), axis=0)
        inputs.append(img)

        current_batch = 0
        if i % (batch_size + 1) == 0:
            current_batch = batch_size
        elif i == len(img_paths) - 1:
            current_batch = len(img_paths) % batch_size

        if current_batch:
            inputs = np.concatenate(inputs, axis=0)
            probs.append(model.predict_on_batch(preprocess_model(inputs)))
            print("\r{} completed: {:.2%}".format(model_name, i / len(img_paths)), end="")
            sys.stdout.flush()
            inputs = []

    probs = np.vstack(probs)
    if save_all_probs:
        np.save(wfn_base + "probs.npy", probs.astype(np.float16))
    
    # Extract top 5 predictions for each example
    n = 5
    top = np.argpartition(-probs, n, axis = 1)[:,:n]
    top_probs = probs[np.arange(probs.shape[0])[:, None], top]
    acc1 = sum(top[range(len(top)), np.argmax(top_probs, axis = 1)] == labels) / float(len(labels))
    acc5 = sum([labels[i] in row for i, row in enumerate(top)]) / float(len(labels))
    print('\n{}: acc1: {:.2%}, acc5: {:.2%}'.format(model_name, acc1, acc5))
    
    # Save top 5 predictions and associated probabilities
    np.save(wfn_base + "top5preds.npy", top)
    np.save(wfn_base + "top5probs.npy", top_probs.astype(np.float16))


# In[ ]:


if __name__ == '__main__':
    main()

