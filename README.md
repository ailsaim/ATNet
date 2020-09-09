# Attention Transfer Network For Nature Image Matting 
This is a keras implementation of ATNet.

## Abstract
Natural image matting is an important problem that widely applied in computer vision and graphics. Recent deep learning matting approaches have made an impressive process in both accuracy and efficiency. However, there are still two fundamental problems remain largely unsolved: 1) accurately separating an object from the image with similar foreground and background color or lots of details; 2) exactly extracting an object with fine structures from complex background. In this paper, we propose an attention transfer network (ATNet) to overcome these challenges. Specifically, we firstly design a feature attention block to effectively distinguish the foreground object from the color-similar regions by activating foreground-related features as well as suppressing others. Then, we introduce a scale transfer block to magnify the feature maps without adding extra information. By integrating the above blocks into an attention transfer module, we effectively reduce the artificial content in results and decrease the computational complexity. Besides, we use a perceptual loss to measure the difference between the feature representations of the predictions and the ground-truths. It can further capture the high-frequency details of the image, and consequently, optimize the fine structures of the object. Extensive experiments on two publicly common datasets (i.e., Composition-1k matting dataset, and alphamatting.com dataset) show that the proposed ATNet obtains significant improvements over the previous methods.

## Datasets:
### Adobe Image Matting dataset: 
Follow the [instruction](https://sites.google.com/view/deepimagematting) to contact the author of "Deep Image Matting" for the dataset.
### MSCOCO datasets:
Go to [MSCOCO](http://cocodataset.org/#download) to download: [2014 Train images](http://images.cocodataset.org/zips/train2014.zip).
### PASCAL VOC dataset: 
Go to [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) to download:
1) VOC challenge 2008 [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar).
2) The test data for [the VOC2008 challenge](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/index.html#testdata).

## data preprocessing
### Store training data and test data according to the following path:
The foreground images in training data of the Composition-1k matting dataset: data/fg;

The background images in training data of the Composition-1k matting dataset: data/bg;

The mask images in training data of the Composition-1k matting dataset: data/mask;



The foreground images in test data of the Composition-1k matting dataset: data/fg_test;

The background images in test data of the Composition-1k matting dataset: data/bg_test;

The mask images test data of the Composition-1k matting dataset: data/mask_test;

The combined test dataset: data/merged_test;

## Train model
### Train the encoder-decoder network:
run: $python train_encoder_decoder.py
### Train the refinement network:
run: $python train_refinement.py

## Test model: 
run: $python test_adobe_data.py
