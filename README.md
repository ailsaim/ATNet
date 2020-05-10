# Attention Transfer Network For Nature Image Matting 
This is a keras implementation of ATNet.

## Abstract
Natural image matting is an important problem that widely applied in computer vision and graphics. Existing deep learning matting approaches have made an impressive process in both accuracy and efficiency. However, there are still two fundamental problems remain largely unsolved: 1) accurately separating an object from the image with similar foreground and background color or lots of details; 2) exactly extracting an object with fine structures from complex background. In this paper, we propose an attention transfer network (ATNet) to overcome these challenges. Specifically, we firstly design a feature attention block to effectively distinguish the foreground object from the color-similar regions by activating foreground-related features as well as suppressing others. Then, we introduce a scale transfer block to magnify the feature maps without adding extra information. By doing so, we effectively reduce the artificial content in results and decrease the computational complexity. Besides, we use a perceptual loss to measure the difference between the feature representations of the predictions and the ground-truths. It conducive to further capture the high-frequency details of the image, and consequently, optimize the fine structures of the object. Extensive experiments on two publicly common datasets (i.e., Composition-1k matting dataset, and alphamatting.com dataset) show that the proposed ATNet obtains significant improvements over the previous methods.

Datasets:
1) Adobe Image Matting dataset: Follow the [instruction](https://sites.google.com/view/deepimagematting) to contact the author of "Deep Image Matting" for the dataset.

Test:
$python test_adobe_data.py
