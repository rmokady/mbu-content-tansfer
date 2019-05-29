# Mask Based Unsupervised Content Transfer

PyTorch implementation of "Mask Based Unsupervised Content Transfer"

Given two domains where one contains some additional information compared to the other, our method disentangles the common and the seperate parts and transfers the seperate information from one image to another using a mask, while not using any supervision at train time. For example, we can transfer facial hair from an image of a men with a mustache to an image of a shaved person, see image below. Using a mask enables state-of-the-art quality, but also, the generated mask can be used as a semantic segmentation of the seperate part. Thus our method perform weakly-supervised semantic segmentation, using only class lables as supervision, achieving state-of-the-art performance.

For more details, please refer to the full article.

The first example presents facial hair transfer, where the facial hair of the men in the left column transferred to the shaved men in the upper row. The second example illustrate semantic segmentation of glasses and facial hair:

<img src="images/mustache_grid.png" width="500px">
The second example illustrate semantic segmentation of glasses and facial hair:
<img src="images/mu_gl_segmentation (1).png" width="500px">

## Prerequisites:
Python 2.7 / 3.6, Pytorch 0.4, argparse, Pillow

## Download and Prepare the Data:

Download [celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) directly or by running the following command:
```
bash celeba_downloader.sh
```
Contrary to the notation used in the paper, A is the larger set, for example, A is people with glasses and B is people without.
You can use the provided script preprocess.py to split celebA into the above format (with A and B based on the attribute of your choosing).
For example, you can run the script using the following command:
```
python preprocess.py --root ./img_align_celeba --attributes ./list_attr_celeba.txt --dest ./beard_data --config beard
```
## Training
The Hyperparameters names are different then the article, the mapping between the codes name and the loss functions is:
- alpha1 - <img src="https://latex.codecogs.com/gif.latex?\mathcal{L}_{Recon2}^B" />
- alpha2 - <img src="https://latex.codecogs.com/gif.latex?\mathcal{L}_{Recon2}^A" />
- beta1 - <img src="https://latex.codecogs.com/gif.latex?\mathcal{L}_{Cycle}" /> (first term)
- beta2 - <img src="https://latex.codecogs.com/gif.latex?\mathcal{L}_{Cycle}" /> (second term) 
- delta - <img src="https://latex.codecogs.com/gif.latex?\mathcal{L}_{Recon1}^A" />
- gama -  <img src="https://latex.codecogs.com/gif.latex?\mathcal{L}_{Recon1}^B" />
- discweight - <img src="https://latex.codecogs.com/gif.latex?\mathcal{L}_{DC}" />


Train using mask_train.py. You can use the following example to run with default hyperparameters (for beard):
```
python mask_train.py --root ./beard_data --out ./beard_experiment
```
For choosing GPU use
```
--gpu 3
```
You can use the following example to resume training:
```
python mask_train.py --root ./beard_data --out ./beard_experiment --load ./beard_experiment
```
## Evaluation
You can evalute with the command:
```
python mask_eval.py --root ./beard_data --out ./beard_experiment
```
Note that for evluate the segmentation, one should perform thresholding on the generated mask (search for the best threshold).

## Hyperparameters:

Hyperparameters for beard are the defualt.

- Hyperparameters for glasses (male and female):
```
--disclr 0.0002 --lr 0.0002 --gama 7.0 --delta 5.0 --beta1 0.0 --beta2 0.001 --alpha1 0.7 --alpha2 0.7
```
- Hyperparameters for smile (male and female):
```
--beta2 0.0001 --beta1 0.0 --alpha 1.0 --discweight 0.005  --alpha1 1.0 --alpha2 1.0 --gama 7.0 --delta 5.0
```
- Command line for handbags content transfer: (Augmentation should be performed as the center of the image is different between the domains)
```
python mask_train.py --root ./bags/ --out ./bags0/ --alpha1 0.5 --alpha2 0.5 --beta1 0.0 --beta2 2.5 --delta 7.0 --gama 5.0 --discweight 0.1
```

- Hyperparameters for women hair segmentation ("black to blond content transfer"):
```
--alpha1 2.0 --alpha2 2.0 --beta1 0.01 --beta2 7.0 --gama 10.0 --delta 5.0 --discweight 1.0
```

- Hyperparameters for men hair segmentation ("black to blond content transfer"):
```
--alpha1 1.0 --alpha2 1.0 --beta2 1.0 --beta1 0.0 --gama is 5.0 --delta is 5.0 --discweight is 0.25
```
## Custom Dataset
First, You can then run the preprocessing in the following manner:
```
python preprocess.py --root ./custom_dataset --dest ./custom_train --folders
```
or use the following format for the images:
```
root/
     trainA/
     trainB/
     testA/
     testB/
```
Second, search for the best hyperparameters, as the default hyperparameters might not suit any dataset.

## Acknowledgements
This implementation is heavily based on https://github.com/oripress/ContentDisentanglement.

## Citation
If you found this work useful, please cite.
