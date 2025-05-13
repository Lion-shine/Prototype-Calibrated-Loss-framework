# Prototype-Calibrated-Loss-framework

This repository contains the pytorch code for the paper:

Prototype-Calibrated Loss for Multi-Class Cell Detection in Histological Images with Sparse Annotations

## Introduction
In this paper, we propose a novel training method for robust multi-class cell detection in histological images using sparsely annotated datasets. Traditional fully supervised methods require exhaustive cell annotations, which are impractical given the immense density and diversity of cells in annotated images. To address this limitation, our method leverages sparse annotations by introducing a prototype-based loss calibration framework that corrects the misclassiﬁcation of unannotated cells. Speciﬁcally, we compute a semantic energy (SE) that quantiﬁes the similarity between annotated cell features and those in unannotated regions, thereby weighting the loss to reduce incorrect supervision. To further enhance inter-class discriminability, we incorporate a Stretch Feature Loss (SFL) and an Exclusive Loss (EL), which respectively expand the subtle feature differences between classes and penalize low-conﬁdence predictions. Extensive experiments conducted on three datasets demonstrate that our approach not only outperforms state-of-the-art methods designed for sparse annotations but also surpasses fully supervised baselines across all annotation rates. Our method consistently achieves better performance for multiple cell classes, even when trained with as little as 10% of the annotations. These results indicate that our technique effectively mitigates the challenges associated with sparse annotations and holds signiﬁcant potential for reducing the annotation burden in clinical practice.
![](img/method.png)



## Dependecies
In the environment configuration, we primarily utilize `torch==1.7.1`.

## Usage

### Data preparation
Before training, you need to prepare the training images and ground truth. 
Please see 'data_preprocess_heatmap.py' for more information


### Model training and test
To training a model, set related parameters and run `python train.py`

To evaluate the trained model on the test set, set related parameters in the file `caculate_metric.py` and run `python test.py`. 

