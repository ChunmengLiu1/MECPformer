# MECPformer
Official Implementation of the paper: [MECPformer: Multi-estimations Complementary Patch with CNN-Transformers for Weakly Supervised Semantic Segmentation](https://arxiv.org/pdf/2303.10689.pdf).

submitted to Neural Computing and Applications
<p align="left"><img src="imgs/framework.png" alt="outline" width="95%"></p>

## Abstract
The initial seed based on the convolutional neural network (CNN) for weakly supervised semantic segmentation always highlights the most
discriminative regions but fails to identify the global target information. Methods based on transformers have been proposed successively
benefiting from the advantage of capturing long-range feature representations. However, we observe a flaw regardless of the gifts based on
the transformer. Given a class, the initial seeds generated based on the transformer may invade regions belonging to other classes. Inspired
by the mentioned issues, we devise a simple yet effective method with Multi-estimations Complementary Patch (MECP) strategy and Adaptive
Conflict Module (ACM), dubbed MECPformer. Given an image, we manipulate it with the MECP strategy at different epochs, and the network mines and deeply fuses the semantic information at different levels. In addition, ACM adaptively removes conflicting pixels and exploits the network self-training capability to mine potential target information. Without bells and whistles, our MECPformer has reached new state-of-the-art 72.0% mIoU on the PASCAL VOC 2012 and 42.4% on MS COCO 2014 dataset. 

## Prerequisite

#### 1. install dependencies 
Ubuntu 20.04, with Python 3.6 and the following python dependencies.

```pip install -r requirements.txt```

#### 2. Download dataset 
Download [the PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012).

#### 3. Download pretrained model
Download [Conformer-S pretrained weights](https://drive.google.com/file/d/1qjLDy8MYU_TV2hspyYNCXeWrWho360qa/view?usp=share_link).

#### 4. Download saliency map
Download [saliency map](https://drive.google.com/file/d/1n7hVi8U2ylBMjz_bECsl_wSAmlRqnVr8/view?usp=share_link).

## Usage

#### run the run.sh script for training MECPformer, visualizing and evaluating the generated class-specific initial seeds. 

```bash run.sh```

## Trained model on PASCAL VOC 2012 dataset.
| Stage       | Backbone   | Google drive | Train mIoU (%) |
|--------------|------------|--------------|--------------|
| Initial seeds | Conformer-S | [Weights](https://drive.google.com/file/d/1pYWyNjPTLVrZ5nlvipmgjRa_bDP6qSRE/view?usp=share_link)  | 66.6 |

| Stage       | Backbone   | Google drive |  Val mIoU (%)  |
|--------------|------------|--------------|--------------|
|Final prediction | ResNet101 | [Weights]()  | 72.0 |
