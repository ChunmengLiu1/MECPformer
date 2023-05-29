# MECPformer
Official Implementation of the paper: [MECPformer: Multi-estimations Complementary Patch with CNN-Transformers for Weakly Supervised Semantic Segmentation](https://arxiv.org/pdf/2303.10689.pdf)
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

#### Download pretrained model
Download Conformer-S pretrained weights from [link](https://drive.google.com/file/d/1qjLDy8MYU_TV2hspyYNCXeWrWho360qa/view?usp=share_link)

#### Download saliency map
Download saliency map from [link](https://drive.google.com/file/d/1n7hVi8U2ylBMjz_bECsl_wSAmlRqnVr8/view?usp=share_link)

## 
