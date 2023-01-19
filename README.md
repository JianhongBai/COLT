# On the Effectiveness of Out-of-Distribution Data in Self-Supervised Long-Tail Learning.

ICLR 2023: This repository is the official implementation of [COLT]().

## Introduction
Though Self-supervised learning (SSL) has been widely studied as a promising technique for representation learning, it doesn’t generalize well on long-tailed datasets due to the majority classes dominating the feature space. Recent work shows that the long-tailed learning performance could be boosted by sampling extra in-domain (ID) data for self-supervised training, however, large-scale ID data which can rebalance the minority classes are expensive to collect. 
To this end, we propose an alternative but easy-to-use and effective solution, Contrastive with Out-of-distribution (OOD) data for Long-Tail learning (COLT), which can effectively exploit OOD data to dynamically re-balance the feature space. We empirically identify the counter-intuitive usefulness of OOD samples in SSL long-tailed
learning and principally design a novel SSL method. Concretely, we first localize the ‘head’ and ‘tail’ samples by assigning a tailness score to each OOD sample based on its neighborhoods in the feature space. Then, we propose an online OOD sampling strategy to dynamically re-balance the feature space. Finally, we enforce the model to be capable of distinguishing ID and OOD samples by a distribution-level supervised contrastive loss. Extensive experiments are conducted on various datasets and several state-of-the-art SSL frameworks to verify the effectiveness of the proposed method. The results show that our method significantly improves the performance of SSL on long-tailed datasets by a large margin, and even outperforms previous work which uses external ID data.

## Method
<div align=center>
<img src="pipeline.png" width="800" >
</div>

Overview of Contrastive with Out-of-distribution data for Long-Tail learning (COLT). COLT can be easily plugged into most SSL frameworks. Proposed components are denoted as red.

## Environment
Requirements:
```bash
pytorch 1.7.1 
opencv-python
scikit-learn 
matplotlib
```

## Datasets
You can download 300K Random Images datasets in the following url:

[300K Random Images](https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy)

## Pretrained models downloading
[CIFAR-10]()

[CIFAR-100]()

[ImageNet-100]()

[Places-365]()

## Train and evaluate pretrained models
### CIFAR-10
SimCLR on long-tail training datasets
```
# pre-train and finetune
for split_num in 1 2 3 4 5
do
./cmds/shell_scrips/cifar-10-LT_extra.sh -g 1 -p 4867 -w 8 --split split${split_num}_D_i
done
```

SimCLR+COLT on long tail training datasets
```
# pre-train and finetune
for split_num in 1 2 3 4 5
do
./cmds/shell_scrips/cifar-10-LT_extra.sh -g 1 -p 4867 -w 8 --split split${split_num}_D_i --save_dir COLT --COLT True
done
```

### CIFAR-100
SimCLR on long-tail training datasets
```
# pre-train and finetune
for split_num in 1 2 3 4 5
do
./cmds/shell_scrips/cifar-100-LT_extra.sh -g 1 -p 4867 -w 8 --split cifar100_split${split_num}_D_i
done
```

SimCLR+COLT on long tail training datasets
```
# pre-train and finetune
for split_num in 1 2 3 4 5
do
./cmds/shell_scrips/cifar-100-LT_extra.sh -g 1 -p 4867 -w 8 --split cifar100_split${split_num}_D_i --save_dir COLT --COLT True
done
```

### ImageNet-100
SimCLR on long-tail training datasets
```
# pre-train and finetune
./cmds/shell_scrips/imagenet-100-res50-LT_extra.sh --data \path\to\imagenet -g 2 -p 4867 -w 10 --split imageNet_100_LT_train
```

SimCLR+COLT on long tail training datasets
```
# pre-train and finetune
./cmds/shell_scrips/imagenet-100-res50-LT_extra.sh --data \path\to\imagenet -g 2 -p 4867 -w 10 --split imageNet_100_LT_train --save_dir COLT --COLT True
```

### Places-365
SimCLR on long-tail training datasets
```
# pre-train and finetune
./cmds/shell_scrips/places365-LT_extra.sh --data \path\to\places -g 2 -p 4867 -w 10 --split Places_LT_train
```

SimCLR+COLT on long tail training datasets
```
# pre-train and finetune
./cmds/shell_scrips/places365-LT_extra.sh --data \path\to\places -g 2 -p 4867 -w 10 --split Places_LT_train --save_dir COLT --COLT True
```


