# Improving Vision Transformers by Overlapping Heads in Multi-Head Self-Attention

## Introduction

Vision Transformers have made remarkable progress in recent years, achieving state-of-the-art performance in most vision tasks. A key component of this success is due to the introduction of the Multi-Head Self-Attention (MHSA) module, which enables each head to learn different representations by applying the attention mechanism independently. In this paper, we empirically demonstrate that Vision Transformers can be further enhanced by overlapping the heads in MHSA. We introduce Multi-Overlapped-Head Self-Attention (MOHSA), where heads are overlapped with their two adjacent heads for queries, keys, and values, while zero-padding is employed for the first and last heads, which have only one neighboring head. Various paradigms for overlapping ratios are proposed to fully investigate the optimal performance of our approach. The proposed approach is evaluated using five Transformer models on four benchmark datasets and yields a significant performance boost. Our paper is available at [link](https://arxiv.org/abs/2410.14874).

## Approach

### Heads Overlap
<div style="color:#0000FF" align="center">
<img src="figures/heads_overlap.pdf"/>
</div>

### Main Architecture
<div style="color:#0000FF" align="center">
<img src="figures/MOHSA.pdf"/>
</div>


## Installation
The implementation of the original Vision Transformers comes from [vit-pytorch](https://github.com/lucidrains/vit-pytorch) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer). For more information about the installation and implementation, please refer to [get_started.md](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md).


## Training
For training the models on the the datasets, for example, we could utilize the following command line to train CaiT model on the dataset CIFAR10:

```
python -m torch.distributed.launch --nproc_per_node=[num of GPUs] --master_port 12345 main.py --cfg configs/cait/cait_xxs24_16_224_cifar10.yaml --data-path [data path to CIFAR10] --batch-size [batch size]
```

Similar implementations could be applied to other models and datasets.


