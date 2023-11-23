# Dataset Pruning

In the field of computer vision and multimodal learning, the emerging large models, e.g., vision transformers, CLIP, EVA, SAM, Emu, can achieve various tasks and significantly outperform the traditional neural networks, when large-scale training data, e.g., ImageNet-21K, JFT-300M, LAION-5B is available. 



However, storing large datasets and training on them are expensive and even unaffordable. It is known that large-scale datasets have much redundant and easy samples which contribute little to model training. 



Dataset pruning (or coreset selection) aims to remove those less-informative training samples and remain the informative ones of original dataset, such that models trained on remained subset can achieve comparable performance. 



This repository contains code of pruning large-scale datasets written by BAAI-DCAI, including ImageNet and LAION. Please open the corresponding folder for more information.



If you want the condensed ImageNet1K/21K and LAION2B, feel free to contact us: zhaobo@baai.ac.cn .
