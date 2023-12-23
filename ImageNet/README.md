# Pruning ImageNet with Dynamic Uncertainty

The code of ["Large-scale Dataset Pruning with Dynamic Uncertainty"](https://arxiv.org/abs/2306.05175). 

We propose a simple yet effective dataset pruning method by exploring both the prediction uncertainty and training dynamics.  Our method outperforms the state of the art and achieves 75% lossless compression ratio on both ImageNet-1K and ImageNet-21K. 



## 1. Install

* CUDA and cuDNN

  We use CUDA 11.8 and cuDNN 8.7.0. We actually use the CUDA docker by NVIDIA: `docker pull nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04`

* Create a conda virtual environment and activate it:

  ```
  conda create -n dynunc python=3.10
  conda activate dynunc
  ```

* requirements

  ```
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  pip install timm
  pip install nvidia-dali-cuda110
  pip install yacs termcolor
  ```

* Install fused window process for acceleration for Swin Transformer, activated by editing the `ACCELERATION-FUSED_WINDOW_PROCESS` item in the config file

  ```
  cd kernels/window_process
  python setup.py install #--user
  ```

* Install apex for fused optimizers and fused layernorm

  Fused layernorm is for acceleration for ViT and Swin Transformer, activated by editing the `ACCELERATION-FUSED_LAYERNORM` item in the config file
  
  ```
  # https://github.com/NVIDIA/apex#from-source
  pip install ninja
  git clone https://github.com/NVIDIA/apex
  cd apex
  # if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
  # otherwise
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  ```
  



## 2. Data preparation

* We support ImageNet-1K and ImageNet-21K. 
  * In respect of the latter, both Fall11 and Winter21 release are supported. 
  * We also support the processed version of ImageNet-21K_Fall11 or ImageNet-21K_Winter21 using the script of ["ImageNet-21K Pretraining for the Masses"](https://arxiv.org/abs/2104.10972).
  * For ImageNet-1K, ImageNet-21K_Fall11 and ImageNet-21K_Winter21, we use ImageNet-1K-val and [ImageNet-1K-ReaL](https://github.com/google-research/reassessed-imagenet) as the validation set. `real.json` is needed for using ReaL.
  * ImageNet-21K-Processed_Fall11 or ImageNet-21K-Processed_Winter21 has been split into training set and validation set.

* You can download the datasets from Internet (https://www.image-net.org) and apply necessary processing on them.

- The file structure should look like:

  ```
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
  ```

* For ImageNet-21K_Fall11 and ImageNet-21K_Winter21, we use ImageNet-1K val (and ReaL) as the validation set. You should make an empty directory named as `n04399382` in the training set folder and create a mapping dictionary to select 1000 classes of ImageNet-1K val out of the ImageNet-21K classes.

  ```python
  import os
  import numpy as np
  
  release = 'Fall11'  # or Winter21
  path_21k = 'ImageNet-21K_Fall11'
  path_1k_val = 'ImageNet-1K/val'
  
  os.makedirs(f'{path_21k}/n04399382', exist_ok=True)
  class_21k = list(filter(lambda x: x.startswith('n'), os.listdir(path_21k)))
  class_21k.sort()
  
  if release == 'Fall11':
      assert len(class_21k) == 21842
  elif release == 'Winter21':
      assert len(class_21k) == 19168
  
  class_1k_val = list(filter(lambda x: x.startswith('n'), os.listdir(path_1k_val)))
  class_1k_val.sort()
  assert len(class_1k_val) == 1000
  
  mapping_dict = []
  ptr = 0
  for class_name in class_1k_val:
      while True:
          if class_21k[ptr] == class_name:
              mapping_dict.append(ptr)
              break
          ptr += 1
  
  np.savetxt(f'{path_21k}/mapping_dict.txt', np.asarray(mapping_dict))
  ```

* Due to that NVIDIA DALI is used for accelerating data loading and pre-processing, you should create a `file_list.txt` in training and validation set folder containing the relative path and label of all data. Noted that the labels are assigned to classes in the lexicographical order.

  ```python
  import os
  
  path = 'ImageNet-1K/train'  # or ImageNet-1K/val
  # ImageNet-21K_Fal11, ImageNet-21K_Winter21 
  # ImageNet-21K-Processed_Fall11/imagenet21k_train, ImageNet-21K-Processed_Fall11/imagenet21k_val,
  # ImageNet-21K-Processed_Winter21/imagenet21k_train, ImageNet-21K-Processed_Winter21/imagenet21k_val
  
  
  classes = list(filter(lambda x: x.startswith('n'), os.listdir(path)))
  classes.sort()
  
  with open(f'{path}/file_list.txt', 'w') as f:
      for label in range(len(classes)):
          class_dir = f'{path}/{classes[label]}'
          class_images = os.listdir(class_dir)
          class_images.sort()
          for image_name in class_images:
              f.write(f'/{classes[label]}/{image_name} {label}\n')
  ```



## 3. Evaluate coreset

*Remember to modify the data paths in the config file.*

We supply the 75% coreset (25% pruned) of ImageNet-1K and ImageNet-21K_Fall11, namely `IN1K_75_file_list.txt` and `IN21KFall_75_file_list.txt` ([download link](https://huggingface.co/datasets/Isaachhe/Dataset-Pruning/tree/main/ImageNet)). You can move them to the correspond training set folder.

* evaluate the 75% coreset of ImageNet-1K on Swin-T

  ```
  torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=8 main.py --config ./configs/swin-tiny-1k.yaml --coreset ImageNet-1K/train/IN1K_75_file_list.txt --output output-swint-1kdynunc75
  ```
  
* evaluate the 75% coreset of ImageNet-21K_Fall11 on Swin-T

  first pre-train on 21K coreset

  ```
  torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=8 main.py --config ./configs/swin-tiny-21k.yaml --coreset ImageNet-21K_Fall11/IN21KFall_75_file_list.txt --output output-swint-21k-dynunc75
  ```
  
  then fine-tune on 1K

  ```
  torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=8 main.py --config ./configs/swin-tiny-21kto1k.yaml --pretrain output-swint-21k-dynunc75/ckpts/ckpt_epoch_89.pth --output output-swint-21kto1k-dynunc75
  ```
  
  

## 4. Reproduce coreset

We supply a series of template config file in `configs` folder. You can choose config of models and datasets that interest you. *Remember to modify the data paths in the config file.*

For example, produce 75% coreset of ImageNet-1K on Swin-T.

```
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=8 main_and_produce.py --config ./configs/swin-tiny-1k.yaml --output output-swint-produce

python get_coreset.py --input output-swint-produce --dataset_file_list ImageNet-1K/file_list.txt --fraction 0.75 --window 10 --output ImageNet-1K/train/IN1K_75_file_list_own.txt
```