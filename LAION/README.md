# Pruning LAION-2B: more than SemDeDup

Large vision-language models like CLIP, Stable Diffusion, Flamingo show capabilities of good comprehension of image and text and transfer to various downstream tasks with zero-shot predictions, serving as a backbone architecture of modern computer vision. They learn rich vision-language correlation from large amounts of noisy image-text data from the Internet. 

LAION presents two massive openly accessible image-text datasets, [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) and [LAION-5B](https://laion.ai/blog/laion-5b/), containing 413 million and 5.85 billion image-text pairs respectively.  Derived from Common Crawl, these roughly filted image-text datasets are overall of low quality, such as duplicates and noisy data.

LAION-2B is the English image-text subset of LAION-5B containing 2.32 billion samples.

SemDeDup ([link1](https://openreview.net/forum?id=IRSesTQUtb&referrer=%5BTMLR%5D(%2Fgroup%3Fid%3DTMLR)), [link2](https://arxiv.org/abs/2303.09540)) introduce a method to identify and remove semantic duplicates: data pairs which are semantically similar, but not exactly identical. SemDeDup can remove 50% of LAION-CAT (a filtered subset of LAION-2B containing 438 million samples, introduced by [*Filtering, Distillation, and Hard Negatives for Vision-Language Pre-Training*](https://arxiv.org/abs/2301.02280)) with minimal performance loss, effectively halving training time.



We explore selecting coresets of LAION-2B with less duplicates and more informative samples in this repository. Our work is split to four parts.

In part 0, we introduce necessary preparations and how to evaluate selected coresets.

In part 1, we present a more general method derived from the idea of SemDeDup and illustrate that SemDeDup is a case of our method. 

In part 2, we implement SemDeDup on LAION-2B and show experimental results.

In part 3, we implement the more general method and show some experimental results.





# Part 0

## 0.1 Data preparation

* **ImageNet-1K-val as metric**

  We use the validation set of ImageNet-1K as the evaluating metric.

  You can download the dataset from Internet (https://www.image-net.org) and apply necessary processing on them.

  The file structure should look like:

  ```
  imagenet-val
  ├── class1
      ├── img1.jpeg
      ├── img2.jpeg
      └── ...
  ├── class2
      ├── img3.jpeg
      └── ...
  └── ...
  ```

* **image-text data**

  Following the [instrutions](https://laion.ai/blog/laion-5b/#download-the-data), you should download the metadata of LAION-2B and use [img2dataset](https://github.com/rom1504/img2dataset) to download the actual image-text data (see [tutorial](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion5B.md)).

  We strongly suggest that you keep the `number_sample_per_shard = 10000` config as it is in the [tutorial](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion5B.md). Thus, the file structure should look like:

  ```
  LAION-2B-data
  ├── 000000.tar
  ├── 000001.tar
  ├── 000002.tar
  └── ...
  
  one tar containing up to 10000 samples:
  ├── 012345.tar
      ├── 0123450000.jpg
      ├── 0123450000.txt
      ├── 0123450000.json
      ├── 0123450001.jpg
      ├── 0123450001.txt
      ├── 0123450001.json
      ├── 0123450003.jpg
      ├── 0123450003.txt
      ├── 0123450003.json
      └── ...
  ```

  We use the 10-digit file name as the primary key to identify a sample. The first 6 digits are the shard id, and the last 4 are the "index" in the shard. The "index" may be discontinuous due to url expiration and download failure.
  
  We finally get 232320 tars with 2,053,332,591 samples (one tar containing **up to** 10000 samples).
  
* **embeddings and clip_scores**

  Then, you should use [clip-retrieval](https://github.com/rom1504/clip-retrieval) to compute image and text embeddings of LAION-2B (see [tutorial](https://github.com/rom1504/clip-retrieval/blob/main/docs/distributed_clip_inference.md) and [API](https://github.com/rom1504/clip-retrieval#clip-inference)). **We use `clip_model="ViT-L/14"`**.

  We strongly suggest that you keep the `write_batch_size=1000000` config as it is in the [tutorial](https://github.com/rom1504/clip-retrieval/blob/main/docs/distributed_clip_inference.md).

  Thus, the file structure should look like:
  
    ```
    LAION-2B-embeddings
    ├── img_emb
        ├── img_emb_0000.npy
        ├── img_emb_0001.npy
        ├── img_emb_0002.npy
        └── ...
    ├── text_emb
        ├── text_emb_0000.npy
        ├── text_emb_0001.npy
        ├── text_emb_0002.npy
        └── ...
    ├── metadata
    		├── metadata_0000.parquet
        ├── metadata_0001.parquet
        ├── metadata_0002.parquet
        └── ...
    ```

  For each folder, we finally get 2324 files ($\lceil232320/100\rceil$).

  *Note: LAION supplies pre-computed embeddings in [LAION-5B blog](https://laion.ai/blog/laion-5b/#download-the-data), but it's too hard to identify which sample an embedding belongs to because the primary key (10-digit file name) is different between ours and LAION and we have to use sample's URL to identify. We strongly suggest to compute the embeddings by yourself.*
  
  
  
  Then, simply use the image embeddings and text embeddings to compute the cosine similarity of image-text pair, referred as the clip_score. Noted that inner dot is enough because the embeddings have been normalized to 1 L2-norm ([normalization relevant code](https://github.com/rom1504/clip-retrieval/blob/main/clip_retrieval/clip_inference/mapper.py#L55)).
  
  ```python
  import os
  import torch
  import numpy as np
  
  
  def bdot(a, b):
      B = a.shape[0]
      S = a.shape[1]
      return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)
  
  
  num_emb_files = 
  os.makedirs(f'LAION-2B-embeddings/clip_score', exist_ok=True)
  for i in range(num_emb_files):
      img_emb = torch.from_numpy(np.load(f'LAION-2B-embeddings/img_emb/img_emb_{i:0=4d}.npy')).cuda().double()
      text_emb = torch.from_numpy(np.load(f'LAION-2B-embeddings/text_emb/text_emb_{i:0=4d}.npy')).cuda().double()
      clip_score = bdot(img_emb, text_emb)
      clip_score = clip_score.cpu().numpy()
      np.save(f'LAION-2B-embeddings/clip_score/clip_score_{i:0=4d}.npy', clip_score)
  ```
  
  Finally, the file structure should look like:
  
    ```
    LAION-2B-embeddings
    ├── img_emb
        ├── img_emb_0000.npy
        └── ...
    ├── text_emb
        ├── text_emb_0000.npy
        └── ...
    ├── metadata
        ├── metadata_0000.parquet
        └── ...
    ├── clip_score
        ├── clip_score_0000.npy
        ├── clip_score_0001.npy
        ├── clip_score_0002.npy
        └── ...
    ```



## 0.2 Install

* CUDA and cuDNN

  We use CUDA 11.8 and cuDNN 8.7.0. We actually use the CUDA docker by NVIDIA: `docker pull nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04`
  
* env for de-duplicating

  * Create a conda virtual environment and activate it:

    ```
    conda create -n dedup python=3.10
    conda activate dedup
    ```

  * requirements

    ```
    pip install torch
    # https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
    conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
    pip install webdataset
    pip install pyarrow
    pip install pandas
    ```

* env for evaluating

  * Create a conda virtual environment and activate it:

    ```
    conda create -n eval python=3.10
    conda activate eval
    ```

  * requirements

    ```
    pip install torch torchvision
    pip install deepspeed
    pip install webdataset
    pip install pandas ftfy regex fsspec
    ```



## 0.3 Evaluate coresets

A selected coreset of LAION-2B is organized as a folder with the same amount of  `npy` files as `LAION-2B-data`, each one containing the primary keys of selected samples of the corresponding `tar` file.

For example, the file structure should look like:

 ```
 coreset
  ├── 000000.npy
      containing the primary keys of selected samples
      ├── 0000000145
      ├── 0000002678
      ├── 0000008912
      └── ...
  ├── 000001.npy
      ├── 0000012345
      ├── 0000015869
      ├── 0000017238
      └── ...
  ├── 000002.npy
      ├── 0000023568
      ├── 0000023999
      ├── 0000024001
      └── ...
  └── ...
 ```



There are two methods to use the coreset due to trade-off between time and space.

If the size of coreset is small enough, we suggest to use `retar.py` to select the samples out of original data and re-organize them as webdataset format (tars). Actually we make a partial copy of LAION-2B and more storage is needed without decresing the training speed. (costing 2c hours for c% coreset on a node with 100 cores)

The other way is to insert a filter process in `training/data.py` when the program sees the whole `LAION-2B-data` folder and the coreset folder. When training, the pipeline would traverse all samples in a tar file but only the selected ones would be used. Loading all but using part increases the training time (costing 1.5X for 10% coreset and 4X for 2% coreset) but no more storage is needed.

```python
def group_by_keys_nothrow(data, coreset_path, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    ...
        if coreset_path is not None:
            curr_file_base = os.path.basename(filesample['__url__']).split('.')[0]
            if file_base is None or file_base != curr_file_base:
                file_base = curr_file_base
                coreset = set(np.load(f'{coreset_path}/tar_{file_base}.npy').astype(int))
    ...
                if coreset_path is not None:
                    if int(current_sample['__key__']) in coreset:
                        yield current_sample
    ...
```



We use [OpenCLIP](https://github.com/mlfoundations/open_clip) and [CLIPA](https://github.com/UCSC-VLAA/CLIPA) to evaluate the coresets. Unlike the main setting of SemDeDup keeping the training epoch fixed, we keep the samples seen the same in all conditions to compare the coreset's quality.

* OpenCLIP

  Compared with the [original repo](https://github.com/mlfoundations/open_clip), `training/data.py` and `training/params.py` are modified to insert the filter process mentioned before. `training/distributed.py`, `training/train.py`, `training/params.py` and `training/main.py` are modified to apply [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/) for acceleration. 
  
  For traning OpenCLIP-B/32, edit the path to data and coreset in `sh` file and choose a proper number of workers (4 for us) respect to memory limit and then:
  
  ```
  cd openclip
  
  # single node
  sh run_vit_b_32.sh 4
  
  # resume from a checkpoint, single node; remember to edit the path to checkpoint
  sh run_vit_b_32_resume.sh 4
  
  # multi nodes
  sh run_vit_b_32_multi.sh 4
  ```
  It costs approximately 1 week to run it once on a node with 8 NVIDIA A100 GPUs.

* CLIPA

  Compared with the [original repo](https://github.com/UCSC-VLAA/CLIPA/tree/master/clipa_torch), `training/data.py` and `training/params.py` are modified to insert the filter process mentioned before. And we comment out `import tensorflow` in `open_clip/tokenizer.py` because we use webdataset format rather than TFRecord.

  For traning CLIPA-B/16, edit the path to data and coreset in `sh` file, and then:

  ```
  cd clipa
  
  # single node
  bash b16_pretrain.sh
  ## after pre-train done, edit the path to checkpoint
  bash b16_finetune.sh
  
  # multiple nodes
  bash b16_pretrain_multi.sh
  ## after pre-train done, edit the path to checkpoint
  bash b16_finetune_multi.sh
  ```
It costs approximately 2 days to run it once on a node with 8 NVIDIA A100 GPUs.




# Part 1: semantically de-duplicating

The main idea of SemDeDup is to use image embeddings to identify semantic duplicates. The progress is:

1. cluster all 2B embeddings by k-means
2. in each cluster, build an undirected graph that making an edge for two embeddings if their cosine similarity is above threshold
3. keep only 1 in each weakly connected component of the graph



We believe that above is the key idea of semantically de-duplicating. But the difference is to keep which one in a weakly connected component.

SemDeDup proposes three options for choosing the examples we keep: 1) keeping examples with low similarity to cluster centroids, 2) keeping random examples, and 3) keeping examples with high similarity to cluster centroids.

However, there is no need to build the graph in the implemention of SemDeDup as the pseudo code in the paper:

```python
# Input : cluster_embeddings , num_clusters , epsilon

for i in range(num_clusters):
    # Load cluster embeddings.
    cluster_i_embeddings = cluster_embeddings[i]

    # Sort the cluster embeddings by the distance to the cluster centroid.
    cluster_i_embeddings = sort_by_distance_to_cluster_centroid(
        cluster_i_embeddings, descending=True)

    # We use descending = True / False for keeping examples with low/ high similarity to cluster centroids. We ignore this step for keeping random examples from each group of similar examples. 
    
    # Compute the pairwise cosine similarity between embeddings
    pairwise_sim_matrix = cluster_i_embeddings @ cluster_i_embeddings.T

    triu_sim_matrix = torch.triu(pairwise_sim_matrix, diagonal=1)

    M = torch.max(triu_sim_matrix, dim=0)[0]

    # Check if the maximum similarity <= the thresholdold.
    points_to_keep_from_cluster_i = M <= 1 - epsilon
```

The code sorts the embeddings by the distance to the cluster centroid and then calculates upper triangular part of the cosine similarity matrix.

So, for the j-th embedding (j-th column), only its cosine similarity with i-th (i<j) embedding is considered.

And that `M = torch.max(triu_sim_matrix, dim=0)[0] > 1 - epsilon` means that, there exists i-th (i<j) embedding being too similar to j-th embedding and we should remove j-th embedding.

Then for all i-j pairs, we keep the smaller-index one and remove the bigger-index one.

Therefore, in a weakly connected component, we keep the embedding with smallest index.

Noted that the smaller the index is, the bigger the distance to the centroid is.

So, in a weakly connected component, we keep the embedding with biggest distance to the cluster centroid, which is consistent with the idea.
(It does not explicitly find the weakly connected component, but traverses all i-j pairs by compute the $n^2$ similarity matrix)

Also, changing the order to smaller distance or randomly shuffling, we can implement the high similarity or random de-duplicating.

In this way, we don't need to build the graph and we can save `M`.



Noted that the graph is changing with threshold changing. 

Changing the threshold, we can get SemDeDup coresets of various sizes quickly with `M` saved. We refer `M` as `SemDeDup_score`. Therefore, in part 2, we would follow the pseudo code in the paper and conduct experiments on LAION-2B.

In part 3, we would implement the graph-building process and explore different strategies of keeping which one in a weakly connected component.

We also combline `clip_score` (the cosine similarity of image-text pair) in pruning.



# Part 2: SemDeDup on LAION-2B

As mentioned before, the process of SemDeDup is:

1. cluster all 2B embeddings by k-means

2. (optional) in each cluster, sort the cluster embeddings by the distance to the cluster centroid

3. in each cluster, compute `M` (and also save `clip_score`)

4. in each cluster, get coreset through `M`, threshold and  `clip_score`



* The first step is implemented in `preprocess.py`. The progress is:
  * train kmeans for some epochs, sampling some embedding files in each epoch because the total 2B embeddings are too large
  
  * select a good result of kmeans, and then assign all 2B embeddings to clusters (ours result of kmeans is supplied as `centroids_supplied.npy`, [download link](https://huggingface.co/datasets/Isaachhe/Dataset-Pruning/tree/main/LAION))
  
  * inversely, re-organize the clusters to `csv` files that record the index ( [`file_id`, `row_id`] means that the embedding is the `row_id`-th of `file_id`-th `npy` file in `img_emb` folder) , primary key and `clip_score` of each embedding belonging to the cluster

  * save the embeddings (select the `row_id`-th of `file_id`-th `npy` file) of a cluster to a `npy` file
  
  It takes approximately 1 day to run these on a node with 8 NVIDIA A100 GPUs.

* The 2-3 steps are implemented in `SemDeDup_compute_score.py`, and `M`, `clip_score` are saved. 

  * compute `M` and save it (chunkwise due to memory limit)
  
  It takes approximately a half day to run these on a node with 8 NVIDIA A100 GPUs.
  
* The step 4 is implemented in `SemDeDup_get_coreset.py`. There are 2 stages:

  * firstly, use a threshold of `SemDeDup_score (M)` to filter the samples
  
  * secondly, further select remaining samples by `clip_score`
  

The distribution of `SemDeDup_score` is shown below.

<img src="/Users/isaache/Downloads/Dataset-Pruning/LAION/assets/distribution.png" alt="distribution" style="zoom:6%;" /><img src="/Users/isaache/Downloads/Dataset-Pruning/LAION/assets/distribution-large.png" alt="distribution" style="zoom:6%;" />

We cluster embeddings to 100K clusters and keep examples with low similarity to cluster centroids.

* *Notes: In our experiments, cluster_787 is too large to compute `M`. We find that the image of cluster_787 is almost an image of ["no obejct"](https://i.ebayimg.com/00/s/ODAwWDgwMA==/z/vngAAOSwgx9geX2H/$_20.JPG), so we remove the whole cluster_787.*


```
python preprocess.py

python SemDeDup_compute_score.py

python SemDeDup_get_coreset.py
```



As mentioned before, we start from 2,053,332,591 samples.

We conduct experiments on Open_CLIP-B/32 and samples seen are set to 4B. (See the training recipe in `openclip` for more information.)

Experimental results are:

| 100% (2.05B) |                           |
| :----------- | :------------------------ |
| *condition*  | *IN-1K-val zero-shot Acc* |
| LAION-2B     | 59.75                     |

| 65% (1.33B)  |                           |
| :----------- | :------------------------ |
| *condition*  | *IN-1K-val zero-shot Acc* |
| SemDeDup_65% | 61.11                     |

| 50% (1.03B)  |                           |
| :----------- | :------------------------ |
| *condition*  | *IN-1K-val zero-shot Acc* |
| SemDeDup_50% | 61.62                     |

| 30% (616M)                        |                           |
| :-------------------------------- | :------------------------ |
| *condition*                       | *IN-1K-val zero-shot Acc* |
| SemDeDup_30%                      | 60.26                     |
| SemDeDup_50% -> clip_score_top60% | 61.33                     |

| 20%                                           |                           |
|:----------------------------------------------| :------------------------ |
| *condition*                                   | *IN-1K-val zero-shot Acc* |
| LAION-400M (407M$`^*`$)                           | 58.28                     |
| SemDeDup_20% (411M)                           | 56.73                     |
| SemDeDup_50% -> clip_score_top40% (411M)      | 60.92                     |
| SemDeDup_50% -> clip_score_rank15%-55% (411M) | 61.26                     |

\**For LAION-400M, we finally get 41456 tars with 407,305,278 samples.*


| 10% (205M)                             |                           |
| :------------------------------------- | :------------------------ |
| *condition*                            | *IN-1K-val zero-shot Acc* |
| SemDeDup_50% -> clip_score_rank30%-50% | 59.70                     |
| SemDeDup_50% -> random20%              | 59.09                     |

| 5% (103M)                              |                           |
| :------------------------------------- | :------------------------ |
| *condition*                            | *IN-1K-val zero-shot Acc* |
| SemDeDup_50% -> clip_score_rank30%-40% | 54.62                     |




* Using SemDeDup to select coresets can improve data quality and performance. For example, 50% SemDeDup coreset improve the performance to 61.62 (+1.87 vesus the whole LAION-2B). 65% and 30% work, too.
* Setting a too small threshold makes the result worse. 20% SemDeDup coreset is worse than LAION-2B (-3.02) and LAION-400M (-1.55).
* Combing SemDeDup and `clip_score`, we get 30% coreset with performance +1.58, 20% coreset with performance +1.51 and 10% coreset with negligible performance loss, when comparing them with the whole LAION-2B. Notably, the best 400M dataset achieves a performance improvement of 2.98 than LAION-400M.
* We observe that `clip_score` top 40% is worse than rank 15%-55%, it may be led by that the image-text pairs with too high cosine similarity are "cheater", which means there exist text region in the image which is almost the caption. [Less is More: Removing Text-regions Improves CLIP Training Efficiency and Robustness](https://arxiv.org/abs/2305.05095) and [T-MARS: Improving Visual Representations by Circumventing Text Feature Learning](https://arxiv.org/abs/2307.03132) pay attention to this, too.






# Part 3: more than SemDeDup

In this part, we implement the graph-building process of semantically de-duplicating:

1. cluster all 2B embeddings by k-means
2. in each cluster, build an undirected graph that making an edge for two embeddings if their cosine similarity is above threshold
3. keep only 1 in each weakly connected component of the graph

We use batch BFS to build the graph that we explore all the next-depth-level nodes at the same time by matrix operations.



We explore several strategies of keeping which one in a weakly connected component:

1. keeping the one with biggest distance to cluster centroid (SemDeDup), referred as "far"

1. keeping the one whose distance to cluster centroid is exactly the median, reffered as "middle"

1. keeping the one whose distance to mean of weakly connected component is exactly the median, reffered as "inner-middle"

1. keeping the one with biggest `clip_score`, referred as "clip_score-max" (this method can be converted to SemDeDup-like computing without building the graph by sorting the cluster embeddings by `clip_score`)

Besides, we save `clip_score` and cosine similarity with cluster centroid of each kept sample for further experiments.

We would like to point out that our framework can support other strategies simply.

```
# python preprocess.py

python dedup.py
(costing 1 day to run it once on a node with 8 NVIDIA A100 GPUs)
```



We conduct experiments on CLIPA-B/16 and samples seen are set to 2.4B 112px + 131M 224px. (See the training recipe in `clipa` for more information.)

Experimental results are:

| *condition*                 | *IN-1K-val zero-shot Acc* |
| :-------------------------- | :------------------------ |
| LAION-2B                    | 64.41                     |
| far 50% 1.027B              | 66.30                     |
| middle 43.9% 901.4M         | 66.62                     |
| middle 46.4% 951.7M         | **66.81**                 |
| middle 50.4% 1.035B         | 66.22                     |
| inner-middle 46.4% 951.7M   | 66.31                     |
| inner-middle 50.3% 1.033B   | 66.46                     |
| clip_score-max 46.4% 951.7M | 66.43                     |



* Current experiments show that "middle", "inner-middle" and "clip_score-max" beat "far" (SemDeDup).
* 46.4% "middle" coreset performs best.
