import multiprocessing
import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize


def dedup_multigpu(input_cluster_emb_dir, input_cluster_map_dir, input_kmeans_result, output_dir, num_clusters, ngpus,
                   threshold, strategy):
    os.makedirs(output_dir, exist_ok=True)

    def dedup(input_cluster_emb_dir, input_cluster_map_dir, input_kmeans_result, output_dir, num_clusters, gpu_id,
              threshold, strategy):
        centroids = np.load(input_kmeans_result)

        device = f'cuda:{gpu_id}'

        total_keep_num = 0
        total_process_num = 0

        for cluster_id in range(gpu_id * num_clusters // ngpus, (gpu_id + 1) * num_clusters // ngpus):
            if os.path.exists(f'{output_dir}/cluster_{cluster_id}.npy'):
                continue
            if cluster_id == 787:
                continue

            centroid = centroids[cluster_id]
            centroid_for_cosine = torch.from_numpy(centroid).float().to(device)
            centroid_for_cosine = normalize(centroid_for_cosine, dim=0)

            embs = torch.from_numpy(np.load(f'{input_cluster_emb_dir}/cluster_{cluster_id}.npy')).float().to(device)
            embs_copy = embs.clone()

            cosine_with_centroid = torch.sum(centroid_for_cosine * embs, dim=1).cpu().numpy()

            clip_scores = torch.from_numpy(
                pd.read_csv(f'{input_cluster_map_dir}/cluster_{cluster_id}.csv')[
                    'clip_score_ViT-L/14'].values).to(
                device)

            total_n = len(embs)

            index_map = torch.Tensor([*range(total_n)]).int().to(device)

            selected = []

            # batch BFS to build the graph that we explore all the next-depth-level nodes at the same time by matmul
            while len(embs):
                try:
                    sim = torch.sum(embs[0] * embs, dim=1)
                except:
                    split_num = len(embs) // 100000
                    embs_len = len(embs)
                    sim = None
                    for counter in range(split_num):
                        part_sim = torch.matmul(
                            embs[0],
                            embs[counter * embs_len // split_num:(counter + 1) * embs_len // split_num].t()
                        )
                        if sim is None:
                            sim = part_sim
                        else:
                            sim = torch.cat((sim, part_sim), dim=0)

                merged = torch.nonzero(sim > threshold).squeeze(-1)

                group = index_map[merged]

                left = torch.nonzero(sim <= threshold).squeeze(-1)

                try:
                    pre = embs[merged[1:]]
                    embs = embs[left]
                except:
                    embs = embs.to('cpu')
                    merged = merged.to('cpu')
                    pre = embs[merged[1:]].to(device)
                    merged = merged.to(device)

                    left = left.to('cpu')
                    embs = embs[left].to(device)
                    left = left.to(device)

                index_map = index_map[left]

                if len(merged) > 1 and len(left) > 0:
                    while True:
                        split_num = max(1, len(pre) * len(embs) // 700000)
                        while True:
                            try:
                                merge_flag = torch.zeros(len(embs)).to(device)
                                pre_len = len(pre)

                                for counter in range(split_num):
                                    part_sim = torch.matmul(
                                        pre[counter * pre_len // split_num:(counter + 1) * pre_len // split_num],
                                        embs.t()
                                    )
                                    merge_flag += torch.sum(torch.where(part_sim > threshold, 1, 0), dim=0)
                                break
                            except:
                                split_num *= 2

                        merged = torch.nonzero(merge_flag).squeeze(-1)

                        if len(merged) == 0:
                            break

                        group = torch.cat((group, index_map[merged]), dim=0)

                        left = (merge_flag == 0).nonzero().squeeze(-1)

                        pre = embs[merged]
                        embs = embs[left]
                        index_map = index_map[left]

                try:
                    group_embs = embs_copy[group].cpu().numpy()
                except:
                    embs_copy = embs_copy.cpu()
                    group = group.cpu()
                    group_embs = embs_copy[group].numpy()

                if strategy in ['middle', 'far', 'inner-middle']:
                    if strategy == 'inner-middle':
                        centroid = np.mean(group_embs, axis=0)

                    dis_and_rank = [(np.linalg.norm(emb - centroid), i) for (i, emb) in enumerate(group_embs)]
                    dis_and_rank.sort(key=lambda x: x[0])

                    if strategy in ['middle', 'inner-middle']:
                        selected.append(group[dis_and_rank[len(dis_and_rank) // 2][1]].item())
                    else:
                        selected.append(group[dis_and_rank[-1][1]].item())
                elif strategy == 'clip_score-max':
                    group_clip_scores = clip_scores[group].cpu().numpy()
                    selected.append(np.argmax(group_clip_scores))
                else:
                    raise ValueError('Unknown strategy')

            key_df = pd.read_csv(f'{input_cluster_map_dir}/cluster_{cluster_id}.csv')['key'].values.astype(int)

            selected.sort()
            selected = np.asarray(selected)
            keep_array = key_df[selected]
            clip_score_array = clip_scores.cpu().numpy()[selected]
            cosine_with_centroid_array = cosine_with_centroid[selected]
            total_array = np.asarray([keep_array, clip_score_array, cosine_with_centroid_array])

            np.save(f'{output_dir}/cluster_{cluster_id}.npy', total_array)

            keep_num = len(keep_array)
            total_keep_num += keep_num
            total_process_num += total_n

            if gpu_id == 0:
                print(
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} GPU {gpu_id} "
                    f"calculate coreset {cluster_id + 1 - gpu_id * num_clusters // ngpus}/{num_clusters // ngpus} done, "
                    f"{keep_num / total_n * 100:.2f}% kept, average {total_keep_num / total_process_num * 100:.2f}% kept")

    process_lst = []
    for process_id in range(ngpus):
        p = multiprocessing.Process(target=dedup,
                                    args=(
                                        input_cluster_emb_dir, input_cluster_map_dir, input_kmeans_result, output_dir,
                                        num_clusters, process_id, threshold, strategy))
        p.start()
        process_lst.append(p)
    for p in process_lst:
        p.join()


if __name__ == '__main__':
    output_dir = 'LAION-2B-dedup'
    num_clusters = 100000
    threshold = 0.86
    total_num = 2053332591
    tar_num = 232320

    chosen_kmeans_result = 'centroids_supplied.npy'

    strategy = 'middle'  # 'far', 'inner-middle', 'clip_score-max'

    dedup_multigpu('LAION-2B-SemDeDup/cluster_emb', 'LAION-2B-SemDeDup/cluster_map', chosen_kmeans_result,
                   f'{output_dir}/{strategy}{threshold}/cluster-wise/', num_clusters, 8, threshold, strategy)

    clip_score_range = None  # (0.4, 0.6)
    cosine_range = None  # (0.2,0.4)

    total_array = []
    total_score_array = []
    total_cosine_array = []
    for cluster_id in range(num_clusters):
        if cluster_id == 787:
            continue
        temp = np.load(f"{output_dir}/{strategy}{threshold}/cluster-wise//cluster_{cluster_id}.npy")

        total_array.append(temp[0])
        total_score_array.append(temp[1])
        total_cosine_array.append(temp[2])

    total_array = np.concatenate(total_array)
    total_score_array = np.concatenate(total_score_array)
    total_cosine_array = np.concatenate(total_cosine_array)

    result_array = total_array
    s1, s2 = '', ''
    if clip_score_range is not None:
        result_array = result_array[np.argsort(total_score_array)[
                                    -int(clip_score_range[1] * len(total_score_array)):
                                    -int(clip_score_range[0] * len(total_score_array) + 1)]
        ]
        s1 = f' and clip_score range ({clip_score_range[0], clip_score_range[1]})'

    if cosine_range is not None:
        result_array = result_array[np.argsort(total_cosine_array)[
                                    -int(cosine_range[1] * len(total_cosine_array)):
                                    -int(cosine_range[0] * len(total_cosine_array) + 1)]
        ]
        s2 = f' and cosine_with_centroid range ({cosine_range[0], cosine_range[1]})'

    print(
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} calculate coreset on threshold {threshold}{s1}{s2}"
        f" done, {len(result_array) / total_num * 100:.2f}% ({len(result_array)})kept")

    assign_array = [[] for _ in range(tar_num)]
    for i in result_array:
        assign_array[i // 10000].append(i)
    for i in range(tar_num):
        np.save(f"{output_dir}/{strategy}{threshold}/tar-wise/tar_{i:0=6d}.npy", np.asarray(assign_array[i]))
        if i % 10000 == 0 or (i + 1) == tar_num:
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} calculate {i + 1}/{tar_num} done")
