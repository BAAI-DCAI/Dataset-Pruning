import multiprocessing
import os
import datetime
import numpy as np
import pandas as pd


def calc(process_id, num_process, num_cluster, threshold, output_dir, clip_score_range):
    os.makedirs(output_dir, exist_ok=True)
    for cluster_id in range(process_id * num_cluster // num_process, (process_id + 1) * num_cluster // num_process):
        if cluster_id == 787:
            continue
        SemDeDup_res = pd.read_csv(f'LAION-2B-SemDeDup/res_SemDeDup_far/cluster_{cluster_id}.csv')
        SemDeDup_score = SemDeDup_res['SemDeDup_score'].values
        dis_rank = SemDeDup_res['dis_rank'].values
        clip_score = SemDeDup_res['clip_score_ViT-L/14'].values
        SemDeDup_score_lst = [(SemDeDup_score[i], dis_rank[i], i) for i in range(len(SemDeDup_res))]
        SemDeDup_score_lst.sort(key=lambda x: x[1])

        key_df = pd.read_csv(f'LAION-2B-SemDeDup/cluster_map/cluster_{cluster_id}.csv')['key'].values.astype(int)
        keep_array = []
        clip_score_array = []
        keep_num = 0

        for (cos, _, idx) in SemDeDup_score_lst:
            if cos <= threshold:
                keep_num += 1
                keep_array.append(key_df[idx])
                if clip_score_range is not None:
                    clip_score_array.append(clip_score[idx])
        keep_array = np.array(keep_array)
        np.save(f'{output_dir}/cluster_{cluster_id}.npy', keep_array)
        if clip_score_range is not None:
            clip_score_array = np.array(clip_score_array)
            np.save(f'{output_dir}/cluster_{cluster_id}_score.npy', clip_score_array)
        keep_num = keep_num / len(SemDeDup_score_lst) * 100
        if process_id == 0:
            print(
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} process {process_id} "
                f"calculate coreset {cluster_id + 1} done, {keep_num:.2f}% kept")


if __name__ == '__main__':
    num_cluster = 100000
    total_num = 2053332591
    tar_num = 232320
    threshold = 0.945
    clip_score_range = (0.3, 0.5)  # or None
    if clip_score_range is None:
        output_dir = f"LAION-coreset/SemDeDup_{threshold}"
    else:
        output_dir = f"LAION-coreset/SemDeDup_{threshold}_then_clip_score_{clip_score_range[0]}_{clip_score_range[1]}"

    num_process = 500
    process_lst = []
    for process_id in range(num_process):
        p = multiprocessing.Process(target=calc,
                                    args=(
                                        process_id, num_process, num_cluster, threshold, output_dir, clip_score_range))
        p.start()
        process_lst.append(p)
    for p in process_lst:
        p.join()

    total_array = []
    total_score_array = []
    for cluster_id in range(num_cluster):
        if cluster_id == 787:
            continue
        total_array.append(np.load(f"{output_dir}/cluster_{cluster_id}.npy"))
        if clip_score_range is not None:
            total_score_array.append(np.load(f"{output_dir}/cluster_{cluster_id}_score.npy"))
    total_array = np.concatenate(total_array)
    np.save(f"{output_dir}/total_key.npy", total_array)
    if clip_score_range is not None:
        total_score_array = np.concatenate(total_score_array)
        np.save(f"{output_dir}/total_score.npy", total_score_array)

    if clip_score_range is not None:
        keep_array = total_array[np.argsort(total_score_array)[-int(clip_score_range[1] * len(total_array)):-int(
            clip_score_range[0] * len(total_array) + 1)]]
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} calculate coreset on threshold {threshold} "
              f"and clip_score range ({clip_score_range[0], clip_score_range[1]}"
              f"done, {len(keep_array) / total_num * 100:.2f}% ({len(keep_array)})kept")
    else:
        keep_array = total_array
        print(
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} calculate coreset on threshold {threshold} "
            f"done, {len(keep_array) / total_num * 100:.2f}% ({len(keep_array)})kept")

    assign_array = [[] for _ in range(tar_num)]
    for i in keep_array:
        assign_array[i // 10000].append(i)
    for i in range(tar_num):
        np.save(f"{output_dir}/tar_{i:0=6d}.npy", np.asarray(assign_array[i]))
        if i % 10000 == 0 or (i + 1) == tar_num:
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} calculate {i + 1}/{tar_num} done")
