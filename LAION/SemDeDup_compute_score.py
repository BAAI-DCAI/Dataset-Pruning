import multiprocessing
import os
import datetime
import torch
import numpy as np
import pandas as pd


def calc_SemDeDup_score_multigpu(input_cluster_emb_dir, input_cluster_map_dir, input_kmeans_result,
                                 output_dir, num_clusters, ngpus):
    os.makedirs(output_dir, exist_ok=True)

    def calc_SemDeDup_score(input_cluster_emb_dir, input_cluster_map_dir, input_kmeans_result, output_dir, num_clusters,
                            gpu_id, ngpus):
        device = f'cuda:{gpu_id}'
        centroids = np.load(input_kmeans_result)

        for cluster_id in range(gpu_id * num_clusters // ngpus, (gpu_id + 1) * num_clusters // ngpus):
            if os.path.exists(f'{output_dir}/cluster_{cluster_id}.csv'):
                continue
            if cluster_id == 787:
                continue

            centroid = centroids[cluster_id]

            embs = np.load(f'{input_cluster_emb_dir}/cluster_{cluster_id}.npy')

            embs_and_distance = [(emb, np.linalg.norm(centroid - emb), idx) for (idx, emb) in enumerate(embs)]

            # tend to keep samples near to centroids
            # embs_and_distance.sort(key=lambda x: x[1])
            # tend to keep samples far from centroids
            embs_and_distance.sort(key=lambda x: -x[1])
            # random
            # random.shuffle(embs_and_distance)

            embs = torch.from_numpy(np.array([emb for (emb, d, i) in embs_and_distance])).float().to(device)
            total_n = len(embs)
            SemDeDup_score_lst = []

            while len(SemDeDup_score_lst) == 0:
                try:
                    split_num = max(1, len(embs) * len(embs) // 2000000)
                    for counter in range(split_num):
                        pairwise_sim_matrix = torch.matmul(
                            embs,
                            embs[counter * total_n // split_num:(counter + 1) * total_n // split_num].t()
                        )
                        triu_sim_matrix = torch.triu(pairwise_sim_matrix, diagonal=1 - counter * total_n // split_num)
                        SemDeDup_score_lst.extend(torch.max(triu_sim_matrix, dim=0)[0].cpu().numpy())
                except:
                    split_num *= 2
                    SemDeDup_score_lst = []

            SemDeDup_score_lst = [[SemDeDup_score, i, embs_and_distance[i][2]] for (i, SemDeDup_score) in
                                  enumerate(SemDeDup_score_lst)]
            SemDeDup_score_lst.sort(key=lambda x: x[2])
            SemDeDup_score_lst = np.array(SemDeDup_score_lst)

            cluster_map = pd.read_csv(f'{input_cluster_map_dir}/cluster_{cluster_id}.csv')

            cluster_map.insert(loc=len(cluster_map.columns), column='SemDeDup_score', value=SemDeDup_score_lst[:, 0])
            cluster_map.insert(loc=len(cluster_map.columns), column='dis_rank', value=SemDeDup_score_lst[:, 1])

            cluster_map.to_csv(f'{output_dir}/cluster_{cluster_id}.csv')
            if gpu_id == 0:
                print(
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} GPU {gpu_id} "
                    f"calculate similarity {cluster_id + 1 - gpu_id * num_clusters // ngpus}/{num_clusters // ngpus} done")

    process_lst = []
    for process_id in range(ngpus):
        p = multiprocessing.Process(target=calc_SemDeDup_score,
                                    args=(
                                        input_cluster_emb_dir, input_cluster_map_dir, input_kmeans_result, output_dir,
                                        num_clusters, process_id, ngpus))
        p.start()
        process_lst.append(p)
    for p in process_lst:
        p.join()


if __name__ == '__main__':
    output_dir = 'LAION-2B-SemDeDup'

    num_clusters = 100000

    chosen_kmeans_result = 'centroids_supplied.npy'

    calc_SemDeDup_score_multigpu(f'{output_dir}/cluster_emb', f'{output_dir}/cluster_map', chosen_kmeans_result,
                                 f'{output_dir}/res_SemDeDup_far', num_clusters, ngpus=8)
