import multiprocessing
import os
import random
import datetime
import faiss
import bisect
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def train_kmeans(img_emb_dir, output_dir, feature_dims, num_clusters, epochs, sample_num_per_epoch, iter_per_epoch):
    os.makedirs(output_dir, exist_ok=True)

    img_emb_files = os.listdir(img_emb_dir)

    kmeans = faiss.Kmeans(feature_dims, k=num_clusters, niter=iter_per_epoch, verbose=True, gpu=True,
                          max_points_per_centroid=100000)

    for epoch in range(epochs):
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch: {epoch} Start")
        sample_emb_files = random.sample(img_emb_files, sample_num_per_epoch)
        samples = []
        for (idx, sample_emb_file) in enumerate(sample_emb_files):
            if idx % 10 == 0 or idx + 1 == sample_num_per_epoch:
                print(
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} reading embedding file {idx + 1}/{sample_num_per_epoch} {sample_emb_file}")
            samples.append(np.load(f'{img_emb_dir}/{sample_emb_file}'))
        samples = np.concatenate(samples)
        kmeans.train(samples, init_centroids=kmeans.centroids)

        np.save(f'{output_dir}/centroids_{epoch}.npy', kmeans.centroids)
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch: {epoch} Done")


def calc_cluster(img_emb_dir, input_kmeans_result, output_dir, feature_dims, num_clusters):
    os.makedirs(output_dir, exist_ok=True)

    kmeans = faiss.Kmeans(feature_dims, k=num_clusters, niter=0, verbose=True, gpu=True, max_points_per_centroid=100000)

    centroids = np.load(input_kmeans_result)
    kmeans.train(centroids, init_centroids=centroids)
    assert (kmeans.centroids == centroids).all()

    img_emb_files = os.listdir(img_emb_dir)
    img_emb_files.sort()
    indices = []
    for (idx, img_emb_file) in enumerate(img_emb_files):
        sample = np.load(f'{img_emb_dir}/{img_emb_file}')
        _, cluster_indices = kmeans.assign(sample)
        indices.append(cluster_indices)
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {idx + 1}/{len(img_emb_files)} done")
    indices = np.concatenate(indices)
    np.save(output_dir, indices)


def join_keys(metadata_dir, output_dir):
    meta_files = os.listdir(metadata_dir)
    meta_files.sort()
    joint_key = []
    for (idx, meta_file) in enumerate(meta_files):
        joint_key.append(pq.read_table(f'{metadata_dir}/{meta_file}').to_pandas()['key'].values.astype(int))
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {idx + 1}/{len(meta_files)} done")
    joint_key = np.concatenate(joint_key)
    np.save(output_dir, joint_key)


def join_clip_scores(clip_score_dir, output_dir):
    clip_score_files = os.listdir(clip_score_dir)
    clip_score_files.sort()
    joint_clip_score = []
    for (idx, clip_score_file) in enumerate(clip_score_files):
        joint_clip_score.append(np.load(f'{clip_score_dir}/{clip_score_file}'))
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {idx + 1}/{len(clip_score_files)} done")
    joint_clip_score = np.concatenate(joint_clip_score)
    np.save(output_dir, joint_clip_score)


def calc_global_local_relation(img_emb_dir, output_dir):
    img_emb_files = os.listdir(img_emb_dir)
    img_emb_files.sort()

    pre_sum = [0]

    for img_emb_file in img_emb_files:
        pre_sum.append(np.load(f'{img_emb_dir}/{img_emb_file}').shape[0])

    for i in range(2, len(pre_sum)):
        pre_sum[i] += pre_sum[i - 1]

    pre_sum = np.asarray(pre_sum)
    np.savetxt(output_dir, pre_sum)


def reorganize_multiprocessing(input_clutser_index, joint_key_file, joint_clip_score_file, output_dir, num_clusters,
                               rela_file, num_process):
    def reorganize(input_cluster_index, joint_key_file, joint_clip_score_file, output_dir, num_clusters, rela_file,
                   process_id, num_process):
        os.makedirs(output_dir, exist_ok=True)
        gl_rela = np.loadtxt(rela_file).astype(int)

        def global2local(idx):
            file_id = bisect.bisect(gl_rela, idx) - 1
            local_idx = idx - gl_rela[file_id]
            return (file_id, local_idx)

        cluster_indices = np.load(input_cluster_index)
        primary_keys = np.load(joint_key_file)
        joint_clip_score = np.load(joint_clip_score_file)
        for cluster_id in range(process_id * num_clusters // num_process,
                                (process_id + 1) * num_clusters // num_process):
            global_indices = np.where(cluster_indices == cluster_id)[0].tolist()
            local_indices = list(map(global2local, global_indices))
            needed_keys = primary_keys[global_indices]
            needed_clip_scores = joint_clip_score[global_indices]
            with open(f'{output_dir}/cluster_{cluster_id}.csv', 'w') as f:
                f.write('file_id,row_id,key,clip_score_ViT-L/14\n')
                for (idx, (file_id, local_idx)) in enumerate(local_indices):
                    f.write(f'{file_id}, {local_idx}, {needed_keys[idx]}, {needed_clip_scores[idx]}\n')
            if process_id == 0:
                print(
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} process {process_id} "
                    f"{cluster_id - process_id * num_clusters // num_process + 1}/{num_clusters // num_process} done")

    process_lst = []
    for process_id in range(num_process):
        p = multiprocessing.Process(target=reorganize,
                                    args=(input_clutser_index, joint_key_file, joint_clip_score_file, output_dir,
                                          num_clusters, rela_file, process_id, num_process))
        p.start()
        process_lst.append(p)
    for p in process_lst:
        p.join()


def calc_cluster_embedding(img_emb_dir, input_cluster_map_dir, output_dir, num_clusters, split_num):
    os.makedirs(output_dir, exist_ok=True)

    for counter in range(split_num):
        many_cluster_emb_files_dict = []

        for cluster_id in range(counter * num_clusters // split_num,
                                (counter + 1) * num_clusters // split_num):

            emb_files_dict = {}

            cluster_map = pd.read_csv(f'{input_cluster_map_dir}/cluster_{cluster_id}.csv')
            file_id_lst = cluster_map['file_id'].values
            local_idx_lst = cluster_map['row_id'].values

            for i in range(len(file_id_lst)):
                file_id = file_id_lst[i]
                local_idx = local_idx_lst[i]
                if file_id not in emb_files_dict:
                    emb_files_dict[file_id] = []
                emb_files_dict[file_id].append(local_idx)

            many_cluster_emb_files_dict.append(emb_files_dict)

            if (cluster_id + 1) % 100 == 0 or cluster_id + 1 == (counter + 1) * num_clusters // split_num:
                print(
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} load cluster index {cluster_id + 1}/{num_clusters} done")

        many_cluster_embs = [[] for _ in range(counter * num_clusters // split_num,
                                               (counter + 1) * num_clusters // split_num)]

        img_emb_files = os.listdir(img_emb_dir)
        img_emb_files.sort()
        for (idx, img_emb_file) in enumerate(img_emb_files):
            img_emb = np.load(f'{img_emb_dir}/{img_emb_file}')

            for cluster_id in range(counter * num_clusters // split_num,
                                    (counter + 1) * num_clusters // split_num):
                bias = cluster_id - counter * num_clusters // split_num

                if idx not in many_cluster_emb_files_dict[bias]:
                    continue

                local_indices = many_cluster_emb_files_dict[bias][idx]
                local_indices.sort()
                many_cluster_embs[bias].extend(img_emb[local_indices])

            print(
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} load embedding file {idx + 1}/{len(img_emb_files)} and assign to clusters done")

        for cluster_id in range(counter * num_clusters // split_num,
                                (counter + 1) * num_clusters // split_num):
            bias = cluster_id - counter * num_clusters // split_num
            embs = np.asarray(many_cluster_embs[bias])
            np.save(f'{output_dir}/cluster_{cluster_id}.npy', embs)


def calc_SemDeDup_score_and_clip_score_multigpu(input_cluster_emb_dir, input_cluster_map_dir, input_kmeans_result,
                                                output_dir, num_clusters, ngpus):
    os.makedirs(output_dir, exist_ok=True)

    def calc_SemDeDup_score_and_clip_score(input_cluster_emb_dir, input_cluster_map_dir, input_kmeans_result,
                                           output_dir, num_clusters, gpu_id, ngpus):
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
                        pairwise_sim_matrix = torch.sum(
                            embs[counter * total_n // split_num:(counter + 1) * total_n // split_num].unsqueeze(
                                0) * embs.unsqueeze(1),
                            dim=2)
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
        p = multiprocessing.Process(target=calc_SemDeDup_score_and_clip_score,
                                    args=(
                                        input_cluster_emb_dir, input_cluster_map_dir, input_kmeans_result, output_dir,
                                        num_clusters, process_id, ngpus))
        p.start()
        process_lst.append(p)
    for p in process_lst:
        p.join()


if __name__ == '__main__':
    img_emb_dir = 'LAION-2B-embeddings/img_emb'
    output_dir = 'LAION-2B-SemDeDup'
    feature_dims = 768
    num_clusters = 100000
    epochs = 200
    sample_num_per_epoch = 20
    iter_per_epoch = 10
    train_kmeans(img_emb_dir, f'{output_dir}/centroids', feature_dims, num_clusters, epochs, sample_num_per_epoch,
                 iter_per_epoch)

    chosen_kmeans_result = 'centroids_supplied.npy'
    calc_cluster(img_emb_dir, chosen_kmeans_result, f'{output_dir}/indices.npy', feature_dims, num_clusters)

    metadata_dir = 'LAION-2B-embeddings/metadata'
    join_keys(metadata_dir, f'{output_dir}/joint_key.npy')
    clip_score_dir = 'LAION-2B-embeddings/clip_score'
    join_clip_scores(clip_score_dir, f'{output_dir}/joint_clip_score.npy')
    calc_global_local_relation(img_emb_dir, f'{output_dir}/gl_rela.txt')

    reorganize_multiprocessing(f'{output_dir}/indices.npy', f'{output_dir}/joint_key.npy',
                               f'{output_dir}/joint_clip_score.npy', f'{output_dir}/cluster_map', num_clusters,
                               f'{output_dir}/gl_rela.txt', num_process=40)

    calc_cluster_embedding(img_emb_dir, f'{output_dir}/cluster_map', f'{output_dir}/cluster_emb', num_clusters,
                           split_num=10)

    calc_SemDeDup_score_and_clip_score_multigpu(f'{output_dir}/cluster_emb', f'{output_dir}/cluster_map',
                                                chosen_kmeans_result, f'{output_dir}/res_SemDeDup_far', num_clusters,
                                                ngpus=8)
