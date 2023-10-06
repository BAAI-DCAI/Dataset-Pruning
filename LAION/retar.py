import os
import datetime
import multiprocessing
import numpy as np
import webdataset as wds


def tar_samples(process_id, num_process, LAION_2B_data_path):
    with wds.ShardWriter(pattern=f'{output_dir}/{process_id:02d}%06d.tar', maxcount=10000) as sink:
        LAION_2B_data = os.listdir(LAION_2B_data_path)
        LAION_2B_data.sort()

        for idx in range(process_id * len(LAION_2B_data) // num_process,
                         (process_id + 1) * len(LAION_2B_data) // num_process):
            path = f'{LAION_2B_data_path}/{LAION_2B_data[idx]}'
            coreset = set(np.load(f'{coreset_path}/tar_{LAION_2B_data[idx][:-4]}.npy').astype(int))

            temp = wds.WebDataset(path)
            for sample in temp:
                key = int(sample['__key__'])
                if key in coreset:
                    coreset.remove(key)
                    sink.write(sample)

            if process_id == 0:
                print(
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {idx + 1}/{len(LAION_2B_data) // num_process} done")


if __name__ == '__main__':
    output_dir = ''
    coreset_path = ''
    LAION_2B_data_path = ''
    num_process = 70

    os.makedirs(output_dir)
    process_lst = []

    for process_id in range(num_process):
        p = multiprocessing.Process(target=tar_samples, args=(process_id, num_process, LAION_2B_data_path))
        p.start()
        process_lst.append(p)
    for p in process_lst:
        p.join()

    res = os.listdir(output_dir)
    res.sort()
    for i in range(len(res)):
        os.system(f'mv {output_dir}/{res[i]} {output_dir}/{i:0=6d}_retar.tar')
