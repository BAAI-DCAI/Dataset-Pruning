batch_size=4096
workers=$1
total_gpus=8
global_batch_size=$(($total_gpus * $batch_size))

OMP_NUM_THREADS=1 torchrun --nproc_per_node 8 --nnodes 1 -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --log-every-n-steps 62 \
    --dataset-type webdataset \
    --train-data LAION-2B-data/{000000..232319}.tar \
    --imagenet-val ImageNet-1K/val \
    --coreset LAION-coreset/SemDeDup_0.945 \
    --train-num-samples=20000000 \
    --dataset-resampled \
    --warmup 2000 \
    --optimizer lamb \
    --lr 2e-3 \
    --wd 0.05 \
    --batch-size=$batch_size \
    --grad-clip-norm 5.0 \
    --epochs=200 \
    --workers=$workers \
    --model ViT-B-32 \
    --seed 42 \
    --precision 'amp' \
    --local-loss \
    --gather-with-grad \
    --enable-deepspeed \
    --grad-checkpointing