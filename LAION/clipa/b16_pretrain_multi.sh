OMP_NUM_THREADS=1 torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=8 --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --rdzv_id=pytorchddp --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:8888 -m training.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --train-data 'LAION-2B-data/{000000..232319}.tar' \
    --coreset 'LAION-coreset/SemDeDup_0.945' \
    --dataset-type webdataset \
    --lr "2.048e-3" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 782 \
    --wd 0.2 \
    --batch-size $((8192/${WORLD_SIZE})) \
    --aug-cfg scale='(0.4, 1.0)' \
    --pos-embed 'sin_cos_2d' \
    --epochs=6 \
    --train-num-samples=400000000 \
    --dataset-resampled \
    --workers=6 \
    --model ViT-B-16-CL16 \
    --precision 'amp_bf16' \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --force-image-size 112 \
    --to-float-on-device \
    --grad-checkpointing \
    --log-every-n-steps 61 --zeroshot-steps 610 --val-steps 610 \
    --seed 0 \
    --logs ./logs/ \
    --imagenet-val 'ImageNet-1K/val'



