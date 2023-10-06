OMP_NUM_THREADS=1 torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=8 --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --rdzv_id=pytorchddp --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:8888 -m training.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --train-data 'LAION-2B-data/{000000..232319}.tar' \
    --coreset 'LAION-coreset/SemDeDup_0.945' \
    --dataset-type webdataset \
    --lr "2.56e-5" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 3072 \
    --wd 0.2 \
    --batch-size $((1024/${WORLD_SIZE})) \
    --aug-cfg scale='(0.4, 1.0)' \
    --pos-embed 'learnable' \
    --epochs=1 \
    --train-num-samples 131072000 \
    --workers=6 \
    --model ViT-B-16-CL16 \
    --pretrained 'clipa/logs/2023_08_29-19_09_29-model_ViT-B-16-CL16-lr_0.002048-b_8192-j_6-p_amp_bf16/checkpoints/epoch_latest.pt' \
    --precision 'amp_bf16' \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --to-float-on-device \
    --grad-checkpointing \
    --log-every-n-steps 256 \
    --seed 0 \
    --logs ./logs/ \
    --imagenet-val 'ImageNet-1K/val'



