
setting='S4'
visual_backbone="pvt" # "resnet" or "pvt"
        
CUDA_VISIBLE_DEVICES=0 python train.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --train_batch_size 8 --num_tokens 32 --Adapter_downsample 8 \
        --lr 0.0003 \
        --tpavi_stages 0 1 2 3 \
        --wandb 0 \
        --model_name s4-swinv2-tune-av 


