# Data root
DATA_ROOT="/data/owcl_data/hdf5"
RESULTS_DIR="./results/for_tmlr"
DATASET="CIFAR100,SUN397,EuroSAT,OxfordIIITPet,Flowers102,FGVCAircraft,StanfordCars,Food101"
HELD_OUT_DATASET="ImageNet,UCF101,DTD"

# CLIP zero-shot
python main_task.py \
    --use_encoded_dataset \
    --data_root ${DATA_ROOT} \
    --backbone ViT-B/32 \
    --results_dir ${RESULTS_DIR} \
    --save \
    --engine main \
    --datasets ${DATASET} \
    --held_out_dataset ${HELD_OUT_DATASET} \
    --incremental dataset \
    --mix_mode clip_only \
    --csv_file task_incremental.csv \
    --retriever knn \
    --num_clusters 1 \
    --exp_name  clip_zero_shot_task_incremental

# CLIP + KNN (AIM-Emb)
python main_task.py \
    --use_encoded_dataset \
    --data_root ${DATA_ROOT} \
    --backbone ViT-B/32 \
    --results_dir ${RESULTS_DIR} \
    --save \
    --engine main \
    --datasets ${DATASET} \
    --held_out_dataset ${HELD_OUT_DATASET} \
    --incremental dataset \
    --mix_mode complementary \
    --buffer_type aim_emb \
    --k 9 \
    --retriever knn \
    --csv_file task_incremental.csv \
    --exp_name clip_knn_task_incremental

# Linear probe
python main_task.py \
    --use_encoded_dataset \
    --data_root ${DATA_ROOT} \
    --backbone ViT-B/32 \
    --results_dir ${RESULTS_DIR} \
    --save \
    --engine main \
    --datasets ${DATASET} \
    --held_out_dataset ${HELD_OUT_DATASET} \
    --incremental dataset \
    --mix_mode complementary \
    --buffer_type exemplar_only \
    --k 9 \
    --retriever tree_probe \
    --tree_probe_max_instances 50000000 \
    --tree_probe_min_samples 1 \
    --csv_file task_incremental.csv \
    --exp_name linear_probe_task_incremental

# CLIP + Linear probe (AVG-Emb, 0.5)
python main_task.py \
    --use_encoded_dataset \
    --data_root ${DATA_ROOT} \
    --backbone ViT-B/32 \
    --results_dir ${RESULTS_DIR} \
    --save \
    --engine main \
    --datasets ${DATASET} \
    --held_out_dataset ${HELD_OUT_DATASET} \
    --incremental dataset \
    --mix_mode complementary \
    --buffer_type avg_emb \
    --k 9 \
    --retriever tree_probe \
    --tree_probe_max_instances 50000000 \
    --tree_probe_min_samples 1 \
    --csv_file task_incremental.csv \
    --exp_name clip_linear_avg_half_emb_task_incremental

# CLIP + Linear probe (AIM-Emb)
python main_task.py \
    --use_encoded_dataset \
    --data_root ${DATA_ROOT} \
    --backbone ViT-B/32 \
    --results_dir ${RESULTS_DIR} \
    --save \
    --engine main \
    --datasets ${DATASET} \
    --held_out_dataset ${HELD_OUT_DATASET} \
    --incremental dataset \
    --mix_mode complementary \
    --buffer_type aim_emb \
    --k 9 \
    --retriever tree_probe \
    --tree_probe_max_instances 50000000 \
    --tree_probe_min_samples 1 \
    --csv_file task_incremental.csv \
    --exp_name clip_linear_aim_emb_task_incremental

# CLIP + Tree-Probe (AIM-Emb)
for tp_max_instances in 1000 2000 5000 10000 20000 50000 100000; do
    python main_task.py \
        --use_encoded_dataset \
        --data_root ${DATA_ROOT} \
        --backbone ViT-B/32 \
        --results_dir ${RESULTS_DIR} \
        --save \
        --engine main \
        --datasets ${DATASET} \
        --held_out_dataset ${HELD_OUT_DATASET} \
        --incremental dataset \
        --mix_mode complementary \
        --buffer_type aim_emb \
        --k 9 \
        --retriever tree_probe \
        --tree_probe_max_instances ${tp_max_instances} \
        --tree_probe_min_samples 1 \
        --csv_file task_incremental.csv \
        --exp_name clip_tree_probe_aim_emb_task_incremental_${tp_max_instances}
done

# ZSCL
python main_zscl.py \
    --use_encoded_dataset \
    --data_root ${DATA_ROOT} \
    --backbone ViT-B/32 \
    --results_dir ${RESULTS_DIR} \
    --save \
    --engine ZSCL \
    --datasets ${DATASET} \
    --held_out_dataset ${HELD_OUT_DATASET} \
    --incremental dataset \
    --optimizer adamw \
    --lr 1e-5 \
    --weight_decay 0 \
    --n_epochs 100 \
    --batch_size 64 \
    --ref_caption_root /data/conceptual_captions \
    --ref_image_dataset ImageNet \
    --warmup_length 100 \
    --csv_file task_incremental.csv \
    --exp_name zscl_task_incremental

# CLIP Fine-Tune
python main_tune.py \
    --use_encoded_dataset \
    --data_root ${DATA_ROOT} \
    --backbone ViT-B/32 \
    --results_dir ${RESULTS_DIR} \
    --save \
    --engine tune \
    --datasets ${DATASET} \
    --held_out_dataset ${HELD_OUT_DATASET} \
    --incremental dataset \
    --optimizer adamw \
    --lr 1e-5 \
    --weight_decay 0 \
    --n_epochs 100 \
    --batch_size 64 \
    --warmup_length 100 \
    --csv_file task_incremental.csv \
    --exp_name clip_finetune_task_incremental

