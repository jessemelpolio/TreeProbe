# CIFAR100,SUN397,EuroSAT,OxfordIIITPet,Flowers102,FGVCAircraft,StanfordCars,Food101

DATA=/data/owcl_data/hdf5
RESULTS_DIR=./results/tmlr
HOLD_OUT_DATASET=ImageNet,UCF101,DTD

# CLIP Zero-shot
for dataset in CIFAR100 SUN397 EuroSAT OxfordIIITPet Flowers102 FGVCAircraft StanfordCars Food101; do
    python main_data.py \
        --use_encoded_dataset \
        --data_root ${DATA} \
        --backbone ViT-B/32 \
        --results_dir ${RESULTS_DIR} \
        --save \
        --engine main \
        --datasets ${dataset} \
        --held_out_dataset ${HOLD_OUT_DATASET} \
        --incremental data \
        --mix_mode clip_only \
        --csv_file data_incremental.csv \
        --exp_name clip_zero_shot_data_incremental
done

# CLIP + KNN (AIM-Emb)
for dataset in CIFAR100 SUN397 EuroSAT OxfordIIITPet Flowers102 FGVCAircraft StanfordCars Food101; do
    python main_data.py \
        --use_encoded_dataset \
        --data_root ${DATA} \
        --backbone ViT-B/32 \
        --results_dir ${RESULTS_DIR} \
        --save \
        --engine main \
        --datasets ${dataset} \
        --held_out_dataset ${HOLD_OUT_DATASET} \
        --incremental data \
        --mix_mode complementary \
        --buffer_type aim_emb \
        --k 9 \
        --retriever knn \
        --csv_file data_incremental.csv \
        --exp_name clip_knn_data_incremental
done

# LinProbe
for dataset in CIFAR100 SUN397 EuroSAT OxfordIIITPet Flowers102 FGVCAircraft StanfordCars Food101; do
    python main_data.py \
        --use_encoded_dataset \
        --data_root ${DATA} \
        --backbone ViT-B/32 \
        --results_dir ${RESULTS_DIR} \
        --save \
        --engine main \
        --datasets ${dataset} \
        --held_out_dataset ${HOLD_OUT_DATASET} \
        --incremental data \
        --mix_mode complementary \
        --buffer_type exemplar_only \
        --k 9 \
        --retriever tree_probe \
        --tree_probe_max_instances 50000000000 \
        --tree_probe_min_samples 1 \
        --csv_file data_incremental.csv \
        --exp_name clip_linprobe_data_incremental
done

# CLIP + Linear Probe (AIM-Emb)
for dataset in CIFAR100 SUN397 EuroSAT OxfordIIITPet Flowers102 FGVCAircraft StanfordCars Food101; do
    python main_data.py \
        --use_encoded_dataset \
        --data_root ${DATA} \
        --backbone ViT-B/32 \
        --results_dir ${RESULTS_DIR} \
        --save \
        --engine main \
        --datasets ${dataset} \
        --held_out_dataset ${HOLD_OUT_DATASET} \
        --incremental data \
        --mix_mode complementary \
        --buffer_type aim_emb \
        --k 9 \
        --retriever tree_probe \
        --tree_probe_max_instances 50000 \
        --tree_probe_min_samples 1 \
        --csv_file data_incremental.csv \
        --exp_name clip_linprobe_aim_emb_data_incremental
done

# CLIP + TreeProbe (AIM-Emb)
for tp_max_instances in 1000 2000 5000 10000 20000 50000 100000; do
    for dataset in CIFAR100 SUN397 EuroSAT OxfordIIITPet Flowers102 FGVCAircraft StanfordCars Food101; do
        python main_data.py \
            --use_encoded_dataset \
            --data_root ${DATA} \
            --backbone ViT-B/32 \
            --results_dir ${RESULTS_DIR} \
            --save \
            --engine main \
            --datasets ${dataset} \
            --held_out_dataset ${HOLD_OUT_DATASET} \
            --incremental data \
            --mix_mode complementary \
            --buffer_type aim_emb \
            --k 9 \
            --retriever tree_probe \
            --tree_probe_max_instances ${tp_max_instances} \
            --tree_probe_min_samples 1 \
            --csv_file data_incremental.csv \
            --exp_name clip_tree_probe_aim_emb_data_incremental_${tp_max_instances}
    done
done

# ZSCL
for dataset in CIFAR100 SUN397 EuroSAT OxfordIIITPet Flowers102 FGVCAircraft StanfordCars Food101; do
    python main_data.py \
        --use_encoded_dataset \
        --data_root ${DATA} \
        --backbone ViT-B/32 \
        --results_dir ${RESULTS_DIR} \
        --save \
        --engine ZSCL \
        --datasets ${dataset} \
        --held_out_dataset ${HOLD_OUT_DATASET} \
        --incremental data \
        --optimizer adamw \
        --lr 1e-5 \
        --weight_decay 0 \
        --n_epochs 100 \
        --batch_size 64 \
        --ref_caption_root /data/conceptual_captions \
        --ref_image_dataset ImageNet \
        --warmup_length 100 \
        --csv_file data_incremental.csv \
        --exp_name zscl_data_incremental
done

# CLIP Fine-Tune
for dataset in CIFAR100 SUN397 EuroSAT OxfordIIITPet Flowers102 FGVCAircraft StanfordCars Food101; do
    python main_data.py \
        --use_encoded_dataset \
        --data_root ${DATA} \
        --backbone ViT-B/32 \
        --results_dir ${RESULTS_DIR} \
        --save \
        --engine tune \
        --datasets ${dataset} \
        --held_out_dataset ${HOLD_OUT_DATASET} \
        --incremental data \
        --optimizer adamw \
        --lr 1e-5 \
        --weight_decay 0 \
        --n_epochs 100 \
        --batch_size 64 \
        --warmup_length 100 \
        --csv_file data_incremental.csv \
        --exp_name clip_finetune_data_incremental
done

