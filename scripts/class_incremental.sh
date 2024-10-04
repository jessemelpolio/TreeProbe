# CIFAR100,SUN397,EuroSAT,OxfordIIITPet,Flowers102,FGVCAircraft,StanfordCars,Food101
# 20,80,2,8,20,20,40,20
DATA=/data/owcl_data/hdf5
RESULTS_DIR=./results/tmlr
HOLD_OUT_DATASET=ImageNet,UCF101,DTD

# CLIP Zero-shot
for dataset in CIFAR100 SUN397 EuroSAT OxfordIIITPet Flowers102 FGVCAircraft StanfordCars Food101; do
    for num_classes in 20 80 2 8 20 20 40 20; do
        python main_class.py \
            --use_encoded_dataset \
            --data_root ${DATA} \
            --backbone ViT-B/32 \
            --results_dir ${RESULTS_DIR} \
            --save \
            --engine main \
            --datasets ${dataset} \
            --held_out_dataset ${HOLD_OUT_DATASET} \
            --incremental class \
            --mix_mode clip_only \
            --csv_file class_incremental.csv \
            --num_classes ${num_classes} \
            --exp_name clip_zero_shot_class_incremental
    done
done

# CLIP + KNN (AIM-Emb)
for dataset in CIFAR100 SUN397 EuroSAT OxfordIIITPet Flowers102 FGVCAircraft StanfordCars Food101; do
    for num_classes in 20 80 2 8 20 20 40 20; do
        python main_class.py \
            --use_encoded_dataset \
            --data_root ${DATA} \
            --backbone ViT-B/32 \
            --results_dir ${RESULTS_DIR} \
            --save \
            --engine main \
            --datasets ${dataset} \
            --held_out_dataset ${HOLD_OUT_DATASET} \
            --incremental class \
            --mix_mode complementary \
            --buffer_type aim_emb \
            --k 9 \
            --retriever knn \
            --csv_file class_incremental.csv \
            --num_classes ${num_classes} \
            --exp_name clip_knn_class_incremental
    done
done

# LinProbe
for dataset in CIFAR100 SUN397 EuroSAT OxfordIIITPet Flowers102 FGVCAircraft StanfordCars Food101; do
    for num_classes in 20 80 2 8 20 20 40 20; do
        python main_class.py \
            --use_encoded_dataset \
            --data_root ${DATA} \
            --backbone ViT-B/32 \
            --results_dir ${RESULTS_DIR} \
            --save \
            --engine main \
            --datasets ${dataset} \
            --held_out_dataset ${HOLD_OUT_DATASET} \
            --incremental class \
            --mix_mode complementary \
            --buffer_type exemplar_only \
            --k 9 \
            --retriever tree_probe \
            --tree_probe_max_instances 50000000000 \
            --tree_probe_min_samples 1 \
            --csv_file class_incremental.csv \
            --num_classes ${num_classes} \
            --exp_name linear_probe_class_incremental
    done
done


# CLIP + TreeProbe
for dataset in CIFAR100 SUN397 EuroSAT OxfordIIITPet Flowers102 FGVCAircraft StanfordCars Food101; do
    for num_classes in 20 80 2 8 20 20 40 20; do
        python main_class.py \
            --use_encoded_dataset \
            --data_root ${DATA} \
            --backbone ViT-B/32 \
            --results_dir ${RESULTS_DIR} \
            --save \
            --engine main \
            --datasets ${dataset} \
            --held_out_dataset ${HOLD_OUT_DATASET} \
            --incremental class \
            --mix_mode complementary \
            --buffer_type aim_emb \
            --k 9 \
            --retriever tree_probe \
            --tree_probe_max_instances 50000 \
            --tree_probe_min_samples 1 \
            --csv_file class_incremental.csv \
            --num_classes ${num_classes} \
            --exp_name clip_tree_probe_max_instances_50k_class_incremental
    done
done

# CLIP + Linear Probe
for dataset in CIFAR100 SUN397 EuroSAT OxfordIIITPet Flowers102 FGVCAircraft StanfordCars Food101; do
    for num_classes in 20 80 2 8 20 20 40 20; do
        python main_class.py \
            --use_encoded_dataset \
            --data_root ${DATA} \
            --backbone ViT-B/32 \
            --results_dir ${RESULTS_DIR} \
            --save \
            --engine main \
            --datasets ${dataset} \
            --held_out_dataset ${HOLD_OUT_DATASET} \
            --incremental class \
            --mix_mode complementary \
            --buffer_type aim_emb \
            --k 9 \
            --retriever tree_probe \
            --tree_probe_max_instances 50000000000 \
            --tree_probe_min_samples 1 \
            --csv_file class_incremental.csv \
            --num_classes ${num_classes} \
            --exp_name clip_linear_probe_class_incremental
    done
done


# ZSCL
for dataset in CIFAR100 SUN397 EuroSAT OxfordIIITPet Flowers102 FGVCAircraft StanfordCars Food101; do
    for num_classes in 20 80 2 8 20 20 40 20; do
        python main_zscl.py \
            --use_encoded_dataset \
            --data_root ${DATA} \
            --backbone ViT-B/32 \
            --results_dir ${RESULTS_DIR} \
            --save \
            --engine ZSCL \
            --datasets ${dataset} \
            --held_out_dataset ${HOLD_OUT_DATASET} \
            --incremental class \
            --optimizer adamw \
            --lr 1e-5 \
            --weight_decay 0 \
            --n_epochs 100 \
            --batch_size 64 \
            --ref_caption_root /data/conceptual_captions \
            --ref_image_dataset ImageNet \
            --warmup_length 100 \
            --csv_file class_incremental.csv \
            --num_classes ${num_classes} \
            --exp_name zscl_class_incremental
    done
done

# CLIP Fine-Tune
for dataset in CIFAR100 SUN397 EuroSAT OxfordIIITPet Flowers102 FGVCAircraft StanfordCars Food101; do
    for num_classes in 20 80 2 8 20 20 40 20; do
        python main_tune.py \
            --use_encoded_dataset \
            --data_root ${DATA} \
            --backbone ViT-B/32 \
            --results_dir ${RESULTS_DIR} \
            --save \
            --engine tune \
            --datasets ${dataset} \
            --held_out_dataset ${HOLD_OUT_DATASET} \
            --incremental class \
            --optimizer adamw \
            --lr 1e-5 \
            --weight_decay 0 \
            --n_epochs 100 \
            --batch_size 64 \
            --warmup_length 100 \
            --csv_file class_incremental.csv \
            --num_classes ${num_classes} \
            --exp_name clip_finetune_class_incremental
    done
done
