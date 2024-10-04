#!/bin/bash
SOURCE_DIR='/data/owcl_data'
DEST_DIR='/data/owcl_data/temporal'

# Define the arrays
datasets=("CIFAR100" "SUN397" "EuroSAT" "OxfordIIITPet" "Flowers102" "FGVCAircraft" "StanfordCars" "Food101")
length=${#datasets[@]}

for (( i=0; i<$length; i++ )); do
    python encode_features/encode_clip_intermediate_features.py \
        --backbone ViT-B/32 \
        --datasets ${datasets[$i]} \
        --data_root ${SOURCE_DIR} \
        --store_folder ${DEST_DIR} \
        --subsets train_test
done
