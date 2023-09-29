#!/bin/bash

for img in datasets/Segmentation/images/valid/*.jpg; do
    # Extract filename without path to use in output
    filename="segmentation/valid/$(basename "$img" .jpg)"

    # Run the python script with current image
    python grounded_sam_demo.py \
        --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint groundingdino_swint_ogc.pth \
        --sam_checkpoint sam_vit_h_4b8939.pth \
        --input_image "$img" \
        --output_dir "outputs/$filename" \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt "fish" \
        --device "cuda"
done
