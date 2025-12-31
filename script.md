# test the environment
```bash
./docker_run.sh python gen_bounding-box.py \
    --input_dir fake_images \
    --txt_output_dir fake_labels \
    --img_output_dir fake_detections \
    --save_txt True \
    --save_img False \
    --text_prompt person,vehicle,engineering machinery \
    --ignore_label building \
    --box_threshold 0.45 \
    --text_threshold 0.25
```