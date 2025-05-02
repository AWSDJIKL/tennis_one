CUDA_VISIBLE_DEVICES=0 torchrun --master_port=9909 --nproc_per_node=1 tools/train.py -c ./configs/rtdetrv2/rtdetrv2_r18vd_sp3_120e_coco.yml -r ./output/rtdetrv2_r18vd_sp3_120e_coco/checkpoint0119.pth --test-only
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=9909 --nproc_per_node=1 tools/train.py -c ./configs/rtdetrv2/rtdetrv2_r18vd_sp3_120e_coco.yml --use-amp --seed=0 &> log.txt 2>&1 &
python references/deploy/rtdetrv2_torch.py -c ./configs/rtdetrv2/rtdetrv2_r18vd_sp3_120e_coco.yml -r ./output/rtdetrv2_r18vd_sp3_120e_coco/best.pth --im-file=xxx --device=cuda:0

python references/deploy/rtdetrv2_video.py -c ./configs/rtdetrv2/rtdetrv2_r18vd_sp3_120e_coco.yml -r ./output/rtdetrv2_r18vd_sp3_120e_coco/best.pth


# yolo11
python references/deploy/yolo11_print_pose.py

# hot
python demo/vis.py --video ./demo/crop_test.mp4