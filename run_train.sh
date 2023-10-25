# # 以下是载入已有模型重训练的示例
python tools/train_dil.py -f cfgs/streamnet_s \
                          -c /home/xiang.huang/project/StreamDynamicTest/data/output/streamnet_s_2023102301/epoch_2_ckpt.pth \
                          -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
                          --resume \
                          --experiment-name streamnet_s_20231025 \
                          --eval-batch-size 8 \
                          -d 2 -b 4 --fp16

# python tools/train_dil.py -f cfgs/streamnet_s \
#                           -c ./models/coco_pretrained_models/yolox_s_drfpn.pth \
#                           -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
#                           --experiment-name streamnet_s_20231025 \
#                           --eval-batch-size 8 \
#                           -d 4 -b 4 --fp16

# python tools/train_dil.py -f cfgs/streamnet_m \
#                           -c ./models/coco_pretrained_models/yolox_m_drfpn.pth \
#                           -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
#                           --experiment-name streamnet_m \
#                           -d 4 -b 16 --fp16

# python tools/train_dil.py -f cfgs/streamnet_l \
#                           -c ./models/coco_pretrained_models/yolox_l_drfpn.pth \
#                           -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
#                           --experiment-name streamnet_l \
#                           -d 4 -b 8 --fp16

# python tools/train.py -f cfgs/streamnet_l_1200x1920 \
#                       -c ./models/coco_pretrained_models/yolox_l_drfpn.pth \
#                       --experiment-name streamnet_l_1200x1920 \
#                       -d 4 -b 8 --fp16
