python tools/eval.py -f cfgs/streamnet_s \
                     -c data/output/streamnet_s_20231019/best_ckpt.pth \
                     --experiment-name streamnet_s_20231019 \
                     -d 4 -b 8 --conf 0.01 --fp16

# python tools/eval.py -f cfgs/streamnet_m \
#                      -c models/checkpoints/streamnet_m.pth \
#                      -d 4 -b 32 --conf 0.01 --fp16

# python tools/eval.py -f cfgs/streamnet_l \
#                      -c models/checkpoints/streamnet_l.pth \
#                      -d 4 -b 8 --conf 0.01 --fp16

# python tools/eval.py -f cfgs/streamnet_l_1200x1920 \
#                      -c models/checkpoints/streamnet_l_1200x1920.pth \
#                      -d 4 -b 8 --conf 0.01 --fp16

# # 场景实验
# python tools/eval.py -f cfgs/streamnet_s_scene \
#                      -c models/checkpoints/streamnet_s.pth \
#                      -d 1 -b 32 --conf 0.01 --fp16
# 
# python tools/eval.py -f cfgs/streamnet_m_scene \
#                      -c models/checkpoints/streamnet_m.pth \
#                      -d 1 -b 32 --conf 0.01 --fp16
# 
# python tools/eval.py -f cfgs/streamnet_l_scene \
#                      -c models/checkpoints/streamnet_l.pth \
#                      -d 1 -b 8 --conf 0.01 --fp16
# 
# python tools/eval.py -f cfgs/streamnet_l_1200x1920_scene \
#                      -c models/checkpoints/streamnet_l_1200x1920.pth \
#                      -d 1 -b 8 --conf 0.01 --fp16

# # 新的验证集
# python tools/eval.py -f cfgs/streamnet_s_scene_new \
#                      -c models/checkpoints/streamnet_s.pth \
#                      -d 1 -b 32 --conf 0.01 --fp16
#
# python tools/eval.py -f cfgs/streamnet_m_scene_new \
#                      -c models/checkpoints/streamnet_m.pth \
#                      -d 1 -b 32 --conf 0.01 --fp16
#
# python tools/eval.py -f cfgs/streamnet_l_scene_new \
#                      -c models/checkpoints/streamnet_l.pth \
#                      -d 1 -b 8 --conf 0.01 --fp16
#
# python tools/eval.py -f cfgs/streamnet_l_1200x1920_scene_new \
#                      -c models/checkpoints/streamnet_l_1200x1920.pth \
#                      -d 1 -b 8 --conf 0.01 --fp16
