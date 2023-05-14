py-spy record -o profile.svg --format speedscope -- python3 pymxs_sbx_box.py \
    --algo ppo \
    --training_render_mode ansi \
    --frame_skip 10 \
    --env "gym_mxs/MXSBox2DLidar-v0" \
    --dryden "light" \
    --n_vec_env 12 \
    -s 1000000

# py-spy record -o profile.svg --format speedscope -- python3 pymxs_sbx_box.py \
#     --algo droq \
#     --training_render_mode ansi \
#     --frame_skip 10 \
#     --env gym_mxs/MXSBox2DLidar-v0 \
#     --n_vec_env 1 \
#     --gradient_steps 20 \
#     -s 1000000

# python pymxs_sbx_box.py \
#     --algo ppo \
#     --training_render_mode ansi \
#     --frame_skip 10 \
#     --env "gym_mxs/MXSBox2D-v0" \
#     --n_vec_env 12 \
#     -s 4000000

# python pymxs_sbx_box.py \
#     --algo droq \
#     --training_render_mode ansi \
#     --frame_skip 10 \
#     --env gym_mxs/MXSBox2D-v0 \
#     --n_vec_env 1 \
#     --gradient_steps 20 \
#     -s 1000000


# python pymxs_sbx_box.py \
#     --algo ppo \
#     --training_render_mode ansi \
#     --frame_skip 5 \
#     --n_vec_env 12 \
#     -s 2000000

# python pymxs_sbx_box.py \
#     --algo ppo \
#     --training_render_mode ansi \
#     --frame_skip 10 \
#     --n_vec_env 1 \
#     -s 2000000

# python pymxs_sbx_box.py \
#     --algo tqc \
#     --training_render_mode ansi \
#     --frame_skip 10 \
#     --n_vec_env 1 \
#     -s 1000000

# python pymxs_sbx_box.py \
#     --algo sac \
#     --training_render_mode ansi \
#     --frame_skip 10 \
#     --n_vec_env 1 \
#     -s 1000000

# python pymxs_sbx_box.py \
#     --algo droq \
#     --training_render_mode ansi \
#     --frame_skip 10 \
#     --env gym_mxs/MXSBox2DLidar-v0 \
#     --n_vec_env 1 \
#     --gradient_steps 20 \
#     -s 500000

