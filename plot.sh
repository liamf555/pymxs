apptainer exec --nv \
-B /home/liamf/pymxs_sbx/pymxs:/app \
-B /home/liamf/pymxs_sbx/thesis_data:/app/thesis_data \
pymxs_sbx.sif \
python3 /app/plotting/unified_plot.py 2023-10-21T11-53-02-052701-YK3MB  \
-d /app/thesis_data/land_droq/ \
--save \
--env land