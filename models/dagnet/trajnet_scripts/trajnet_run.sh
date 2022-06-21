MODES=3

# Train the model 
python3 trajnet_train.py \
    --graph_model gcn --run trajnet_run \
    --dataset_name colfree_trajdata --obs_len 9 --num_epochs 500 \
    --fill_missing_obs 0 --keep_single_ped_scenes 0 --batch_size 16

# Evaluate on Trajnet++ 
python -m trajnet_evaluator \
    --graph_model gcn --run trajnet_run \
    --dataset_name colfree_trajdata --write_only --modes ${MODES} \
    --fill_missing_obs 1 --keep_single_ped_scenes 1 --batch_size 1 --n_jobs 1
    