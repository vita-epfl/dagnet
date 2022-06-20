import os
import argparse
import pickle

from joblib import Parallel, delayed
import scipy
import torch
from tqdm import tqdm
import trajnetplusplustools
import numpy as np

from evaluator.trajnet_evaluator import trajnet_evaluate
from evaluator.write_utils import \
    load_test_datasets, preprocess_test, write_predictions

from trajnet_loader import trajnet_loader

# ==============
# === DAGnet ===
# Add the dagnet directory to the path
import random
import sys 
import pathlib
ROOT_DIR = pathlib.Path('.').absolute().parent.parent
print('ROOT DIR = ', ROOT_DIR)
sys.path.append(str(ROOT_DIR))
BASE_DIR = ROOT_DIR / 'runs' / 'dagnet'

from models.dagnet.model import DAGNet
from models.utils.adjacency_matrix import \
    compute_adjs_distsim, compute_adjs_knnsim, compute_adjs
from models.utils.utils import relative_to_abs, to_goals_one_hot
# ==============



def predict_scene(model, batch, args):
    assert len(batch) == 9
    batch = [tensor.cuda() for tensor in batch]
    (
        obs_traj, pred_traj_gt, 
        obs_traj_rel, pred_traj_gt_rel,
        obs_goals, pred_goals_gt, 
        non_linear_ped, 
        loss_mask, 
        seq_start_end
    ) = batch 

    # Goals one-hot encoding
    obs_goals_ohe = to_goals_one_hot(obs_goals, args.g_dim).cuda()

    # Adj matrix for current batch
    if args.adjacency_type == 0:
        adj_out = compute_adjs(args, seq_start_end).cuda()
    elif args.adjacency_type == 1:
        adj_out = compute_adjs_distsim(
            args, seq_start_end, 
            obs_traj.detach().cpu(), pred_traj_gt.detach().cpu()
            ).cuda()
    elif args.adjacency_type == 2:
        adj_out = compute_adjs_knnsim(
            args, seq_start_end, 
            obs_traj.detach().cpu(), pred_traj_gt.detach().cpu()
            ).cuda()   

    # Find the distribution by passing through the model
    _, _, _, h = model(
        obs_traj, obs_traj_rel, obs_goals_ohe, seq_start_end, adj_out
        )
    
    # Get the predictions and save them
    multimodal_outputs = {}
    for num_p in range(args.modes):
        # Sample one trajectory (per pedestrian)
        pred_traj_fake_rel = model.sample(
            args.pred_len, h, obs_traj[-1], obs_goals_ohe[-1], seq_start_end
            )
        
        # Convert to absolute coordinates
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
        # Samples are of shape [T, num_peds, 2]

        output_primary = pred_traj_fake[:, 0]
        output_neighs = pred_traj_fake[:, 1:]
        multimodal_outputs[num_p] = [output_primary, output_neighs]

    return multimodal_outputs 


def load_predictor(args):
    print(f'Loading checkpoint at path {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint)
    n_max_agents = checkpoint['n_max_agents']

    args_cp = checkpoint['args']  
    for key in args_cp:
        if key not in args:
            args.__dict__[key] = args_cp[key]

    model = DAGNet(args, n_max_agents).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def get_predictions(args):
    """
    Get model predictions for each test scene and write the predictions 
    in appropriate folders.
    """
    # List of .json file inside the args.path 
    # (waiting to be predicted by the testing model)
    datasets = sorted([
        f.split('.')[-2] for f in os.listdir(args.path.replace('_pred', '')) \
        if not f.startswith('.') and f.endswith('.ndjson')
        ])

    # Extract Model names from arguments and create its own folder 
    # in 'test_pred' for storing predictions
    # WARNING: If Model predictions already exist from previous run, 
    # this process SKIPS WRITING
    for model in args.output:
        model_name = model.split('/')[-1].replace('.pkl', '')
        model_name = model_name + '_modes' + str(args.modes)

        ## Check if model predictions already exist
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        if not os.path.exists(args.path + model_name):
            os.makedirs(args.path + model_name)
        else:
            print(f'Predictions corresponding to {model_name} already exist.')
            print('Loading the saved predictions')
            continue

        print("Model Name: ", model_name)
        model = load_predictor(args)
        goal_flag = False

        # Iterate over test datasets
        for dataset in datasets:
            # Load dataset
            dataset_name, scenes, scene_goals = \
                load_test_datasets(dataset, goal_flag, args)

            # Convert it to a trajnet loader
            scenes_loader = trajnet_loader(
                scenes, 
                args, 
                drop_distant_ped=False, 
                test=True,
                keep_single_ped_scenes=args.keep_single_ped_scenes,
                fill_missing_obs=args.fill_missing_obs
                ) 

            # Can be removed; it was useful for debugging
            scenes_loader = list(scenes_loader)

            # Get all predictions in parallel. Faster!
            scenes_loader = tqdm(scenes_loader)
            pred_list = Parallel(n_jobs=args.n_jobs)(
                delayed(predict_scene)(model, batch, args)
                for batch in scenes_loader
                )
            
            # Write all predictions
            write_predictions(pred_list, scenes, model_name, dataset_name, args)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()

    # =================
    # === TrajNet++ ===
    parser.add_argument("--dataset_name", default="colfree_trajdata", type=str)
    parser.add_argument("--obs_len", default=9, type=int)
    parser.add_argument("--pred_len", default=12, type=int)
    parser.add_argument("--fill_missing_obs", default=1, type=int)
    parser.add_argument("--keep_single_ped_scenes", default=1, type=int)
    parser.add_argument('--modes', default=1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--n_jobs", default=8, type=int)
    parser.add_argument('--write_only', action='store_true')
    
    parser.add_argument('--normalize_scene', action='store_true')
    parser.add_argument('--disable-collision', action='store_true')
    parser.add_argument("--delim", default="\t")
    parser.add_argument("--loader_num_workers", default=4, type=int)
    parser.add_argument("--skip", default=1, type=int)
    parser.add_argument("--sample", type=float, default=1.0)
    # =================

    # ==============
    # === DAGnet ===

    # Dataset options
    parser.add_argument('--num_workers', default=4, type=int, required=False)
    parser.add_argument('--n_cells_x', default=32, type=int, required=False)
    parser.add_argument('--n_cells_y', default=30, type=int, required=False)
    parser.add_argument('--goals_window', type=int, default=3, required=False)
    
    # Model
    parser.add_argument('--clip', default=10, type=int, required=False)
    parser.add_argument('--n_layers', default=2, type=int, required=False)
    parser.add_argument('--x_dim', default=2, type=int, required=False)
    parser.add_argument('--h_dim', default=64, type=int, required=False)
    parser.add_argument('--z_dim', default=32, type=int, required=False)
    parser.add_argument('--rnn_dim', default=64, type=int, required=False)

    # Graph
    parser.add_argument(
        '--graph_model', type=str, required=True, 
        choices=['gat','gcn'], help='Graph type'
        )
    parser.add_argument(
        '--graph_hid', type=int, default=8, help='Number of hidden units'
        )
    parser.add_argument(
        '--sigma', type=float, default=1.2, 
        help='Sigma value for similarity matrix'
        )
    parser.add_argument(
        '--adjacency_type', type=int, default=1, choices=[0,1,2], 
        help='Type of adjacency matrix: '
            '0 (fully connected graph),'
            '1 (distances similarity matrix),'
            '2 (knn similarity matrix).'
            )
    parser.add_argument('--top_k_neigh', type=int, default=3)

    # Miscellaneous
    parser.add_argument('--seed', default=128, type=int, required=False)
    parser.add_argument('--run', required=True, type=str, help='Current run name')
    parser.add_argument(
        '--best', default=True, action='store_true', 
        help='Evaluate with best checkpoint'
        )
    parser.add_argument(
        '--epoch', type=int, 
        help='Evaluate with the checkpoint of a specific epoch'
        )
    # ==============
    
    args = parser.parse_args()

    # Set random seed (kept from DAGnet)
    set_random_seed(args.seed)

    scipy.seterr('ignore')

    # Prepare checkpoint path
    curr_run_dir = os.path.join(BASE_DIR, args.run)
    saves_dir = os.path.join(curr_run_dir, 'saves')
    saves_best = os.path.join(saves_dir, 'best')

    if args.best:
        print('Loading best checkpoint')
        args.checkpoint = list(pathlib.Path(saves_best).glob('*.pth'))[-1]
    elif args.epoch is not None:
        print(f'Loading checkpoint from epoch {args.epoch}')
        args.checkpoint = os.path.join(
            saves_dir.absolute(), 'checkpoint_epoch_{}.pth'.format(args.epoch)
            )
        if not pathlib.Path(args.checkpoint).is_file():
            raise(Exception("Couldn't find a checkpoint for the specified epoch"))
    else:
        print('Loading latest checkpoint')
        args.checkpoint = list(pathlib.Path(saves_dir).glob('*.pth'))[-1]

    args.output = [str(args.checkpoint)]
    args.path = os.path.join(
        str(ROOT_DIR), 'datasets', args.dataset_name, 'test_pred/'
        )

    # Adding arguments with names that fit the evaluator module
    # in order to keep it unchanged
    args.obs_length = args.obs_len
    args.pred_length = args.pred_len

    # Writes to Test_pred
    # Does NOT overwrite existing predictions if they already exist ###
    get_predictions(args)
    if args.write_only: # For submission to AICrowd.
        print("Predictions written in test_pred folder")
        exit()

    ## Evaluate using TrajNet++ evaluator
    trajnet_evaluate(args)


if __name__ == '__main__':
    main()


