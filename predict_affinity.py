import os
import pickle
import logging
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from make_feature import get_features
from models.affinity_net_mpnn import Net

def get_args():
    description = "you should add those parameter"                   
    parser = argparse.ArgumentParser(description=description)    

    # the proaffinity++ model config
    parser.add_argument('--dim',     type = int,   default  = 256,      help = 'model input dim')
    parser.add_argument('--epsilon', type = float, default  = 0.6,      help = 'adv_sample epsilon')
    parser.add_argument('--alpha',   type = float, default  = 0.5,      help = 'the weight in loss ')
    parser.add_argument('--device',  type = str,   default  = 'cuda:1', help = 'set device')
    parser.add_argument('--model',   type = str,   required = True,     help = 'model dir',)

    # the input pdb and the process config
    parser.add_argument("--outdir",          type = str,    default  = "./tmp/",    help = "Output directory, defaults to tmp")
    parser.add_argument('--foldxdir',        type = str,    default  = "./foldx/",  help = 'foldx result path')
    parser.add_argument('--padding',         type = int,    default  = 180,         help = 'make feature same length')
    parser.add_argument('--interfacedis',    type = float,  default  = 12.0,        help = 'resdisues distance in protein')
    parser.add_argument('--pdb_file',        type = str,    default  = None,        help = 'specify the pdb file')
    parser.add_argument('--mutate_pdb_file', type = str,    default  = None,        help = 'specify the mutated pdb file')
    parser.add_argument('--origin_pdb_file', type = str,    required = True,        help = 'specify the pdb file')
    parser.add_argument('--des',             type = str,    required = True,        help = 'describe the protein,eg:1BRS_A_D-EA71F+DD35A')

    # the process tool config
    parser.add_argument('--esmif_device', type = str, default = 'cuda:1', help = 'specify the esmif device')
    parser.add_argument('--esm1v_device', type = str, default = 'cuda:0', help = 'specify the esm1v device')
    parser.add_argument('--esm1v_model',  type = str, 
                        default = '../models/esm1v_t33_650M_UR90S_1.pt',
                        help='specify the path of the esm1v model file')
    parser.add_argument('--esmif_model',  type=str, 
                        default = '../models/esm_if1_gvp4_t16_142M_UR50.pt',
                        help='specify the path of the esm inverse folding model file')
    args = parser.parse_args()
    return args

def load_model(args):
    device = torch.device(args.device)
    net    = Net(input_dim = args.dim, hidden_dim = 64, output_dim = 64)
    ckpt   = torch.load(args.model)
    net.load_state_dict(ckpt)# .eval().to(device)
    return net.eval().to(device)

def preprocess_input(args):
    # get feature from the input structure/sequence
    features = get_features(args)
    data = Data(x = features["x"].to(args.device),
                edge_index = features["edge_index"].to(args.device),
                edge_attr  = features["edge_attr"].to(args.device),
                energy     = torch.tensor(0.).to(args.device))
    return data

@torch.no_grad()
def predict_deltaG(model, data):
    affinity = model(data, data.x)
    return affinity

def predict_delta_deltaG(model, feature_original, feature_mutation):
    affinity_original = predict_deltaG(model, feature_original)
    affinity_mutation = predict_deltaG(model, feature_mutation)
    return affinity_original, affinity_mutation, affinity_mutation - affinity_original


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    assert 'cpu' in args.device or 'cuda' in args.device,f"Illegal device {args.device}"
    model = load_model(args)

    # precess featureas
    args.pdb_file = args.origin_pdb_file
    feature_original = preprocess_input(args)

    if args.mutate_pdb_file is not None:
        args.pdb_file = args.mutate_pdb_file
        feature_mutation = preprocess_input(args)
        affinity_original, affinity_mutation, delta_deltaG = predict_delta_deltaG(model, feature_original, feature_mutation)
        print(f"original_affinity mutation_affinity  DeltaDeltaG")
        print(f"{affinity_original.cpu().item():.3f} {affinity_mutation.cpu().item():.3f} {delta_deltaG.cpu().item():.3f}")
    else:
        affinity_original = predict_deltaG(model, feature_original)
        print(f"original affinity")
        print(f"{affinity_original.cpu().item():.3f}")
