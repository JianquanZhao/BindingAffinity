import os
import sys
import re
import torch
import pickle
import logging
import argparse
import esm
import esm.inverse_folding
import torch.nn.functional as F
import numpy as np
# from utils.parse_args import get_args
from utils.getInterfaceRate import getInterfaceRateAndSeq
from utils.generateGraph import generate_residue_graph
from utils.resFeature import getAAOneHotPhys
from utils.getSASA import getDSSP,getRD
from utils.readFoldX import readFoldXResult
from utils.run_esm import run_esmif1,run_esm1v

import pyrosetta
from pyrosetta import pose_from_sequence
from pyrosetta import pose_from_pdb
from pyrosetta import *
from pyrosetta.teaching import *
pyrosetta.init()


from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO


def get_args():
    """
    :return:进行参数的解析
    """
    description = "you should add those parameter"                   
    parser = argparse.ArgumentParser(description=description)    
    parser.add_argument("--outdir",default="../tmp_jq/",help="Output directory, defaults to tmp")
    parser.add_argument('--device', type=str, default='cuda:1',help='set device')
    parser.add_argument('--inputdir',type=str,default="./data/",help='input data dictionary')
    parser.add_argument('--foldxdir',type=str,default="../foldx/",help='foldx result path')
    parser.add_argument('--dim',type=int,default=256,help='model input dim')
    parser.add_argument('--epoch',type=int,default=3000,help='trainning epoch')
    parser.add_argument('--batch_size',type=int,default=128,help='batch size ')
    parser.add_argument('--epsilon',type=float,default=0.6,help='adv_sample epsilon')
    parser.add_argument('--alpha',type=float,default=0.5,help='the weight in loss ')
    parser.add_argument('--padding',type=int,default=180,help='make feature same length')
    parser.add_argument('--interfacedis',type=float,default=12.0,help='resdisues distance in protein')
    parser.add_argument('--logdir',type=str,default='./log/val',help='log dir,defaults to log')
    parser.add_argument('--datadir',type=str,default='/mnt/data/xukeyu/PPA_Pred/',help='pdb dir')
    parser.add_argument('--featdir',type=str,default="/mnt/data/xukeyu/PPA_Pred/",help='complex graph node feat and edge feat dir')
    parser.add_argument('--modeldir',type=str,default="./models/saved/",help='models dir')
    parser.add_argument('--des', type = str, required = True, 
                        help = 'describe the pdb name, including the chain information, mutation information,eg:1BRS_A_D-EA71F+DD35A')
    parser.add_argument('--pdb_file',type=str,required = True,help='specify the pdb file')
    parser.add_argument('--esm1v_model',type=str, 
                        default =  '/mnt/data/xukeyu/PPA_Pred/feats/esm/esm1v_t33_650M_UR90S_1.pt',
                        help='specify the path of the esm1v model file')
    parser.add_argument('--esmif_model',type=str, 
                        default =  '/mnt/data/xukeyu/PPA_Pred/feats/esm/esm_if1_gvp4_t16_142M_UR50.pt',
                        help='specify the path of the esm inverse folding model file')
    parser.add_argument('--esmif_device', type=str, default = 'cuda:1', help = 'specify the esmif device')
    parser.add_argument('--esm1v_device', type=str, default = 'cuda:0', help = 'specify the esm1v device')
 
    # parser.add_argument('--mutation',type=str, default = '', help='specify the mutation format:ResChainResnumRes,eg:the 26 res Nof chainA mutate to H NA26H, connect  with + when more than 1 mutation')
    args = parser.parse_args()
    return args

def get_complex_mols(args):
    # mols means the chains of the corresponding pdb:1SBB_A_B-LB20T+VB26Y+YB91V
    contents = args.des.split('-')
    chain_list_list = [list(chains) for chains in contents[0].split('_')[1:]]
    pdb_name = contents[0]
    mutation = len(contents) == 1
    mols = {}
    for i,cs in enumerate(chain_list_list):
        chain_index_dict = {c:i for c in cs}
        mols.update(chain_index_dict)
    if mutation:
        mutation = None
    else:
        mutation = contents[-1].split('+')
    return pdb_name, mols, mutation

def generate_graph(args):
    logging.info(f"Generting grapth according to the {args.pdb_file}")
    file_prefix = os.path.basename(args.pdb_file).split('.')[0] + '_'
    graph_file = os.path.join(args.outdir, args.des ,file_prefix + "graph.pkl")
    if os.path.exists(graph_file):
        with open(graph_file, 'rb') as f:
            graph_dict = pickle.load(f)
    else:
        pdb_name, mols, mutation = get_complex_mols(args)
        seq,interfaceDict,connect,dis,pos = getInterfaceRateAndSeq(args.des, args.pdb_file, mols,
                                                                   interfaceDis = args.interfacedis,
                                                                   mutation=mutation)
        with open(graph_file,'wb') as f:
            graph_dict = {"seq":seq, "interfaceDict":interfaceDict, "connect":connect, "dis":dis, "pos":pos}
            pickle.dump(graph_dict, f)
    return graph_dict

@torch.no_grad()
def get_sequence_embedding(args, seq):
    # load esm model
    logging.info("Loading esm1v model...")

    esm1v_model, alphabet = esm.pretrained.load_model_and_alphabet(args.esm1v_model)
    batch_converter = alphabet.get_batch_converter()
    esm1v_model = esm1v_model.eval().to(args.esm1v_device)

    logging.info("Esm1v loaded")
    sequence_embedding = {}
    for k,v in seq.items():
        if  len(v[0])>1024:
            logging.warning(f"Spliting the chain {k} seqeunce {len(v[0])}...")
            first  = v[0][:1000]
            second = v[0][1000:]
            ss = [first,second]
        else:
            ss = [v[0]]
        # get esm1v embedding
        try:
            res=[]
            for s in ss:
                data = [("tmp", s),]
                _, _, batch_tokens = batch_converter(data)
                for i in range(len(s)):
                    batch_tokens_masked = batch_tokens.clone()
                    batch_tokens_masked[0][i+1]=alphabet.mask_idx  #mask the residue,32
                    x = esm1v_model(batch_tokens_masked.to(args.esm1v_device), repr_layers = [33])['representations'][33]
                    # B = 0, T = i + 1
                    res.append(x[0][i+1].tolist())
            sequence_embedding[k] = F.avg_pool1d(torch.tensor(res), 40, 40).cpu()
            logging.info('Finish esm1v\'s embedding : '+k)
        except Exception:
            logging.error("Pdb imformation not clear :"+k)

    return sequence_embedding

@torch.no_grad()
def get_structure_embedding(args, chain_list):
    logging.info("Loading esm1v model...")

    esmif_model, alphabet = esm.pretrained.load_model_and_alphabet(args.esmif_model)
    esmif_model = esmif_model.eval().to(args.esmif_device)

    logging.info("Esmif_model loaded")

    # read the structure
    structure = esm.inverse_folding.util.load_structure(args.pdb_file, chain_list)
    coords, _ = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)

    # get the structure embedding
    try:
        structure_embedding = {chain:F.avg_pool1d(
                                                  esm.inverse_folding.multichain_util.get_encoder_output_for_complex(esmif_model, alphabet, coords, chain)
                                       ,16,16).cpu() for chain in chain_list}
    except Exception:
        logging.error('Embedding structure faild')
    return structure_embedding

def get_resdeepth(args):

    return None

def get_dssp(args):
	

    return None

def get_interface_info():

    return None

def get_sidechain_center(chain_pdb_file, backbone_atoms = ["N", "CA", "C", "O"]):
    logging.info(f"Getting  sidechain coor center from {chain_pdb_file} ")
    parser = PDBParser()
    # structure = parser.get_structure(pdbname, pdbs_path)  # 替换为实际的 PDB 文件路径
    structure = parser.get_structure('test', chain_pdb_file)  # 替换为实际的 PDB 文件路径
    model = structure[0]
    
    for chain in model:
        chain_name = chain.get_id()
        sidechain=[]
        for residue in chain:
            try:
                coords = residue['CA'].get_coord().tolist()
            except:
                for a in backbone_atoms:
                    try:
                        coords = residue[a].get_coord().tolist()
                        break
                    except Exception: 
                        continue
            # 获取残基的侧链原子
            sidechain_atoms = [atom for atom in residue.get_atoms() if atom.get_name() not in backbone_atoms]
            # 计算侧链的重心
            if sidechain_atoms:
                sidechain_center = sum(atom.get_coord() for atom in sidechain_atoms) / len(sidechain_atoms)
                sidechain_center = sidechain_center.tolist()
            else:
                sidechain_center = coords
            coords  += sidechain_center
            sidechain.append(coords)
    return sidechain

def get_angle(pose, index):
    try:
        phi=pose.phi(i)
    except Exception:
        phi=0.0
	
    try:
        psi=pose.psi(i)
    except Exception:
        psi=0.0

    try:
        chi=pose.chi(i)
    except Exception:
        chi=0.0
    return [phi, psi, chi]
def get_sidechain_angle(chain_pdb_file):
    # get the pose through the rosetta and get the angle from pdb file
    # logging.info(f"Getting  sidechain angle phi,psi,chi from {chain_pdb_file}")
    pose = pose_from_pdb(chain_pdb_file)
    res_num = pose.total_residue()
    sidechain_angle = [get_angle(pose, i) for  i in range(1,1 + res_num)]
    return sidechain_angle

def make_chain_pdb(args, chain, chain_pdb_file):
    logging.info(f"Getting  chain {chain} from {args.des} ")
    if not os.path.exists(chain_pdb_file):
        parser = PDBParser(PERMISSIVE=1)
        io = PDBIO()
        structure_id = "test"

    try:
        structure = parser.get_structure(structure_id, args.pdb_file)[0][chain]
        io.set_structure(structure)
        io.save(chain_pdb_file)
    except Exception:
        logging.error(f"False chain id :{chain}")
        # sys.exit(1)

def get_sidechain_info(args, chain):
    logging.info(f"Getting sidechain info of {args.des} chain {chain}")
    file_prefix = os.path.basename(args.pdb_file).split('.')[0] + '_'
    chain_pdb_file = os.path.join(args.outdir, args.des, file_prefix + chain + ".pdb")
    make_chain_pdb(args, chain, chain_pdb_file)
    sidechain_file = os.path.join(args.outdir, args.des, file_prefix + chain + "_sidechain.pkl")
    if os.path.exists(sidechain_file):
        with open(sidechain_file, 'rb')  as f:
            sidechain_info = pickle.load(f)
    else:
        angle = get_sidechain_angle(chain_pdb_file)
        center_coord = get_sidechain_center(chain_pdb_file)
        sidechain_info = {"angle":angle, "center_coord":center_coord}

    return sidechain_info

def get_node_features(args):
    file_prefix = os.path.basename(args.pdb_file).split('.')[0] + '_'
    logging.info(f"Getting node features")
    pdb_name, mols, mutation = get_complex_mols(args)

    logging.info(f"Getting residue deepth, dssp, and the residue physical feature")
    rd   = getRD(args.pdb_file)
    dssp = getDSSP(args.pdb_file)
    resfeat = getAAOneHotPhys()

    graph_dict = generate_graph(args)
    chain_list = list(graph_dict["interfaceDict"].keys())

    # strucutre and sequence embedding
    esm_embedding_file = os.path.join(args.outdir, args.des ,file_prefix +  "esm.pkl")
    if os.path.exists(esm_embedding_file):
        with open(esm_embedding_file, 'rb') as f:
            esm_embedding = pickle.load(f)
        structure_embedding = esm_embedding["structure_embedding"]
        sequence_embedding  = esm_embedding["sequence_embedding"]
    else:
        # structure embedding using esm inverse folding
        structure_embedding = get_structure_embedding(args, chain_list)
        # sequence embedding using esm1v
        sequence_embedding  = get_sequence_embedding(args, graph_dict["seq"])
        esm_embedding = {"structure_embedding":structure_embedding, "sequence_embedding":sequence_embedding}
        with open(esm_embedding_file, 'wb') as f:
            esm_embedding = pickle.dump(esm_embedding, f)

    sidechain_info = {}
    node_features = {}
    rotation = []
    logging.info(f"Getting the strucutre node features")
    for chain in chain_list:
        logging.info(f"Getting chain:{chain} node features")
        # side chain feature
        seq_chain = graph_dict['seq'][args.des + '_' + chain][0]
        idx_map = {}
        ss = 0
        for o in range(len(seq_chain)):
            if seq_chain[o] != 'X':
                idx_map[o] = ss
                ss += 1
        sidechain_info[chain] = get_sidechain_info(args, chain)

        # interface residues feature
        s = graph_dict["seq"][args.des + '_' + chain][1] # chain count?

        # sequence embedding
        esmif_feat = structure_embedding[chain]
        esm1v_feat = sequence_embedding[args.des + '_' + chain]
        
        interface_res_list = graph_dict["interfaceDict"][chain]
        for res in interface_res_list:
            rotation.append(graph_dict["pos"][res].tolist())
            binary = format(mols[chain],'06b')
            chain_sign = [int(bit) for bit in binary] # 2^6
            reduise = res.split('_')[1]
            index = int(reduise[1:]) - int(s)
            res_key = f"{chain}_{index+1}" # chain+'_'+ str(index+1)
            if dssp is None or res_key not in dssp.keys(): 
                dssp_feat=[0.0,0.0,0.0,0.0,0.0]
            else:
                default_value = lambda d,r,i:d[r][i] if d[r][i] != 'NA' else 0.0
                dssp_feat=[default_value(dssp, res_key, 3),
                           default_value(dssp, res_key, 7),
                           default_value(dssp, res_key, 9),
                           default_value(dssp, res_key, 11),
                           default_value(dssp, res_key, 13),
                           ]#[rSA,...]
            if rd is None or res_key not in rd.keys():
                rd_feat=[0.0,0.0]
            else:
                rd_feat = [rd[res_key][0] if rd[res_key][0] is not None else 0.0,
                           rd[res_key][1] if rd[res_key][1] is not None else 0.0]

            node_features[res]=[chain_sign,
                               resfeat[reduise[0]],
                               rd_feat,
                               dssp_feat,
                               esmif_feat[idx_map[index]].tolist(),
                               esm1v_feat[idx_map[index]].tolist(),
                               sidechain_info[chain]["angle"][idx_map[index]] if sidechain_info[chain]["angle"] is not None else [0.0,0.0,0.0,0.0,0.0,0.0],
                               sidechain_info[chain]["center_coord"][idx_map[index]][:6],
                                ]
            rotation.append(graph_dict["pos"][res].tolist())

    return node_features, rotation, graph_dict

def get_features(args):
    tmp_dir = os.path.join(args.outdir, args.des)
    logging.info(f"Make tmp dir:{tmp_dir}")
    os.makedirs(tmp_dir, exist_ok=True)

    file_prefix = os.path.basename(args.pdb_file).split('.')[0] + '_'
    features_file = os.path.join(args.outdir, args.des, file_prefix + 'features.pkl' )
    if os.path.exists(features_file):
        logging.info(f"Loading existed feature from:{features_file}")
        with open(features_file, 'rb') as f:
            features  = pickle.load(f)
    else:
        logging.info(f"Getting feature from the pdbs file...")
        node_features, rotation, graph_dict = get_node_features(args)
        node_features, edge_index,edge_attr = generate_residue_graph(args.des, node_features, graph_dict["connect"], graph_dict["pos"], args.padding)
        features = {'x':torch.tensor(node_features, dtype=torch.float32), 
                	'edge_index':torch.tensor(edge_index,dtype=torch.long).t().contiguous(),
					'edge_attr' :torch.tensor(edge_attr,dtype=torch.float32),
					'rotation'  :torch.tensor(rotation,dtype=torch.float32)
			   		}
        logging.info(f"Save feature into :{features_file}")
        with open(features_file, 'wb') as f:
            pickle.dump(features, f)
    return features

if __name__ == '__main__':
    args = get_args()
    get_features(args)
