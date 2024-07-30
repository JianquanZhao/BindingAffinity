import pyrosetta;pyrosetta.init()
from pyrosetta import pose_from_sequence
from pyrosetta import *
from pyrosetta.teaching import *
init()
import os
import pickle
import re
import Bio
from Bio.PDB.PDBParser import PDBParser

def  sidechain_center(pdbname,pdbs_path,out_path = None):

    # 定义主链原子的名称
    backbone_atoms = ["N", "CA", "C", "O"]

    print(pdbname)
    parser = PDBParser()
    structure = parser.get_structure(pdbname, pdbs_path)  # 替换为实际的 PDB 文件路径

    # 获取第一个模型
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
        # print('sidechain_center : '+pdbname+'_'+chain_name)
        # with open(out_path+pdbname+'_'+chain_name+'_sidechain_center.picke', 'wb') as file:
        #     pickle.dump(sidechain, file)


def sidechain_angle(seq,out_path = None):
    for k,v in seq.items():
        try :
            pose = pose_from_pdb('/mnt/data/xukeyu/PPA_Pred/data/split_by_chain/'+k+'.pdb')
        except Exception:
            logging.warning('pdb structure has problem!')
        sidechain=[]
        for i in range(1,len(v[0])+1):
            res_sidechain=[]
            try:
                phi=pose.phi(i)
            except Exception :
                phi=0.0
            res_sidechain.append(phi)
            try:
                psi=pose.psi(i)
            except Exception :
                psi=0.0
            res_sidechain.append(psi)
            for j in range(1,5):
                chi=0.0
                try:
                    chi=pose.chi(j,i)
                except Exception :
                    chi=0.0
                res_sidechain.append(chi)
            sidechain.append(res_sidechain)
        # with open(out_path+k+'_sidechain.picke', 'wb') as file:
        #     pickle.dump(sidechain, file)
		return sidechain
        


if __name__ == '__main__':
    dataset_path = '/mnt/data/xukeyu/PPA_Pred/BindingAffinity/data/dg_data/All_set.txt'
    pdbs_path = '/mnt/data/xukeyu/PPA_Pred/data/renumbered_pdb/'
    pdbchains_info_path = '/mnt/data/xukeyu/PPA_Pred/BindingAffinity/data/pdb_mols.txt'
    out_path = '/mnt/data/xukeyu/PPA_Pred/feats/sidechain/'
    seq_path = '/mnt/data/xukeyu/PPA_Pred/BindingAffinity/data/dg_data/antibody_antigen_seq.txt'
    chain_group = {}
    with open(seq_path,'r') as f:
        for line in f:
            b = re.split('\t|\n',line)
            pdbname = b[0].split('_')[0]
            chain = b[0].split('_')[1]
            if pdbname not in chain_group.keys():
                chain_group[pdbname] = [chain]
            else:
                chain_group[pdbname].append(chain)
    # with open(pdbchains_info_path,'r') as f:
    #     for line in f:
    #         b = re.split('\t|\n',line)
    #         chains = set()
    #         for i in range(1,len(b)):
    #             for c in b[i]:
    #                 chains.add(c)
    #         chain_group[b[0]] = chains
    sidechain_center(pdbs_path,out_path)
    # sidechain_angle(seq_path,out_path)
