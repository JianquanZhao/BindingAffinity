#生成esm1v的信息
import re
import os
import gc
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import warnings
warnings.filterwarnings("ignore")
import esm,torch
import logging
from Bio.PDB.PDBParser import PDBParser
import esm.inverse_folding

def run_esm1v(esm1v_model,batch_converter,alphabet,seq,out_path,device='cuda:2'):
    # esm1v_model_location = '/mnt/data/xukeyu/PPA_Pred/feats/esm/esm1v_t33_650M_UR90S_1.pt'
    # esm1v_model, alphabet = esm.pretrained.load_model_and_alphabet(esm1v_model_location)
    # batch_converter = alphabet.get_batch_converter()
    # esm1v_model.to(device)
    # for param in esm1v_model.parameters():
    #     param.requires_grad = False
    # esm1v_model = esm1v_model.eval()
    # logging.info("esm1v esmif1_model load finish")
                    
    for k,v in seq.items():
        if  len(v[0])>1024:
            logging.warning(k+' seq is longger than 1024.')
            first = v[0][:1000]
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
                    with torch.no_grad():
                        x=esm1v_model(batch_tokens_masked.to(device))
                    res.append(x[0][i+1].tolist())
            res=torch.tensor(res)
            logging.info('Finish esm1v\'s embedding : '+k)
            torch.save(res,out_path+k+'.pth')
        except Exception:
            print("pdb imformation not clear :"+k)
    
    
def run_esmif1(esmif1_model,alphabet,pdbname,pdbs_path,chains,out_path,device='cuda:3'):
    # esmif1_model_location = '/mnt/data/xukeyu/PPA_Pred/feats/esm/esm_if1_gvp4_t16_142M_UR50.pt'
    # esmif1_model, alphabet = esm.pretrained.load_model_and_alphabet(esmif1_model_location)
    # for param in esmif1_model.parameters():
    #     param.requires_grad = False
    # esmif1_model = esmif1_model.eval()
    # esmif1_model.to(device)    

    parser = PDBParser()
    structure = parser.get_structure("temp", pdbs_path)[0]
    structure = esm.inverse_folding.util.load_structure(pdbs_path, chains)
    coords, _ = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)

    try:
        for chain_id in chains:
            rep = esm.inverse_folding.multichain_util.get_encoder_output_for_complex(esmif1_model, alphabet, coords, chain_id, device)
            logging.info('Finish esmif1\'s embedding : '+pdbname+'_'+chain_id)
            print(rep.shape)
            torch.save(rep.to(torch.device('cpu')),out_path+pdbname+'_'+chain_id+'.pth')
            del rep 
            gc.collect()
    except Exception:
        print('pdb imformation not clear '+pdbname)
    


if __name__ == '__main__':
    device = 'cuda:1'
    dataset_path = '/mnt/data/xukeyu/PPA_Pred/BindingAffinity/data/dg_data/test.txt'
    seqdata_path = '/mnt/data/xukeyu/PPA_Pred/BindingAffinity/data/dg_data/test.txt'
    pdbs_path = '/mnt/data/xukeyu/PPA_Pred/data/renumbered_pdb/'
    esm1v_out_path = '/mnt/data/xukeyu/PPA_Pred/feats/esm/pdbbind/esm1v/'
    esmif1_out_path = '/mnt/data/xukeyu/PPA_Pred/feats/esm/pdbbind/esmif1/'
    esm1v_model_location = '/mnt/data/xukeyu/PPA_Pred/feats/esm/esm1v_t33_650M_UR90S_1.pt'
    esmif1_model_location = '/mnt/data/xukeyu/PPA_Pred/feats/esm/esm_if1_gvp4_t16_142M_UR50.pt'
    pdbchains_info_path = '/mnt/data/xukeyu/PPA_Pred/BindingAffinity/data/dg_data/pdb_mols.txt'
    # run_esmif1(esmif1_model_location,dataset_path,pdbs_path,pdbchains_info_path,esmif1_out_path,device)
    run_esm1v(esm1v_model_location,seqdata_path,esm1v_out_path,device)