# ProAffinity++
---

ProAffinity++ is a protein-protein binding affinity prediction method that superior to other methods, like FoldX, Prodigy, IsLand, PIPR, and PPI-Affinity.
ProAffinity++ extract features from protein sequence and structure information, then use mpnn network to update the representation of the features, finally infer the binding affinity from the representation.![overview](https://i-blog.csdnimg.cn/direct/b5a6bd3d45d44901bf095c5a82466834.jpeg#pic_center)

**Installation**

---

Please follow these steps:
1. clone this rep
```bash
git clone git@github.com:JianquanZhao/BindingAffinity.git
```

2. create the environment

```bash
cd BindingAffinity
conda env create -f environment.yml
```

3. install esm package

```bash
# install sepcial required packages from special source
conda activate PPA_Pred
pip install git+https://github.com/facebookresearch/esm.git
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.12.0+10.2.html
```
4. download the esm1v model and the esm-inversefolding model
```bash
mkdir ckpts
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt -O ckpts/esm1v_t33_650M_UR90S_1.pt -c
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt -O ckpts/esm_if1_gvp4_t16_142M_UR50.pt -c 
```

**running your first prediction**

---
Attention:
1. Please specify the des of the prediction that you want to predict the bindding affinity in this format PDBID_CHAIN1_CHAIN2
2. If the pdb including the mutation, please specify the mutaion type with the formate MutatedresChainResidOrigainalres (eg:VB26Y) and specify the des as PDBID_CHAIN1_CHAIN2-MUTATION1+MUTATION2+MUTATIONn
```bash
python -W ignore predict_affinity.py \
	   --des 1SBB_A_B-LB20T+VB26Y+YB91V \
	   --origin_pdb_file ./example/1SBB_1.pdb \
	   --mutate_pdb_file ./example/WT_1SBB_1.pdb \
	   --model ./ckpts/ProAffinity++.pt \
	   --esmif_model ./ckpts/esm_if1_gvp4_t16_142M_UR50.pt \
	   --esm1v_model ./ckpts/esm1v_t33_650M_UR90S_1.pt \
	   --dim 133
```

## Citing this work


---
If you use the code, please cite:
```bash

```
