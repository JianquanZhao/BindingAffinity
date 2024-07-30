import Bio.PDB
from Bio.PDB.PDBParser import PDBParser
import networkx as nx
import numpy as np
import scipy.sparse as sp
import logging
import torch
import math
from utils.rotation import vec2rotation
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def generate_atom_graph(pdb_file):
    # 使用BioPython库中的PDBParser来解析PDB文件
    parser = Bio.PDB.PDBParser()
    structure = parser.get_structure('protein', pdb_file)

    # 使用NetworkX库来创建空的无向图
    graph = nx.Graph()

    # 遍历PDB文件中的所有原子
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # 获取原子的坐标
                    x, y, z = atom.get_coord()

                    # 添加原子到图中
                    graph.add_node(atom.get_serial_number(), x=x, y=y, z=z)

                    # 遍历相邻的原子，并添加到图中
                    for neighbor in atom.get_neighbors():
                        graph.add_edge(atom.get_serial_number(), neighbor.get_serial_number())

    return graph

def generate_residue_graph(pdb_file,featuredict,connect,pos,padding):
    # 初始化图
    graph = nx.Graph()
        
    for k,v in featuredict.items():
        emb=[]
        for l in v:
            emb=emb+l
        # print(k+':'+str(len(emb)))
        graph.add_node(k,embedding=emb)

    for k,v in connect.items():
        if k not in featuredict.keys():
            continue
        for x in v:
            resName=x.split("=")[0]
            if resName not in featuredict.keys():
                continue
            distance = (float)(x.split("=")[1])
            
            if distance < 0:
                chain_type = [0,1]
            else:
                chain_type = [1,0]
            rotation = torch.flatten(pos[k][:3,:3]).tolist()
            edge_attr = chain_type+[math.fabs(distance)]+rotation
            if k not in featuredict.keys():
                continue
            graph.add_edge(k,resName,weight=edge_attr,undirected=True)

    # 输出图的基本信息
    logging.info(nx.info(graph))
    
    return graph_to_input(nx_graph=graph)

def graph_to_input(nx_graph):
    # 获取节点数量
    g_nodes = nx_graph.nodes()
    
    # 将字符串标识符映射到整数节点编号
    node_dict = {node: i for i, node in enumerate(nx_graph.nodes)}

    # 获取节点特征矩阵
    node_features = []
    for nodeName in g_nodes:
        node_features.append(nx_graph.nodes[nodeName]['embedding'])
    node_features = np.array(node_features)
    
    edge_index = []
    edge_attr = []
    for u, v, attr in nx_graph.edges(data=True):
        u, v = node_dict[u], node_dict[v]
        edge_index.append([u, v])
        edge_index.append([v, u])
        edge_attr.append(attr['weight'])
        edge_attr.append(attr['weight'])

    # 获取邻接矩阵
    # adj = nx.to_numpy_array(nx_graph) 
    
    # edge_attr_dict = {(u, v): torch.tensor(nx_graph[u][v]['weight']) for u, v in nx_graph.edges}
    
    return node_features, edge_index, edge_attr
