import math
import torch
from torch import einsum
from einops import rearrange
import torch.nn.functional as F
import numpy as np

def get_rotation(pos,edge_index,dis):
    edge_attr = []
    print(dis)
    for i in range(len(edge_index)):
        A = pos[edge_index[0][i]]
        B = pos[edge_index[1][i]]
        direction_vector = A - B
        # 单位化方向向量
        unit_vector = F.normalize(direction_vector, p=2, dim=0)

        # 计算旋转轴和角度
        rotation_axis = torch.tensor([0.0, 0.0, 1.0])  # 假设绕 Z 轴旋转
        rotation_angle = torch.acos(torch.dot(unit_vector, rotation_axis))

        # 计算旋转矩阵
        rotation_matrix = torch.tensor([[torch.cos(rotation_angle), -torch.sin(rotation_angle), 0.0],
                                    [torch.sin(rotation_angle), torch.cos(rotation_angle), 0.0],
                                    [0.0, 0.0, 1.0]])
        
        rotation_matrix = torch.flatten(rotation_matrix).tolist()
        edge_attr.append([math.fabs(dis[i])]+rotation_matrix)
        print(edge_attr)

def getResRTFeature(res):  #获取某个残基的translation+rotation feature
    CACoor=res["CA"].get_coord()
    try:
        NCoor=res["N"].get_coord()
    except Exception:
        NCoor=CACoor
    
    try:
        CCoor=res["C"].get_coord()
    except Exception:
        try:
            CCoor=res["CB"].get_coord()
        except Exception:
            CCoor = CACoor
            
    return vec2transform(torch.tensor([NCoor,CACoor,CCoor]))       

def vec2rotation(vec):
    """(..., 3, 3) <- (..., 3, 3)
    Return the transform matrix given three points' coordinate."""
    v1 = vec[..., 2, :] - vec[..., 1, :]  # (..., 3)
    v2 = vec[..., 0, :] - vec[..., 1, :]  # (..., 3)
    e1 = v1 / vector_robust_norm(v1, dim=-1, keepdim=True)  # (..., 3)
    u2 = v2 - e1 * rearrange(einsum('...L,...L->...', e1, v2), '...->...()')
    #(..., 3)
    e2 = u2 / vector_robust_norm(u2, dim=-1, keepdim=True)
    e3 = torch.cross(e1, e2, dim=-1)  # (..., 3)
    return torch.stack((e1, e2, e3), dim=-1)  # (B, 3, 3)

def vec2transform(vec):
    """(..., 4, 4) <- (..., 3, 3)"""
    result = torch.zeros(*vec.shape[:-2], 4, 4, device=vec.device)
    result[..., :3, :3] = vec2rotation(vec)
    result[..., :3, 3] = vec[..., 1, :]
    result[..., 3, 3] = 1
    return result

def transform_invert(transform):
    """(..., 4, 4)"""
    result = torch.zeros(*transform.shape[:-2], 4, 4, device=transform.device)
    result[..., :3, :3] = transform[..., :3, :3].transpose(-1, -2)
    result[..., :3, -1] = einsum('...ij,...j->...i',
    -result[..., :3, :3], transform[..., :3, -1])
    # (..., 3, 3) <- (..., 3, 3), (..., 3)
    result[..., 3, 3] = 1
    return result

def vector_robust_norm(vec, epison=1e-8, **kargs):
    """(B, 3)"""
    return torch.linalg.vector_norm(vec, **kargs) + epison