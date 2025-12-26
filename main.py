import os
import time                # 用于生成带时间戳的文件名
import pickle
import warnings
import random
import copy
import numpy as np
import pandas as pd        # [关键修复] 解决 NameError: name 'pd' is not defined
from collections import Counter # [关键] 后面聚合代码用到了 Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, matthews_corrcoef
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. 全局配置
# ==========================================
DATA_PKL = "data_reified.pkl"
MODEL_SAVE_PATH = "best_model_final.pth"
SEEDS = [0, 7, 100, 3407, 2000]
TOP_K_PRED = 50 
# 训练超参数
NUM_EPOCHS = 100
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EMBED_DIM = 128
NUM_HEADS = 4

# [新增] 原型配置 (替代原来的 NUM_PROTOTYPES)
PROTO_CONFIG = {
    'pesticide': 10,
    'disease': 10,
    'plant': 10
}

# 设备配置 (必须在模型定义前定义，因为新模型代码中引用了全局 DEVICE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"-> Using device: {DEVICE}")

# ==========================================
# 2. 基础组件 (Model Components) - [已替换]
# ==========================================

class AttentionFusion(nn.Module):
    """
    多尺度注意力融合模块：
    对 HGT 不同层 (Layer 1, 2, 3) 的输出进行 Self-Attention 加权融合。
    """
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs):
        # inputs: list of tensors [x1, x2, x3], each shape (Batch, Dim)
        stacked = torch.stack(inputs, dim=1)  # (Batch, Num_Layers, Dim)
        attn_output, _ = self.attention(stacked, stacked, stacked)
        fused = attn_output.mean(dim=1)       # (Batch, Dim)
        return self.layer_norm(fused)

class HGTBlock(nn.Module):
    """
    HGT 编码块：包含 HGTConv, LayerNorm, ReLU 和 Dropout。
    """
    def __init__(self, in_channels, out_channels, metadata, heads=4, dropout=0.2):
        super().__init__()
        # PyG HGTConv 默认 group='sum'
        self.hgt = HGTConv(in_channels, out_channels, metadata, heads)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_dict, res_dict=None):
        x_new = self.hgt(x_dict, edge_dict)
        x_out = {}
        for k in x_new:
            x = x_new[k]
            # Residual Connection
            if res_dict is not None:
                res = res_dict.get(k, None)
                if res is not None and res.shape[-1] == x.shape[-1]:
                    x = x + res
            x = self.norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x_out[k] = x
        return x_out

class PrototypeRefiner(nn.Module):
    """
    原型精炼模块 (带门控融合)：
    1. 学习一组可训练的原型向量 (Prototypes)。
    2. 计算节点与原型的相似度，重构出“理想化特征”。
    3. 使用门控机制 (Gate) 将理想化特征与原始特征融合。
    """
    def __init__(self, num_prototypes, embed_dim, k=3, temperature=0.1):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, embed_dim))
        nn.init.orthogonal_(self.prototypes)
        
        # 变换原型特征的 MLP
        self.proto_transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), 
            nn.ReLU(), 
            nn.LayerNorm(embed_dim)
        )
        
        # [核心] 门控网络：输入 (原始特征 + 原型特征)，输出门控系数 (0~1)
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        self.final_norm = nn.LayerNorm(embed_dim)
        self.k = min(k, num_prototypes)
        self.warmup = True 
        self.temperature = temperature

    def set_warmup(self, status: bool):
        self.warmup = status

    def get_regularization_loss(self, batch_mean_probs):
        """
        计算正则化损失：
        1. 正交损失 (Orthogonal Loss): 保证原型多样性
        2. 均衡损失 (Balance Loss): 保证原型利用率均衡
        """
        p_norm = F.normalize(self.prototypes, dim=1)
        sim_matrix = p_norm @ p_norm.T
        identity = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
        ortho_loss = F.mse_loss(sim_matrix, identity)
        
        target_prob = 1.0 / self.prototypes.size(0)
        balance_loss = F.mse_loss(batch_mean_probs, torch.full_like(batch_mean_probs, target_prob))
        return ortho_loss + 2.0 * balance_loss

    def forward(self, x_instance):
        # 1. 相似度匹配
        x_norm = F.normalize(x_instance, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)
        logits = (x_norm @ p_norm.T) / self.temperature
        probs = torch.softmax(logits, dim=1)
        batch_mean_probs = probs.mean(dim=0) 
        
        # 2. Top-K 稀疏化
        if not self.warmup and self.k < logits.size(1):
            topk_values, topk_indices = torch.topk(logits, self.k, dim=1)
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(1, topk_indices, topk_values)
            sim = torch.softmax(mask, dim=1)
        else:
            sim = probs 
            
        # 3. 特征重构 (Abstraction)
        x_abstract = sim @ self.prototypes
        
        # 4. 门控融合 (Gated Fusion)
        x_proto_transformed = self.proto_transform(x_abstract)
        
        # 计算门控系数
        concat_feat = torch.cat([x_instance, x_proto_transformed], dim=1)
        gate = self.gate_net(concat_feat)
        
        # 融合：原始特征 + (门控 * 原型修正量)
        x_fused = x_instance + gate * x_proto_transformed
        
        # 5. 最终归一化
        x_final = self.final_norm(x_fused)
        
        return x_final, sim, batch_mean_probs

class TriplePredictor(nn.Module):
    """
    三元组预测头：输入 (P, D, Pl) -> 输出 Logits
    """
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.lin1 = nn.Linear(in_channels * 3, hidden_channels)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.lin3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, p_emb, d_emb, pl_emb):
        x = torch.cat([p_emb, d_emb, pl_emb], dim=1)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.lin2(x)))
        x = self.dropout(x)
        return self.lin3(x).squeeze()

# ==========================================
# 3. 主模型 (Full Model) - [已替换]
# ==========================================
class MultiModelNetV2(nn.Module):
    # [修改] num_prototypes 接收字典，例如 {'pesticide': 20, 'disease': 15, 'plant': 5}
    def __init__(self, metadata, input_channels_dict, num_events, embed_dim=128, heads=4, 
                 proto_config={'pesticide': 5, 'disease': 5, 'plant': 5}):
        super().__init__()
        
        # 1. Input Projection (特征对齐)
        self.input_projs = nn.ModuleDict()
        for node_type, in_dim in input_channels_dict.items():
            if node_type != 'event':
                self.input_projs[node_type] = nn.Linear(in_dim, embed_dim)
        
        # Event 节点使用 Embedding (可学习 ID)
        self.event_emb = nn.Embedding(num_events, embed_dim)

        # 2. HGT Encoder (3 Layers)
        self.block1 = HGTBlock(embed_dim, embed_dim, metadata, heads)
        self.block2 = HGTBlock(embed_dim, embed_dim, metadata, heads)
        self.block3 = HGTBlock(embed_dim, embed_dim, metadata, heads) 

        # 3. Multi-Scale Fusion
        self.fusion_p = AttentionFusion(embed_dim, heads)
        self.fusion_d = AttentionFusion(embed_dim, heads)
        self.fusion_pl = AttentionFusion(embed_dim, heads)

        # 4. Prototype Refiner (Gated Version)
        # [关键修改] 分别读取配置，针对性初始化
        # 使用 .get 提供默认值，防止报错
        num_p = proto_config.get('pesticide', 10)
        num_d = proto_config.get('disease', 10)
        num_pl = proto_config.get('plant', 10)

        print(f"Initializing Prototypes: Pesticide={num_p}, Disease={num_d}, Plant={num_pl}")

        self.refiner_p = PrototypeRefiner(num_prototypes=num_p, embed_dim=embed_dim, k=min(3, num_p))
        self.refiner_d = PrototypeRefiner(num_prototypes=num_d, embed_dim=embed_dim, k=min(3, num_d))
        self.refiner_pl = PrototypeRefiner(num_prototypes=num_pl, embed_dim=embed_dim, k=min(3, num_pl))
        
        # 5. Predictor
        self.predictor = TriplePredictor(embed_dim)
        
        # 用于存储 Refiner 的概率分布以计算 Loss
        self.last_probs = {'p': None, 'd': None, 'pl': None}

    def set_warmup(self, status):
        """控制 Refiner 是否开启 Top-K (Warmup 期间关闭)"""
        self.refiner_p.set_warmup(status)
        self.refiner_d.set_warmup(status)
        self.refiner_pl.set_warmup(status)

    def get_proto_reg_loss(self):
        """获取所有 Refiner 的正则化损失之和"""
        loss = torch.tensor(0.0, device=DEVICE) # 使用全局 DEVICE
        for key in self.last_probs:
            if self.last_probs[key] is not None:
                refiner = getattr(self, f"refiner_{key}")
                loss += refiner.get_regularization_loss(self.last_probs[key])
        return loss

    def forward(self, x_dict, edge_index_dict):
        # A. Input Embedding
        x_emb = {}
        for ntype, x in x_dict.items():
            if ntype == 'event':
                event_ids = torch.arange(x.shape[0], device=x.device)
                x_emb[ntype] = self.event_emb(event_ids)
            elif ntype in self.input_projs:
                x_emb[ntype] = F.relu(self.input_projs[ntype](x))
        
        # B. HGT Layers (3-hop propagation)
        x1 = self.block1(x_emb, edge_index_dict, x_emb)
        x2 = self.block2(x1, edge_index_dict, x1)
        x3 = self.block3(x2, edge_index_dict, x2)
        
        # C. Fusion & Refinement
        
        # Pesticide Branch
        p_raw = self.fusion_p([x1['pesticide'], x2['pesticide'], x3['pesticide']])
        p_final, _, p_probs = self.refiner_p(p_raw)
        self.last_probs['p'] = p_probs
        
        # Disease Branch
        d_raw = self.fusion_d([x1['disease'], x2['disease'], x3['disease']])
        d_final, _, d_probs = self.refiner_d(d_raw)
        self.last_probs['d'] = d_probs
        
        # Plant Branch
        pl_raw = self.fusion_pl([x1['plant'], x2['plant'], x3['plant']])
        pl_final, _, pl_probs = self.refiner_pl(pl_raw)
        self.last_probs['pl'] = pl_probs
        
        return {'pesticide': p_final, 'disease': d_final, 'plant': pl_final}

    def predict_triplets(self, p_emb, d_emb, pl_emb):
        return self.predictor(p_emb, d_emb, pl_emb)

# ==========================================
# 4. 工具函数
# ==========================================
def seed_everything(seed):
    print(f"-> Setting global seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(y_true, y_probs, threshold=0.5):
    """计算全套评价指标"""
    y_pred = (y_probs > threshold).astype(int)
    return {
        'AUC': roc_auc_score(y_true, y_probs),
        'AP': average_precision_score(y_true, y_probs),
        'Acc': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

def get_triplets_from_events(data, event_mask=None):
    """从 Event 节点提取 (P, D, Pl) 三元组"""
    pe_edges = data['pesticide', 'participates_in', 'event'].edge_index
    de_edges = data['disease', 'is_target_of', 'event'].edge_index
    ple_edges = data['plant', 'is_host_of', 'event'].edge_index
    
    e_to_p = dict(zip(pe_edges[1].tolist(), pe_edges[0].tolist()))
    e_to_d = dict(zip(de_edges[1].tolist(), de_edges[0].tolist()))
    e_to_pl = dict(zip(ple_edges[1].tolist(), ple_edges[0].tolist()))
    
    triplets = []
    all_event_indices = torch.arange(data['event'].num_nodes)
    target_events = all_event_indices[event_mask] if event_mask is not None else all_event_indices

    for e_idx in target_events.tolist():
        if e_idx in e_to_p and e_idx in e_to_d and e_idx in e_to_pl:
            triplets.append([e_to_p[e_idx], e_to_d[e_idx], e_to_pl[e_idx]])
            
    return torch.tensor(triplets, dtype=torch.long)

def sample_negative_triplets(pos_triplets, num_nodes_dict):
    """1:1 负采样"""
    num_pos = len(pos_triplets)
    # 平均分配负采样策略：1/3 替换P，1/3 替换D，1/3 替换Pl
    num_neg_p = num_pos // 3
    num_neg_d = num_pos // 3
    num_neg_pl = num_pos - num_neg_p - num_neg_d

    def _sample(triplets, col_idx, max_idx):
        neg = triplets.clone()
        neg[:, col_idx] = torch.randint(0, max_idx, (len(triplets),))
        return neg

    neg_p = _sample(pos_triplets[:num_neg_p], 0, num_nodes_dict['p'])
    neg_d = _sample(pos_triplets[num_neg_p:num_neg_p+num_neg_d], 1, num_nodes_dict['d'])
    neg_pl = _sample(pos_triplets[num_neg_p+num_neg_d:], 2, num_nodes_dict['pl'])

    return torch.cat([neg_p, neg_d, neg_pl], dim=0)

def mask_graph_by_events(data_orig, event_mask, device):
    """
    根据 event_mask 动态构建图结构。
    训练时只保留 Train Event 边，防止 Val/Test 信息泄露。
    """
    data_masked = data_orig.clone()
    valid_event_indices = torch.where(event_mask.to(device))[0]
    
    for edge_type in data_masked.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = data_masked[edge_type].edge_index.to(device)
        mask = None
        if dst_type == 'event':
            mask = torch.isin(edge_index[1], valid_event_indices)
        elif src_type == 'event':
            mask = torch.isin(edge_index[0], valid_event_indices)
            
        if mask is not None:
            data_masked[edge_type].edge_index = edge_index[:, mask]
            
    return data_masked

# ==========================================
# 5. 主训练流程
# ==========================================
# def main():
def train_model(seed, data_full):
    """
    训练单个种子的模型，并返回保存路径
    """
    seed_everything(seed)
    
    # [修改] 动态文件名，防止覆盖
    save_path = f"best_model_seed_{seed}.pth"
    
    # 2. 获取节点统计
    num_pesticides = data_full['pesticide'].num_nodes
    num_diseases = data_full['disease'].num_nodes
    num_plants = data_full['plant'].num_nodes
    num_events = data_full['event'].num_nodes
    num_nodes_dict = {'p': num_pesticides, 'd': num_diseases, 'pl': num_plants}
    
    # 3. 数据划分 (80/10/10)
    indices = torch.randperm(num_events)
    split1 = int(num_events * 0.8)
    split2 = int(num_events * 0.9)
    
    train_mask = torch.zeros(num_events, dtype=torch.bool); train_mask[indices[:split1]] = True
    val_mask = torch.zeros(num_events, dtype=torch.bool); val_mask[indices[split1:split2]] = True
    # test_mask = torch.zeros(num_events, dtype=torch.bool); test_mask[indices[split2:]] = True 
    # 在集成流程中，这里可以略过 test，或者仅仅打印一下 verify 性能
    
    train_triplets = get_triplets_from_events(data_full, train_mask)
    val_triplets = get_triplets_from_events(data_full, val_mask)
    
    # 4. 初始化模型
    input_channels = {nt: data_full[nt].x.shape[1] for nt in data_full.node_types}
    
    model = MultiModelNetV2(
        metadata=data_full.metadata(),
        input_channels_dict=input_channels,
        num_events=num_events,
        embed_dim=EMBED_DIM,
        heads=NUM_HEADS,
        proto_config=PROTO_CONFIG
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    patience = 10
    no_improve_cnt = 0
    
    # 5. 训练循环 (使用 tqdm)
    # leave=False 避免多轮训练刷屏
    pbar = tqdm(range(NUM_EPOCHS), desc=f"Training Seed {seed}", leave=False)
    
    for epoch in pbar:
        model.train()
        model.set_warmup(epoch < 10)
        
        perm = torch.randperm(train_triplets.size(0))
        triplets_shuffled = train_triplets[perm]
        
        total_loss = 0
        num_batches = (len(triplets_shuffled) + BATCH_SIZE - 1) // BATCH_SIZE
        
        train_graph = mask_graph_by_events(data_full, train_mask, DEVICE).to(DEVICE)
        
        for i in range(num_batches):
            optimizer.zero_grad()
            out_emb = model(train_graph.x_dict, train_graph.edge_index_dict)
            
            batch_pos = triplets_shuffled[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            batch_neg = sample_negative_triplets(batch_pos, num_nodes_dict)
            
            batch_all = torch.cat([batch_pos, batch_neg], dim=0).to(DEVICE)
            labels = torch.cat([torch.ones(len(batch_pos)), torch.zeros(len(batch_neg))]).to(DEVICE)
            
            logits = model.predict_triplets(
                out_emb['pesticide'][batch_all[:, 0]],
                out_emb['disease'][batch_all[:, 1]],
                out_emb['plant'][batch_all[:, 2]]
            )
            
            loss = criterion(logits, labels)
            if epoch >= 10:
                loss += 0.1 * model.get_proto_reg_loss()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        # 验证阶段
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_graph = mask_graph_by_events(data_full, train_mask, DEVICE).to(DEVICE)
                out_val = model(val_graph.x_dict, val_graph.edge_index_dict)
                
                v_neg = sample_negative_triplets(val_triplets, num_nodes_dict)
                v_all = torch.cat([val_triplets, v_neg], dim=0).to(DEVICE)
                v_lbl = torch.cat([torch.ones(len(val_triplets)), torch.zeros(len(v_neg))]).cpu().numpy()
                
                v_logits = model.predict_triplets(
                    out_val['pesticide'][v_all[:, 0]],
                    out_val['disease'][v_all[:, 1]],
                    out_val['plant'][v_all[:, 2]]
                )
                v_probs = torch.sigmoid(v_logits).cpu().numpy()
                val_auc = roc_auc_score(v_lbl, v_probs)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), save_path)
                    no_improve_cnt = 0
                else:
                    no_improve_cnt += 1
            
            pbar.set_postfix({'Loss': total_loss/num_batches, 'Best Val AUC': best_val_auc})
            
            if no_improve_cnt >= patience:
                pbar.close()
                break # 早停

    return save_path
def predict_new_links(current_seed, model_path, data_full):
    print(f"Generating predictions for Seed: {current_seed}...")
    seed_everything(current_seed)

    # 必要的统计数据
    num_pesticides = data_full['pesticide'].num_nodes
    num_events = data_full['event'].num_nodes

    input_channels = {nt: data_full[nt].x.shape[1] for nt in data_full.node_types}
    model = MultiModelNetV2(
        metadata=data_full.metadata(),
        input_channels_dict=input_channels,
        num_events=num_events,
        embed_dim=EMBED_DIM,
        heads=NUM_HEADS,
        proto_config=PROTO_CONFIG 
    ).to(DEVICE)

    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return None
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 预测时使用全图信息
    full_graph = data_full.to(DEVICE)
    with torch.no_grad():
        out_emb_dict = model(full_graph.x_dict, full_graph.edge_index_dict)
    
    p_emb_all = out_emb_dict['pesticide']
    d_emb_all = out_emb_dict['disease']
    pl_emb_all = out_emb_dict['plant']
    
    # 构造预测候选集 (仅预测已知的 Disease-Plant 组合)
    de_edges_np = data_full['disease', 'is_target_of', 'event'].edge_index.cpu().numpy()
    ple_edges_np = data_full['plant', 'is_host_of', 'event'].edge_index.cpu().numpy()
    
    e_to_d_map = dict(zip(de_edges_np[1], de_edges_np[0]))
    e_to_pl_map = dict(zip(ple_edges_np[1], ple_edges_np[0]))
    
    unique_d_pl_pairs = set()
    for e_idx in range(num_events):
        if e_idx in e_to_d_map and e_idx in e_to_pl_map:
            unique_d_pl_pairs.add((e_to_d_map[e_idx], e_to_pl_map[e_idx]))
            
    results = []
    # 使用 batch 处理农药，避免内存溢出
    # 将 unique_d_pl_pairs 转为 list 以便索引
    pairs_list = list(unique_d_pl_pairs)
    
    for d_idx, pl_idx in tqdm(pairs_list, desc=f"[Seed {current_seed}] Predicting", leave=False):
        # 优化：不repeat，直接利用 broadcasting 或者分批
        # 这里为了稳妥，沿用你的逻辑，但要注意 tensor 维度
        d_emb = d_emb_all[d_idx].unsqueeze(0) # (1, Dim)
        pl_emb = pl_emb_all[pl_idx].unsqueeze(0) # (1, Dim)
        
        # 扩展到所有农药
        d_emb_batch = d_emb.expand(num_pesticides, -1)
        pl_emb_batch = pl_emb.expand(num_pesticides, -1)
        
        with torch.no_grad():
            logits = model.predict_triplets(p_emb_all, d_emb_batch, pl_emb_batch)
            scores = torch.sigmoid(logits)
            
        # Top-K
        topk_scores, topk_p_indices = torch.topk(scores, k=TOP_K_PRED)
        
        for score, p_idx in zip(topk_scores.cpu().tolist(), topk_p_indices.cpu().tolist()):
            results.append({
                'd_idx': d_idx,
                'pl_idx': pl_idx,
                'p_idx': p_idx,
                'score': score,
                'seed': current_seed
            })

    return pd.DataFrame(results)

# def aggregate_and_save_results(all_results_dfs, data_full):
#     print(f"\n{'='*60}")
#     print("🚀 开始聚合多种子预测结果 (Ensemble Aggregation)")
#     print(f"{'='*60}\n")

#     if not all_results_dfs:
#         print("没有收集到预测结果。")
#         return

#     full_df = pd.concat(all_results_dfs, ignore_index=True)

#     agg_df = full_df.groupby(['d_idx', 'pl_idx', 'p_idx']).agg(
#         mean_score=('score', 'mean'),
#         std_score=('score', 'std'),
#         count=('seed', 'count')
#     ).reset_index()
    
#     # 过滤低共识度
#     MIN_VOTES = max(3, int(len(SEEDS) * 0.8)) # 动态设定：比如 5个种子至少要4票
#     print(f"过滤低共识结果 (保留得票数 >= {MIN_VOTES})...")
#     agg_df = agg_df[agg_df['count'] >= MIN_VOTES]
    
#     # 映射名称
#     p_names = data_full['pesticide'].names if hasattr(data_full['pesticide'], 'names') else [f"P_{i}" for i in range(data_full['pesticide'].num_nodes)]
#     d_names = data_full['disease'].names if hasattr(data_full['disease'], 'names') else [f"D_{i}" for i in range(data_full['disease'].num_nodes)]
#     pl_names = data_full['plant'].names if hasattr(data_full['plant'], 'names') else [f"Pl_{i}" for i in range(data_full['plant'].num_nodes)]

#     # 构建已知三元组检查
#     pe_edges = data_full['pesticide', 'participates_in', 'event'].edge_index.cpu()
#     de_edges = data_full['disease', 'is_target_of', 'event'].edge_index.cpu()
#     ple_edges = data_full['plant', 'is_host_of', 'event'].edge_index.cpu()
    
#     # 建立映射以快速查找 event 对应的 P, D, Pl
#     event_p = dict(zip(pe_edges[1].tolist(), pe_edges[0].tolist()))
#     event_d = dict(zip(de_edges[1].tolist(), de_edges[0].tolist()))
#     event_pl = dict(zip(ple_edges[1].tolist(), ple_edges[0].tolist()))
    
#     known_triplets = set()
#     for e in range(data_full['event'].num_nodes):
#         if e in event_p and e in event_d and e in event_pl:
#             known_triplets.add((event_p[e], event_d[e], event_pl[e]))

#     # 计算农药流行度 (用于惩罚)
#     p_popularity = Counter(pe_edges[0].tolist())

#     final_output = []
#     for _, row in agg_df.iterrows():
#         p_idx, d_idx, pl_idx = int(row['p_idx']), int(row['d_idx']), int(row['pl_idx'])
        
#         is_known = (p_idx, d_idx, pl_idx) in known_triplets
#         pop = p_popularity.get(p_idx, 0)
#         penalty_factor = np.sqrt(pop) if pop > 0 else 1.0 # 避免除0
#         penalized_score = row['mean_score'] / (1 + 0.1 * penalty_factor) # 稍微温和一点的惩罚
        
#         final_output.append({
#             'Disease': d_names[d_idx],
#             'Plant': pl_names[pl_idx],
#             'Recommended Pesticide': p_names[p_idx],
#             'Mean Score': row['mean_score'],
#             'Popularity': pop,
#             'Penalized Score': penalized_score,
#             'Std Score': row['std_score'],
#             'Vote Count': f"{int(row['count'])}/{len(SEEDS)}",
#             'Type': 'Known' if is_known else 'Novel Prediction'
#         })

#     df_final = pd.DataFrame(final_output)
#     df_final['Std Score'] = df_final['Std Score'].fillna(0.0)
    
#     # 过滤逻辑
#     FILTER_KEYWORDS = ["线虫"] 
#     df_filtered = df_final[~df_final['Disease'].apply(lambda x: any(k in str(x) for k in FILTER_KEYWORDS))]
    
#     df_novel = df_filtered[df_filtered['Type'] == 'Novel Prediction'].copy()
    
#     BROAD_SPECTRUM_PESTICIDES = [
#         "mancozeb", "代森锰锌", "carbendazim", "多菌灵", 
#         "chlorothalonil", "百菌清", "azoxystrobin", "嘧菌酯"
#     ]
#     # 大小写不敏感过滤
#     df_novel = df_novel[
#         ~df_novel['Recommended Pesticide'].apply(lambda x: any(k.lower() in str(x).lower() for k in BROAD_SPECTRUM_PESTICIDES))
#     ]
    
#     df_novel = df_novel.sort_values(by=['Mean Score'], ascending=False)
    
#     print(f"\n{'='*120}")
#     print("集成预测结果示例 (Top 20):")
#     print(f"{'病害':<15} | {'作物':<15} | {'推荐农药':<25} | {'得分':<6} | {'类型'}")
#     print("-" * 120)
#     for _, row in df_novel.head(20).iterrows():
#         print(f"{str(row['Disease'])[:15]:<15} | {str(row['Plant'])[:15]:<15} | {str(row['Recommended Pesticide'])[:25]:<25} | {row['Mean Score']:.4f} | {row['Type']}")
    
#     timestamp = time.strftime('%Y%m%d_%H%M')
#     output_filename = f"Ensemble_Results_{timestamp}.csv"
#     df_novel.to_csv(output_filename, index=False, encoding='utf-8-sig')
#     print(f"\n✅ 结果已保存: {output_filename}")
def aggregate_and_save_results(all_results_dfs, data_full):
    print(f"\n{'='*60}")
    print("🚀 开始聚合多种子预测结果 (含严格新颖性过滤)")
    print(f"{'='*60}\n")

    if not all_results_dfs:
        print("没有收集到预测结果。")
        return

    # 1. 构建基础映射
    p_names = data_full['pesticide'].names
    d_names = data_full['disease'].names
    pl_names = data_full['plant'].names

    # =========================================================================
    # [核心修改 1] 构建农药的“已知治疗病害档案”
    # 目的：记录每个农药已经能治哪些病（无论在什么作物上）
    # =========================================================================
    print("正在构建农药历史治疗档案...")
    pe_edges = data_full['pesticide', 'participates_in', 'event'].edge_index.cpu().numpy()
    de_edges = data_full['disease', 'is_target_of', 'event'].edge_index.cpu().numpy()
    
    # Event ID -> Pesticide ID / Disease ID
    e_to_p = dict(zip(pe_edges[1], pe_edges[0]))
    e_to_d = dict(zip(de_edges[1], de_edges[0]))
    
    # 记录每个农药ID 已知的 病害名称集合
    # 结构: {p_idx: {'炭疽病', '白粉病', ...}}
    pesticide_known_diseases = {}
    
    for e_idx in e_to_p:
        if e_idx in e_to_d:
            p_idx = e_to_p[e_idx]
            d_idx = e_to_d[e_idx]
            d_name = d_names[d_idx]
            
            if p_idx not in pesticide_known_diseases:
                pesticide_known_diseases[p_idx] = set()
            pesticide_known_diseases[p_idx].add(d_name)
            
    print("历史档案构建完成。")

    # 2. 合并预测结果
    full_df = pd.concat(all_results_dfs, ignore_index=True)

    # 3. 聚合
    agg_df = full_df.groupby(['d_idx', 'pl_idx', 'p_idx']).agg(
        mean_score=('score', 'mean'),
        std_score=('score', 'std'),
        count=('seed', 'count')
    ).reset_index()

    # 4. 基础共识度过滤
    MIN_VOTES = max(3, int(len(SEEDS) * 0.8))
    agg_df = agg_df[agg_df['count'] >= MIN_VOTES]

    # 5. 构建最终列表（加入严格新颖性判断）
    final_output = []
    
    # 常用广谱农药列表 (建议保留过滤)
    BROAD_SPECTRUM = ["mancozeb", "代森锰锌", "carbendazim", "多菌灵", "chlorothalonil", "百菌清"]

    for _, row in tqdm(agg_df.iterrows(), total=len(agg_df), desc="Filtering"):
        p_idx, d_idx, pl_idx = int(row['p_idx']), int(row['d_idx']), int(row['pl_idx'])
        
        p_name = p_names[p_idx]
        d_name = d_names[d_idx]
        pl_name = pl_names[pl_idx]
        
        # 基础过滤：广谱农药
        if any(b in str(p_name) for b in BROAD_SPECTRUM):
            continue
            
        # =====================================================================
        # [核心修改 2] 严格新颖性判断 (Strict Novelty Check)
        # 逻辑：如果这个农药以前治过这种病（即使是在别的作物上），那就不是我们要的“全新发现”
        # =====================================================================
        known_diseases = pesticide_known_diseases.get(p_idx, set())
        
        # 如果该病害名字出现在该农药的历史记录里 -> 说明是“老病新作物” (扩作)
        is_same_disease_extension = d_name in known_diseases
        
        # 我们只保留 (或者高亮) 那些农药从未处理过的病害
        # 这里我增加一个标签字段，由您决定是直接过滤还是在Excel里筛选
        
        # 策略A: 直接过滤掉扩作预测 (只看纯新的)
        # if is_same_disease_extension: continue 
        
        # 策略B: 保留但标记 (推荐)
        prediction_type = "⚠️ 扩作 (同病异作物)" if is_same_disease_extension else "✨ 创新 (未治过的新病)"
        
        # 计算一些辅助分数
        p_popularity = len(known_diseases) # 该农药治多少种病 (万金油程度)
        
        # 只有当它是创新预测时，分数才保持原样；如果是扩作，可以人工降权
        final_score = row['mean_score']
        if is_same_disease_extension:
            final_score *= 0.5 # 强行降权，让创新结果排前面

        final_output.append({
            'Disease': d_name,
            'Plant': pl_name,
            'Recommended Pesticide': p_name,
            'Prediction Type': prediction_type, # 新增列
            'Mean Score': row['mean_score'],
            'Adjusted Score': final_score,      # 新增列：降权后的分数
            'Pesticide Breadth': p_popularity,  # 该农药已知的防治病害数量
            'Vote Count': f"{int(row['count'])}/{len(SEEDS)}"
        })

    df_final = pd.DataFrame(final_output)
    
    if df_final.empty:
        print("筛选后无结果。")
        return

    # 6. 排序策略：优先看“创新”的，且分数高的
    # 我们按 'Adjusted Score' 排序，这样“创新”类会自然排在前面
    df_final = df_final.sort_values(by=['Adjusted Score'], ascending=False)
    
    # 7. 保存
    timestamp = time.strftime('%Y%m%d_%H%M')
    output_filename = f"Novelty_Prediction_{timestamp}.csv"
    df_final.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*120}")
    print("🔥 高创新性预测结果示例 (Top 20, 优先展示农药未治过的新病):")
    print(f"{'类型':<12} | {'病害':<10} | {'作物':<10} | {'推荐农药':<20} | {'原得分':<6}")
    print("-" * 120)
    
    for _, row in df_final.head(50).iterrows():
        print(f"{row['Prediction Type']:<12} | {str(row['Disease'])[:10]:<10} | {str(row['Plant'])[:10]:<10} | {str(row['Recommended Pesticide'])[:20]:<20} | {row['Mean Score']:.4f}")
    
    print(f"\n✅ 结果已保存至: {output_filename}")
# ==========================================
# 7. 主执行循环
# ==========================================
def run_ensemble_pipeline():
    print(f"Checking data file {DATA_PKL} ...")
    if not os.path.exists(DATA_PKL):
        print(f"错误: 未找到数据文件 {DATA_PKL}")
        return

    # [重要] 仅加载一次数据，然后传递给函数
    with open(DATA_PKL, "rb") as f:
        data_full = pickle.load(f)
    print("Data loaded successfully.")

    all_run_results = []
    print(f"开始集成流程，种子列表: {SEEDS}")
    
    for seed in SEEDS:
        print(f"\n>>> Processing SEED: {seed}")
        # 训练
        model_path = train_model(seed, data_full)
        # 预测
        df_seed_result = predict_new_links(seed, model_path, data_full)
        
        if df_seed_result is not None:
            all_run_results.append(df_seed_result)
        
        # 可选：删除临时模型节省空间
        # if os.path.exists(model_path): os.remove(model_path)
            
    # 聚合
    aggregate_and_save_results(all_run_results, data_full)

# if __name__ == "__main__":
#     run_ensemble_pipeline()
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager

def set_manual_font(font_path='SimHei.ttf'):
    """
    手动加载指定路径的字体文件
    """
    # 1. 检查文件是否存在
    if not os.path.exists(font_path):
        print(f"❌ 错误：在当前目录下未找到字体文件 '{font_path}'")
        print("请确保你已经将 SimHei.ttf 上传到了脚本所在的目录！")
        return

    # 2. 将字体文件添加到 Matplotlib 的字体管理器中
    try:
        # addfont 是 Matplotlib 3.2+ 的新特性，最直接有效
        font_manager.fontManager.addfont(font_path)
        
        # 获取该字体的内部名称（有时候文件名是 SimHei.ttf，但内部名称叫 SimHei）
        prop = font_manager.FontProperties(fname=font_path)
        font_name = prop.get_name()
        
        # 3. 设置全局字体参数
        plt.rcParams['font.sans-serif'] = [font_name] # 设置无衬线字体为该字体
        plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
        
        print(f"✅ 字体加载成功！已使用: {font_name} ({font_path})")
        
    except Exception as e:
        print(f"❌ 字体加载出错: {e}")

# ==========================================
# 执行配置
# ==========================================
# 假设你上传的文件名是 SimHei.ttf
set_manual_font('SimHei.ttf')
# ==========================================
# 在脚本最开始运行一次即可
# ==========================================
# configure_chinese_font()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import os

# 设置绘图风格
sns.set(style="whitegrid", context="talk")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

def analyze_prototype_mechanisms(model, data, node_type, device, output_dir="vis_results"):
    """
    针对原型模块进行深度分析：热力图、分布图、语义表
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n🔬 正在深度分析 [{node_type}] 的原型机制...")
    
    # 1. 获取对应的 Refiner 和名称
    if node_type == 'pesticide':
        refiner = model.refiner_p
        names = data['pesticide'].names
    elif node_type == 'disease':
        refiner = model.refiner_d
        names = data['disease'].names
    elif node_type == 'plant':
        refiner = model.refiner_pl
        names = data['plant'].names
    else:
        return

    # 2. 提取数据 (前向传播一次以获取最新的节点嵌入)
    model.eval()
    with torch.no_grad():
        # 获取原型向量 (K, Dim)
        prototypes = refiner.prototypes.data.cpu()
        # 获取经过网络处理后的节点向量 (N, Dim)
        out = model(data.x_dict, data.edge_index_dict)
        node_embs = out[node_type].cpu()

    # 归一化，方便计算余弦相似度
    prototypes_norm = F.normalize(prototypes, dim=1)
    node_embs_norm = F.normalize(node_embs, dim=1)
    
    num_protos = prototypes.shape[0]

    # ==========================================
    # 可视化 A: 原型自相似度热力图 (Diversity Check)
    # ==========================================
    # 计算原型之间的相似度矩阵 (K, K)
    proto_sim_matrix = torch.mm(prototypes_norm, prototypes_norm.t()).numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(proto_sim_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                vmin=-0.2, vmax=1.0, square=True,
                xticklabels=[f"P{i}" for i in range(num_protos)],
                yticklabels=[f"P{i}" for i in range(num_protos)])
    plt.title(f"{node_type} - Prototype Similarity (Diversity Check)")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{node_type}_similarity_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [1/3] 自相似热力图已保存: {save_path}")

    # ==========================================
    # 可视化 B: 节点归属分布图 (Utilization Check)
    # ==========================================
    # 计算每个节点最接近哪个原型
    # (N, K)
    similarity_scores = torch.mm(node_embs_norm, prototypes_norm.t())
    # 获取每个节点归属的原型 ID
    assignments = torch.argmax(similarity_scores, dim=1).numpy()
    
    # 统计每个原型有多少个节点
    counts = pd.Series(assignments).value_counts().sort_index()
    # 补全可能为0的原型
    for i in range(num_protos):
        if i not in counts:
            counts[i] = 0
            
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    ax.set_xlabel("Prototype ID")
    ax.set_ylabel("Number of Assigned Nodes")
    ax.set_title(f"{node_type} - Node Assignment Distribution")
    # 在柱子上标数值
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values)*0.01, str(v), ha='center')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{node_type}_assignment_dist.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [2/3] 归属分布图已保存: {save_path}")

    # ==========================================
    # 分析 C: 原型语义解码表 (Interpretation)
    # ==========================================
    # 找出每个原型相似度最高的 Top-5 节点
    print(f"  [3/3] 生成语义解释表...")
    
    proto_semantics = []
    
    # 使用 similarity_scores (N, K) -> 转置为 (K, N)
    sim_t = similarity_scores.t()
    
    for i in range(num_protos):
        # 找到该原型得分最高的 Top K 索引
        values, indices = torch.topk(sim_t[i], k=8)
        
        # 获取名称
        top_names = [str(names[idx.item()]) for idx in indices]
        
        # 记录数据
        proto_semantics.append({
            "Prototype ID": f"P{i}",
            "Count": counts[i],
            "Representative Entities": ", ".join(top_names[:5]), # 只展示前5个防止太长
            "Top 1 Score": f"{values[0].item():.4f}" # 记录最相似的那个分数，看确信度
        })
        
    df_semantics = pd.DataFrame(proto_semantics)
    
    # 打印表格
    print(f"\n{'-'*80}")
    print(f"语义解释: {node_type}")
    print(f"{'-'*80}")
    # 设置pandas显示选项以便在终端看全
    pd.set_option('display.max_colwidth', 100) 
    print(df_semantics[["Prototype ID", "Count", "Representative Entities"]])
    print(f"{'-'*80}\n")
    
    # 保存 CSV
    csv_path = os.path.join(output_dir, f"{node_type}_prototype_semantics.csv")
    df_semantics.to_csv(csv_path, index=False, encoding='utf-8-sig')


def run_prototype_analysis_pipeline(seed, model_path):
    with open(DATA_PKL, "rb") as f:
        data_full = pickle.load(f)
    num_pesticides = data_full['pesticide'].num_nodes
    num_events = data_full['event'].num_nodes
    print(f"\n{'='*60}")
    print(f"🚀 开始原型深度分析 (Seed: {seed})")
    print(f"📂 结果将保存至: ./vis_analysis_results/")
    print(f"{'='*60}")
    
    # 1. 模型初始化 (结构必须与训练时完全一致)
    input_channels = {}
    for node_type in ['pesticide', 'disease', 'plant']:
        input_channels[node_type] = data_full[node_type].x.shape[1]
    PROTO_CONFIG = {
    'pesticide': 10,  # 农药种类繁多，机制复杂，给多一点
    'disease': 10,    # 病害种类中等
    'plant': 10       # 作物种类相对较少，或者我们只关注大类，给少一点
    }
    model = MultiModelNetV2(
        metadata=data_full.metadata(),
        input_channels_dict=input_channels,
        num_events=num_events,
        embed_dim=EMBED_DIM,
        heads=NUM_HEADS,
        proto_config=PROTO_CONFIG 
    ).to(DEVICE)
    
    # 2. 加载权重
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件 {model_path} 不存在。请先训练。")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print("✅ 模型权重已加载。")

    # 3. 准备数据
    data_vis = data_full.to(DEVICE)
    
    # 4. 执行三大类节点的分析
    analyze_prototype_mechanisms(model, data_vis, 'disease', DEVICE, output_dir="vis_analysis_results")
    analyze_prototype_mechanisms(model, data_vis, 'plant', DEVICE, output_dir="vis_analysis_results")
    analyze_prototype_mechanisms(model, data_vis, 'pesticide', DEVICE, output_dir="vis_analysis_results")
    
    print("\n🎉 分析全部完成！请查看 ./vis_analysis_results/ 文件夹下的图片和CSV文件。")

if __name__ == "__main__":
    # 使用集成学习中的第一个种子及其对应的模型
    target_seed = SEEDS[3] 
    target_model_path = f"/home/fine-tune/gnn/best_model_final.pth"
    
    # 为了演示，如果文件不存在，你可以选择先不跑，或者在这里自动触发一次快速训练
    if os.path.exists(target_model_path):
        run_prototype_analysis_pipeline(target_seed, target_model_path)
    else:
        print(f"请确保 {target_model_path} 存在 (可以通过运行主训练脚本生成)")

# -*- coding: utf-8 -*-
import os
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.nn import HGTConv

# ==========================================
# 1. 混合字体配置 (支持 SCI 风格 + 中文)
# ==========================================
# 优先使用 Times New Roman (英文)，如果缺字则回退到 SimHei (中文)
# 注意：Matplotlib 的 font.serif 列表机制可以实现混合显示
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.0

# 配置路径
DATA_PKL = "data_reified.pkl"
MODEL_PATH = "/home/fine-tune/gnn/best_model_final.pth" 
OUTPUT_DIR = "sci_plots_output_mixed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. 中英对照字典 (仅保留作物和病害)
# ==========================================
TRANSLATION_MAP = {
    # --- Sample Targets ---
    "水稻": "Rice",
    "苹果": "Apple",
    "番茄": "Tomato",
    "棉花": "Cotton",
    "炭疽病": "Anthracnose",
    "白粉病": "Powdery Mildew",
    "纹枯病": "Sheath Blight",
    "根结线虫": "Root-knot Nematode",
    "猕猴桃":"Kiwifruit",

    # --- Block 1: 病害 (Disease) ---
    "炭疽病": "Anthracnose", "白粉病": "Powdery Mildew",
    "叶锈病": "Leaf Rust", "储藏病害": "Storage Disease",
    "菌核病": "Sclerotinia Rot", "胡麻叶斑病": "Brown Spot", 
    "白锈病": "White Rust", "苗炭疽病": "Seedling Anthracnose",
    "锈壁虱": "Rust Mite", "美国白蛾": "Fall Webworm",
    "白绢病": "Southern Blight", "全蚀病": "Take-all",
    "枯萎病": "Fusarium Wilt", "立枯病": "Damping-off",
    "蒂腐病": "Stem End Rot", "绿霉病": "Green Mold",
    "菜青虫": "Cabbage Caterpillar", "松毛虫": "Pine Caterpillar",
    "叶霉病": "Leaf Mold", "赤霉病": "Fusarium Head Blight",

    # --- Block 2: 作物 (Plant) ---
    "郁金香": "Tulip", "水稻移栽田": "Transplanted Rice Field",
    "枸杞": "Wolfberry", "菜豆": "Common Bean",
    "大葱": "Welsh Onion", "菊花": "Chrysanthemum",
    "甘蔗": "Sugarcane", "绿萍": "Duckweed",
    "柑橘": "Citrus", "苹果树": "Apple Tree",
    "花卉": "Flowers", "观赏百合": "Ornamental Lily",
    "荞麦": "Buckwheat", "青梅": "Green Plum",
    "观赏菊花": "Ornamental Chrysanthemum", "葡萄": "Grape",
    "枇杷树": "Loquat Tree", "杨梅": "Chinese Bayberry",
    "人参": "Ginseng", "辣椒": "Chili Pepper",
}

def translate(text):
    """翻译函数"""
    if text in TRANSLATION_MAP:
        return TRANSLATION_MAP[text]
    for k, v in TRANSLATION_MAP.items():
        if k == text: return v
    return text 

# ==========================================
# 3. 模型定义 (保持不变)
# ==========================================
class AttentionFusion(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1) 
        attn_output, _ = self.attention(stacked, stacked, stacked)
        return self.layer_norm(attn_output.mean(dim=1))

class HGTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, metadata, heads=4, dropout=0.2):
        super().__init__()
        self.hgt = HGTConv(in_channels, out_channels, metadata, heads)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x_dict, edge_dict, res_dict=None):
        x_new = self.hgt(x_dict, edge_dict)
        x_out = {}
        for k in x_new:
            x = x_new[k]
            if res_dict is not None:
                res = res_dict.get(k, None)
                if res is not None and res.shape[-1] == x.shape[-1]:
                    x = x + res
            x_out[k] = self.dropout(F.relu(self.norm(x)))
        return x_out

class PrototypeRefiner(nn.Module):
    def __init__(self, num_prototypes, embed_dim, k=3, temperature=0.1):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, embed_dim))
        nn.init.orthogonal_(self.prototypes)
        self.proto_transform = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
        self.gate_net = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        self.final_norm = nn.LayerNorm(embed_dim)
        self.k = min(k, num_prototypes)
        self.warmup = True 
        self.temperature = temperature
    def set_warmup(self, status: bool): self.warmup = status
    def get_regularization_loss(self, batch_mean_probs): return 0.0
    def forward(self, x_instance): return None, None, None

class MultiModelNetV2(nn.Module):
    def __init__(self, metadata, input_channels_dict, num_events, embed_dim=128, heads=4, 
                 proto_config={'pesticide': 5, 'disease': 5, 'plant': 5}):
        super().__init__()
        self.input_projs = nn.ModuleDict()
        for node_type, in_dim in input_channels_dict.items():
            if node_type != 'event': self.input_projs[node_type] = nn.Linear(in_dim, embed_dim)
        self.event_emb = nn.Embedding(num_events, embed_dim)
        self.block1 = HGTBlock(embed_dim, embed_dim, metadata, heads)
        self.block2 = HGTBlock(embed_dim, embed_dim, metadata, heads)
        self.block3 = HGTBlock(embed_dim, embed_dim, metadata, heads) 
        self.fusion_p = AttentionFusion(embed_dim, heads)
        self.fusion_d = AttentionFusion(embed_dim, heads)
        self.fusion_pl = AttentionFusion(embed_dim, heads)
        num_p, num_d, num_pl = proto_config.get('pesticide', 10), proto_config.get('disease', 10), proto_config.get('plant', 10)
        self.refiner_p = PrototypeRefiner(num_p, embed_dim, k=min(3, num_p))
        self.refiner_d = PrototypeRefiner(num_d, embed_dim, k=min(3, num_d))
        self.refiner_pl = PrototypeRefiner(num_pl, embed_dim, k=min(3, num_pl))

    def forward(self, x_dict, edge_index_dict):
        x_emb = {}
        for ntype, x in x_dict.items():
            if ntype == 'event': x_emb[ntype] = self.event_emb(torch.arange(x.shape[0], device=x.device))
            elif ntype in self.input_projs: x_emb[ntype] = F.relu(self.input_projs[ntype](x))
        x1 = self.block1(x_emb, edge_index_dict, x_emb)
        x2 = self.block2(x1, edge_index_dict, x1)
        x3 = self.block3(x2, edge_index_dict, x2)
        p_raw = self.fusion_p([x1['pesticide'], x2['pesticide'], x3['pesticide']])
        d_raw = self.fusion_d([x1['disease'], x2['disease'], x3['disease']])
        pl_raw = self.fusion_pl([x1['plant'], x2['plant'], x3['plant']])
        return {'pesticide': p_raw, 'disease': d_raw, 'plant': pl_raw}

# ==========================================
# 4. 核心功能函数 (已修改：农药不翻译)
# ==========================================
def get_prototype_keywords_sci(model, data, node_type, top_k=2):
    """
    计算原型关键词。
    - 如果是 'pesticide'，保留原始中文。
    - 否则，翻译为英文。
    """
    if node_type == 'pesticide':
        refiner = model.refiner_p
        names = data['pesticide'].names
    elif node_type == 'disease':
        refiner = model.refiner_d
        names = data['disease'].names
    elif node_type == 'plant':
        refiner = model.refiner_pl
        names = data['plant'].names
    
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        entity_feats = F.normalize(out[node_type].cpu(), dim=1)
        prototypes = F.normalize(refiner.prototypes.data.cpu(), dim=1)
    
    scores = torch.mm(entity_feats, prototypes.t()) / refiner.temperature
    scores_t = scores.t()
    
    proto_labels = []
    for i in range(prototypes.shape[0]):
        _, indices = torch.topk(scores_t[i], k=top_k)
        
        keywords = []
        for idx in indices:
            raw_name = str(names[idx.item()])
            # [关键逻辑] 农药不翻译，其他翻译
            if node_type == 'pesticide':
                keywords.append(raw_name)
            else:
                keywords.append(translate(raw_name))
        
        short_keywords = "\n".join(keywords)
        proto_labels.append(f"P{i}\n({short_keywords})")
        
    return proto_labels

def plot_sample_similarity_sci(model, data, node_type, sample_names):
    """生成 SCI 风格图表 (农药显示中文，其他显示英文)"""
    print(f"\n🎨 Generating plots for [{node_type}] (No translation for pesticides)...")
    
    if node_type == 'pesticide':
        names_list = data['pesticide'].names
        refiner = model.refiner_p
    elif node_type == 'disease':
        names_list = data['disease'].names
        refiner = model.refiner_d
    elif node_type == 'plant':
        names_list = data['plant'].names
        refiner = model.refiner_pl
    
    name_to_idx = {name: i for i, name in enumerate(names_list)}
    
    # 获取标签 (农药已处理为不翻译)
    proto_labels = get_prototype_keywords_sci(model, data, node_type, top_k=2)
    num_protos = len(proto_labels)
    
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        all_node_embs = out[node_type].cpu()
        prototypes = refiner.prototypes.data.cpu()
        
    all_node_embs_norm = F.normalize(all_node_embs, dim=1)
    prototypes_norm = F.normalize(prototypes, dim=1)
    
    num_samples = len(sample_names)
    cols = 2
    rows = math.ceil(num_samples / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.5 * rows), constrained_layout=True)
    if num_samples == 1: axes = [axes]
    axes = np.array(axes).flatten()
    
    valid_count = 0
    
    for i, sample_name_cn in enumerate(sample_names):
        idx = -1
        # 查找逻辑
        if sample_name_cn in name_to_idx:
            idx = name_to_idx[sample_name_cn]
        else:
            candidates = [k for k in name_to_idx.keys() if sample_name_cn in str(k)]
            if candidates:
                print(f"  > '{sample_name_cn}' matched to '{candidates[0]}'")
                sample_name_cn = candidates[0]
                idx = name_to_idx[sample_name_cn]
        
        if idx == -1:
            print(f"⚠️ Warning: Sample '{sample_name_cn}' not found.")
            continue
            
        valid_count += 1
        
        # [关键逻辑] 标题显示：农药用原名，其他用英文
        if node_type == 'pesticide':
            display_title = sample_name_cn
        else:
            display_title = translate(sample_name_cn)
        
        sample_vec = all_node_embs_norm[idx].unsqueeze(0)
        sim_scores = torch.mm(sample_vec, prototypes_norm.t()).squeeze().numpy()
        
        ax = axes[i]
        norm = plt.Normalize(-0.5, 1.0)
        colors = plt.cm.coolwarm(norm(sim_scores))
        
        bars = ax.bar(range(num_protos), sim_scores, color=colors, 
                      edgecolor='black', linewidth=0.6, width=0.7)
        
        # 标题
        ax.set_title(f"Sample: {display_title}", fontsize=14, fontweight='bold', loc='left')
        
        # 坐标轴
        ax.set_xticks(range(num_protos))
        ax.set_xticklabels(proto_labels, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel("Cosine Similarity", fontsize=11)
        ax.set_ylim(-0.4, 1.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        
        for bar in bars:
            height = bar.get_height()
            offset = 0.05 if height >= 0 else -0.12
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9, color='black')

    for j in range(valid_count, len(axes)):
        axes[j].axis('off')
        
    save_base = f"{OUTPUT_DIR}/sci_analysis_{node_type}"
    plt.savefig(f"{save_base}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(f"{save_base}.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved to: {save_base}.png")

# ==========================================
# 5. 主程序
# ==========================================
def main():
    if not os.path.exists(DATA_PKL):
        print(f"❌ Data file {DATA_PKL} not found.")
        return
        
    print("Loading data...")
    with open(DATA_PKL, "rb") as f:
        data_full = pickle.load(f)
    
    num_events = data_full['event'].num_nodes
    input_channels = {k: data_full[k].x.shape[1] for k in data_full.node_types}

    PROTO_CONFIG = {'pesticide': 10, 'disease': 10, 'plant': 10}
    
    model = MultiModelNetV2(
        metadata=data_full.metadata(),
        input_channels_dict=input_channels,
        num_events=num_events,
        embed_dim=128,
        heads=4,
        proto_config=PROTO_CONFIG 
    ).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    else:
        print(f"❌ Model file {MODEL_PATH} not found.")
        return

    data_vis = data_full.to(DEVICE)

    # -----------------------------------------------------------
    # 1. 作物 (翻译成英文)
    # -----------------------------------------------------------
    plant_targets = ["猕猴桃", "苹果", "玉米", "烟草"] 
    plot_sample_similarity_sci(model, data_vis, 'plant', plant_targets)

    # -----------------------------------------------------------
    # 2. 病害 (翻译成英文)
    # -----------------------------------------------------------
    disease_targets = ["褐斑病", "斑点落叶病", "丝黑穗病", "野火病"]
    plot_sample_similarity_sci(model, data_vis, 'disease', disease_targets)

    # -----------------------------------------------------------
    # 3. 农药 (不翻译，保持中文，字体回退到 SimHei)
    # -----------------------------------------------------------
    # pesticide_targets = ["代森锰锌", "阿维菌素", "吡虫啉", "戊唑醇"]
    pesticide_samples = ["tebuconazole", "thiophanate-methyl", "hexaconazole",'copper oxychloride']
    plot_sample_similarity_sci(model, data_vis, 'pesticide', pesticide_samples)

if __name__ == "__main__":
    main()