import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

# ==========================================
# 1. 基础组件 (Model Components)
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
# 2. 主模型 (Full Model)
# ==========================================
class MultiModelNetV2(nn.Module):
    def __init__(self, metadata, input_channels_dict, num_events, embed_dim=128, heads=4, num_prototypes=10):
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
        self.refiner_p = PrototypeRefiner(num_prototypes, embed_dim)
        self.refiner_d = PrototypeRefiner(num_prototypes, embed_dim)
        self.refiner_pl = PrototypeRefiner(num_prototypes, embed_dim)
        
        # 5. Predictor
        self.predictor = TriplePredictor(embed_dim)
        
        # 用于存储 Refiner 的概率分布以计算 Loss
        self.last_probs = {'p': None, 'd': None, 'pl': None}

    def set_warmup(self, status):
        """控制 Refiner 是否开启 Top-K (Warmup 期间关闭)"""
        self.refiner_p.set_warmup(status)
        self.refiner_d.set_warmup(status)
        self.refiner_pl.set_warmup(status)

    def get_proto_reg_loss(self, device):
        """获取所有 Refiner 的正则化损失之和"""
        loss = torch.tensor(0.0, device=device)
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

# -*- coding: utf-8 -*-
import os
# 1. 环境设置：保证实验可复现性
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
# os.environ['PYTHONHASHSEED'] = '42'

import pickle
import warnings
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import time

# 引入评价指标
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef
)



warnings.filterwarnings("ignore")

# ==========================================
# 1. 全局配置
# ==========================================
DATA_PKL = "data_reified.pkl"
MODEL_SAVE_PATH = "best_model_final.pth"
SEED = 3407

# 训练超参数
NUM_EPOCHS = 100
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EMBED_DIM = 128
NUM_HEADS = 4
NUM_PROTOTYPES = 10  # 原型数量

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"-> Using device: {DEVICE}")

# ==========================================
# 2. 工具函数
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
    # 如果 PyTorch 版本支持，强制使用确定性算法
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except AttributeError:
    #     pass

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
# 3. 主训练流程
# ==========================================
def main():
    seed_everything(SEED)

    # 1. 加载数据
    print(f"Loading data from {DATA_PKL} ...")
    if not os.path.exists(DATA_PKL):
        raise FileNotFoundError(f"Data file {DATA_PKL} not found.")
    
    with open(DATA_PKL, "rb") as f:
        data_full = pickle.load(f)
    print("Data loaded successfully.")

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
    test_mask = torch.zeros(num_events, dtype=torch.bool); test_mask[indices[split2:]] = True
    
    # 验证和测试时，可以使用之前的数据作为上下文
    train_val_mask = train_mask | val_mask 

    # 提取三元组
    train_triplets = get_triplets_from_events(data_full, train_mask)
    val_triplets = get_triplets_from_events(data_full, val_mask)
    test_triplets = get_triplets_from_events(data_full, test_mask)
    
    print(f"Split Info: Train={len(train_triplets)}, Val={len(val_triplets)}, Test={len(test_triplets)}")

    # 4. 初始化模型
    # 获取特征维度
    input_channels = {nt: data_full[nt].x.shape[1] for nt in data_full.node_types}
    
    model = MultiModelNetV2(
        metadata=data_full.metadata(),
        input_channels_dict=input_channels,
        num_events=num_events,
        embed_dim=EMBED_DIM,
        heads=NUM_HEADS,
        num_prototypes=NUM_PROTOTYPES
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    
    # 5. 训练循环
    print("\nStarting Training...")
    pbar = tqdm(range(NUM_EPOCHS), desc="Training")
    
    for epoch in pbar:
        model.train()
        model.set_warmup(epoch < 10) # 前10轮预热 Refiner
        
        # Shuffle
        perm = torch.randperm(train_triplets.size(0))
        triplets_shuffled = train_triplets[perm]
        
        total_loss = 0
        num_batches = (len(triplets_shuffled) + BATCH_SIZE - 1) // BATCH_SIZE
        
        # 训练时：只看 Train Graph
        train_graph = mask_graph_by_events(data_full, train_mask, DEVICE).to(DEVICE)
        
        for i in range(num_batches):
            optimizer.zero_grad()
            
            # Forward
            out_emb = model(train_graph.x_dict, train_graph.edge_index_dict)
            
            # Batch Data
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
            
            # 正则化 Loss (Warmup 后加入)
            if epoch >= 10:
                # loss += 0.1 * model.get_proto_reg_loss(DEVICE)
                loss += 1 * model.get_proto_reg_loss(DEVICE)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        # 验证阶段
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                # 验证时：使用 Train Mask 作为图结构 (Inductive Setting)
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
                
                try:
                    val_auc = roc_auc_score(v_lbl, v_probs)
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        torch.save(model.state_dict(), MODEL_SAVE_PATH)
                except:
                    pass
            
            pbar.set_postfix({'Loss': total_loss/num_batches, 'Best Val AUC': best_val_auc})

    # ==========================================
    # 6. 最终测试 (Test Phase)
    # ==========================================
    print("\nTraining Finished. Loading best model for testing...")
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    else:
        print("Warning: Best model not found, utilizing last epoch weights.")

    model.eval()
    with torch.no_grad():
        # 测试时：使用 Train + Val Mask 构建图 (Transductive Context)
        test_graph = mask_graph_by_events(data_full, train_val_mask, DEVICE).to(DEVICE)
        out_test = model(test_graph.x_dict, test_graph.edge_index_dict)
        
        t_neg = sample_negative_triplets(test_triplets, num_nodes_dict)
        t_all = torch.cat([test_triplets, t_neg], dim=0).to(DEVICE)
        t_lbl = torch.cat([torch.ones(len(test_triplets)), torch.zeros(len(t_neg))]).cpu().numpy()
        
        t_logits = model.predict_triplets(
            out_test['pesticide'][t_all[:, 0]],
            out_test['disease'][t_all[:, 1]],
            out_test['plant'][t_all[:, 2]]
        )
        t_probs = torch.sigmoid(t_logits).cpu().numpy()
        
        # 计算全套指标
        metrics = calculate_metrics(t_lbl, t_probs, threshold=0.5)

    print("\n" + "="*80)
    print(f"🏆 Final Test Results (Ours - MultiModelNetV2)")
    print("-" * 80)
    print(f"{'AUC':<10} | {'AP':<10} | {'Acc':<10} | {'F1':<10} | {'Precision':<10} | {'Recall':<10} | {'MCC':<10}")
    print("-" * 80)
    print(f"{metrics['AUC']:.4f}     | {metrics['AP']:.4f}     | {metrics['Acc']:.4f}     | {metrics['F1']:.4f}     | {metrics['Precision']:.4f}     | {metrics['Recall']:.4f}     | {metrics['MCC']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()