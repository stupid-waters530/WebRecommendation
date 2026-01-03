"""
v5.1: Transformer 作为特征提取器 + item embedding 点积 做推荐 + 文本特征
改动重点：
1）加入文本特征：使用 'prajjwal1/bert-small' 提取商品名称的语义特征
2）文本特征与ID嵌入融合后输入Transformer
3）仍然是 seq-only（不用 user embedding、不用 POP）
4）训练集用 next-item 监督
"""

import ast
import random
from typing import List, Tuple, Dict
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# ======================
# 日志 & 运行目录
# ======================

logger = None  # 全局 logger

class Logger:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str):
        print(msg)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def get_run_dir() -> Path:
    """
    创建本次运行目录，名称格式：YYMMDDHOURSMINUTES，例如 2502011135
    """
    now = datetime.datetime.now()
    run_name = now.strftime("%y%m%d%H%M")
    run_dir = Path(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def log(msg: str):
    """
    统一日志入口：优先写 logger，没有就直接 print
    """
    global logger
    if logger is None:
        print(msg)
    else:
        logger.log(msg)


# ======================
# 一些基础设置
# ======================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MAX_SEQ_LEN = 100
BATCH_SIZE = 256

# 模型参数
EMBED_DIM = 128
TEXT_EMBED_DIM = 512  # BERT-small 输出维度
COMBINED_DIM = 256    # 融合后的维度
N_HEAD = 4
NUM_ENCODER_LAYERS = 2
FFN_DIM = 256
DROPOUT = 0.2

NUM_EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 5.0
PATIENCE = 8

TRAIN_PATH = "text4.csv"
TEST_PATH = "test2.csv"
SUBMISSION_PATH = "submission.csv"

MAX_TRAIN_SAMPLES_PER_USER = 40
MAX_HISTORY_LEN_FOR_TRAIN = 500

# BERT 模型设置
TEXT_MODEL_NAME = 'prajjwal1/bert-small'
TEXT_BATCH_SIZE = 64


# ======================
# 文本特征提取
# ======================

class TextFeatureExtractor:
    """文本特征提取器"""
    def __init__(self, model_name: str = TEXT_MODEL_NAME):
        log(f"Loading text model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        
    def extract_features(self, texts: List[str]) -> torch.Tensor:
        """提取文本特征"""
        features = []
        
        for i in range(0, len(texts), TEXT_BATCH_SIZE):
            batch_texts = texts[i:i+TEXT_BATCH_SIZE]
            
            # 处理空文本
            batch_texts = [text if isinstance(text, str) and text.strip() else "" 
                          for text in batch_texts]
            
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=32
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_features = outputs.last_hidden_state.mean(dim=1)
                features.append(batch_features.cpu())
        
        return torch.cat(features, dim=0)


def build_item_mapping_with_text(train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> Tuple[dict, dict, torch.Tensor]:
    """
    构建商品映射并提取文本特征
    """
    log("Building item mapping with text features...")
    
    # 收集所有商品
    all_items = set()
    item_text_map = {}
    
    # 处理训练数据
    for _, row in train_df.iterrows():
        # 历史商品
        hist_ids = parse_history(row["history_item_id"])
        hist_titles = parse_history(row.get("history_item_title", "[]"))
        
        for item_id, title in zip(hist_ids, hist_titles[:len(hist_ids)]):
            all_items.add(item_id)
            if item_id not in item_text_map and title:
                item_text_map[item_id] = title
        
        # 目标商品
        if 'item_id' in train_df.columns and pd.notna(row['item_id']):
            target_id = int(row['item_id'])
            all_items.add(target_id)
            if target_id not in item_text_map and pd.notna(row.get('item_title', '')):
                item_text_map[target_id] = row['item_title']
    
    # 处理测试数据（如果有）
    if test_df is not None:
        for _, row in test_df.iterrows():
            hist_ids = parse_history(row["history_item_id"])
            hist_titles = parse_history(row.get("history_item_title", "[]"))
            
            for item_id, title in zip(hist_ids, hist_titles[:len(hist_ids)]):
                all_items.add(item_id)
                if item_id not in item_text_map and title:
                    item_text_map[item_id] = title
    
    # 创建映射
    all_items = sorted(all_items)
    item2idx = {item: idx + 1 for idx, item in enumerate(all_items)}   # 0 for PAD
    idx2item = {idx + 1: item for idx, item in enumerate(all_items)}
    
    log(f"Found {len(all_items)} unique items")
    log(f"Found {len(item_text_map)} items with text descriptions")
    
    # 提取文本特征
    extractor = TextFeatureExtractor()
    
    # 为所有商品准备文本
    unique_item_ids = sorted(item_text_map.keys())
    unique_texts = [item_text_map[item_id] for item_id in unique_item_ids]
    
    log(f"Extracting text features for {len(unique_texts)} items...")
    text_features = extractor.extract_features(unique_texts)
    log(f"Text features shape: {text_features.shape}")
    
    # 创建完整的文本特征矩阵（包含padding）
    num_items = len(all_items) + 1  # +1 for padding
    full_text_features = torch.zeros(num_items, TEXT_EMBED_DIM)
    
    # 填充文本特征
    for item_id, text_feat in zip(unique_item_ids, text_features):
        idx = item2idx[item_id]
        full_text_features[idx] = text_feat
    
    # 释放BERT模型内存
    del extractor
    torch.cuda.empty_cache()
    
    return item2idx, idx2item, full_text_features


# ======================
# 数据集定义
# ======================

def parse_history(history_str: str) -> List:
    """
    将 '[1, 2905, 3964, ...]' 这种字符串转成 list
    """
    if isinstance(history_str, list):
        return history_str
    history_str = str(history_str).strip()
    try:
        seq = ast.literal_eval(history_str)
        if isinstance(seq, list):
            return seq
        else:
            return []
    except Exception:
        return []


def encode_seq_to_fixed_len(seq_idx: List[int]) -> List[int]:
    """
    给定已经是 index 的序列 seq_idx，截断 / padding 到 MAX_SEQ_LEN
    """
    if len(seq_idx) > MAX_SEQ_LEN:
        seq_idx = seq_idx[-MAX_SEQ_LEN:]
    pad_len = MAX_SEQ_LEN - len(seq_idx)
    if pad_len > 0:
        seq_idx = [0] * pad_len + seq_idx
    return seq_idx


class TrainSeqDatasetWithText(Dataset):
    """
    训练集用的 Dataset，包含文本特征
    """
    def __init__(self, df: pd.DataFrame, item2idx: dict, text_features: torch.Tensor):
        self.samples = []  # list of (seq_idx, text_seq, target_idx)
        self.text_features = text_features
        
        for _, row in df.iterrows():
            history_raw = row["history_item_id"]
            history_titles_raw = row.get("history_item_title", "[]")
            
            seq = parse_history(history_raw)
            titles = parse_history(history_titles_raw)
            
            # 确保标题长度与序列长度匹配
            if len(titles) < len(seq):
                titles = titles + [""] * (len(seq) - len(titles))
            titles = titles[:len(seq)]
            
            # 转成 idx
            seq_idx_full = [item2idx.get(x, 0) for x in seq if x in item2idx]
            
            # 太短就跳过
            if len(seq_idx_full) < 2:
                continue
            
            # 限制最长 history
            if len(seq_idx_full) > MAX_HISTORY_LEN_FOR_TRAIN:
                seq_idx_full = seq_idx_full[-MAX_HISTORY_LEN_FOR_TRAIN:]
            
            # 有效的 target 位置
            positions = list(range(1, len(seq_idx_full)))
            
            # 如 positions 太多，随机采样
            if len(positions) > MAX_TRAIN_SAMPLES_PER_USER:
                positions = random.sample(positions, MAX_TRAIN_SAMPLES_PER_USER)
            
            # 生成样本
            for pos in positions:
                prefix_idx = seq_idx_full[:pos]
                target_idx = seq_idx_full[pos]
                
                # 获取文本特征
                prefix_text_features = []
                for item_idx in prefix_idx:
                    if item_idx == 0:
                        prefix_text_features.append(torch.zeros(TEXT_EMBED_DIM))
                    else:
                        prefix_text_features.append(self.text_features[item_idx])
                
                # 处理序列长度
                prefix_fixed = encode_seq_to_fixed_len(prefix_idx)
                
                # 处理文本特征序列
                if len(prefix_text_features) > MAX_SEQ_LEN:
                    prefix_text_features = prefix_text_features[-MAX_SEQ_LEN:]
                pad_len = MAX_SEQ_LEN - len(prefix_text_features)
                if pad_len > 0:
                    prefix_text_features = [torch.zeros(TEXT_EMBED_DIM)] * pad_len + prefix_text_features
                
                # 转换为tensor
                seq_tensor = torch.LongTensor(prefix_fixed)
                text_seq_tensor = torch.stack(prefix_text_features)  # [seq_len, text_dim]
                target_tensor = torch.LongTensor([target_idx])
                
                self.samples.append((seq_tensor, text_seq_tensor, target_tensor))
        
        log(f"TrainSeqDataset built with {len(self.samples)} samples.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class EvalSeqDatasetWithText(Dataset):
    """
    验证 / 测试 用的 Dataset，包含文本特征
    """
    def __init__(self, df: pd.DataFrame, item2idx: dict, text_features: torch.Tensor, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.item2idx = item2idx
        self.text_features = text_features
        self.is_train = is_train
    
    def __len__(self):
        return len(self.df)
    
    def _get_seq_and_text(self, seq: List[int]) -> Tuple[List[int], torch.Tensor]:
        # 转 index
        idx_seq = [self.item2idx.get(x, 0) for x in seq if x in self.item2idx]
        
        # 处理文本特征
        text_seq = []
        for idx in idx_seq:
            if idx == 0:
                text_seq.append(torch.zeros(TEXT_EMBED_DIM))
            else:
                text_seq.append(self.text_features[idx])
        
        # 截断和padding
        idx_seq_fixed = encode_seq_to_fixed_len(idx_seq)
        
        if len(text_seq) > MAX_SEQ_LEN:
            text_seq = text_seq[-MAX_SEQ_LEN:]
        pad_len = MAX_SEQ_LEN - len(text_seq)
        if pad_len > 0:
            text_seq = [torch.zeros(TEXT_EMBED_DIM)] * pad_len + text_seq
        
        text_seq_tensor = torch.stack(text_seq)  # [seq_len, text_dim]
        
        return idx_seq_fixed, text_seq_tensor
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        history_raw = row["history_item_id"]
        seq = parse_history(history_raw)
        
        seq_idx, text_seq = self._get_seq_and_text(seq)
        seq_idx = torch.LongTensor(seq_idx)
        
        if self.is_train:
            target_item = int(row["item_id"])
            target_idx = self.item2idx.get(target_item, 0)
            target_idx = torch.LongTensor([target_idx])
            return seq_idx, text_seq, target_idx
        else:
            user_id = int(row["user_id"])
            return seq_idx, text_seq, user_id


# ======================
# Transformer 模型定义（带文本特征）
# ======================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class TransformerRecModelWithText(nn.Module):
    """
    Transformer Encoder 带文本特征
    """
    def __init__(
        self,
        num_items: int,
        text_features: torch.Tensor,
        embed_dim: int = EMBED_DIM,
        text_embed_dim: int = TEXT_EMBED_DIM,
        combined_dim: int = COMBINED_DIM,
        nhead: int = N_HEAD,
        num_layers: int = NUM_ENCODER_LAYERS,
        dim_feedforward: int = FFN_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.num_items = num_items
        
        # Item ID embedding
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        
        # 文本特征（冻结，不训练）
        self.register_buffer('text_features', text_features)
        self.text_projection = nn.Linear(text_embed_dim, embed_dim)
        
        # 特征融合
        self.fusion_projection = nn.Linear(embed_dim * 2, combined_dim)
        
        # Transformer
        self.pos_encoding = PositionalEncoding(combined_dim, max_len=MAX_SEQ_LEN)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=combined_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.output_projection = nn.Linear(combined_dim, embed_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.xavier_uniform_(self.fusion_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.text_projection.bias)
        nn.init.zeros_(self.fusion_projection.bias)
        nn.init.zeros_(self.output_projection.bias)
    
    def encode_sequence(self, seq_idx: torch.Tensor, text_seq: torch.Tensor) -> torch.Tensor:
        """
        编码序列
        seq_idx: [batch, seq_len]
        text_seq: [batch, seq_len, text_dim]
        返回: [batch, embed_dim]
        """
        batch_size, seq_len = seq_idx.shape
        
        # ID embedding
        id_emb = self.item_embedding(seq_idx)  # [batch, seq_len, embed_dim]
        
        # 文本特征
        text_emb = self.text_projection(text_seq)  # [batch, seq_len, embed_dim]
        
        # 特征融合
        combined = torch.cat([id_emb, text_emb], dim=-1)  # [batch, seq_len, embed_dim*2]
        combined = self.fusion_projection(combined)  # [batch, seq_len, combined_dim]
        
        # 位置编码
        combined = self.pos_encoding(combined)
        combined = self.dropout(combined)
        
        # 注意力mask
        pad_mask = seq_idx.eq(0)  # [batch, seq_len]
        
        # Transformer
        encoded = self.transformer_encoder(combined, src_key_padding_mask=pad_mask)
        
        # 取最后一个非padding位置
        lengths = (~pad_mask).sum(dim=1)  # [batch]
        last_indices = (lengths - 1).clamp(min=0)
        user_repr = encoded[torch.arange(batch_size), last_indices]  # [batch, combined_dim]
        
        # 投影到embed_dim空间
        user_repr = self.output_projection(user_repr)  # [batch, embed_dim]
        
        return user_repr
    
    def forward(self, seq_idx: torch.Tensor, text_seq: torch.Tensor) -> torch.Tensor:
        """
        返回对所有 item 的打分 logits
        seq_idx: [batch, seq_len]
        text_seq: [batch, seq_len, text_dim]
        返回: [batch, num_items+1]
        """
        user_repr = self.encode_sequence(seq_idx, text_seq)  # [batch, embed_dim]
        item_emb_weight = self.item_embedding.weight  # [num_items+1, embed_dim]
        logits = torch.matmul(user_repr, item_emb_weight.t())  # [batch, num_items+1]
        return logits


# ======================
# 训练与评估
# ======================

def train_one_epoch(model, dataloader, optimizer, criterion, epoch: int):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, (seq_idx, text_seq, target_idx) in enumerate(dataloader):
        seq_idx = seq_idx.to(DEVICE)
        text_seq = text_seq.to(DEVICE)
        target_idx = target_idx.squeeze(1).to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(seq_idx, text_seq)
        
        loss = criterion(logits, target_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()
        
        bs = seq_idx.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataloader):
            log(
                f"Epoch {epoch} | Step {batch_idx+1}/{len(dataloader)} "
                f"| Loss {loss.item():.4f}"
            )
    
    avg_loss = total_loss / max(total_samples, 1)
    log(f"Epoch {epoch} finished. Train loss: {avg_loss:.4f}")
    return avg_loss


def evaluate_mrr_at_k(model, dataloader, k: int = 10) -> float:
    model.eval()
    mrr_sum = 0.0
    user_count = 0
    
    with torch.no_grad():
        for seq_idx, text_seq, target_idx in dataloader:
            seq_idx = seq_idx.to(DEVICE)
            text_seq = text_seq.to(DEVICE)
            target_idx = target_idx.squeeze(1).to(DEVICE)
            
            logits = model(seq_idx, text_seq)
            logits[:, 0] = -1e9  # 屏蔽 PAD
            
            _, topk_idx = torch.topk(logits, k=k, dim=1)
            
            batch_size = seq_idx.size(0)
            user_count += batch_size
            
            topk_idx = topk_idx.cpu().numpy()
            target_idx_np = target_idx.cpu().numpy()
            
            for i in range(batch_size):
                t = target_idx_np[i]
                if t == 0:
                    continue
                rec_list = topk_idx[i].tolist()
                if t in rec_list:
                    rank = rec_list.index(t) + 1
                    mrr_sum += 1.0 / rank
    
    mrr = mrr_sum / max(user_count, 1)
    log(f"MRR@{k}: {mrr:.6f}")
    return mrr


# ======================
# 主流程
# ======================

def main():
    global logger
    
    run_dir = get_run_dir()
    log_path = run_dir / "log.txt"
    logger = Logger(log_path)
    log(f"Run directory: {run_dir}")
    log(f"Using device: {DEVICE}")
    
    # 1. 加载数据
    log("Loading train data...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df["user_id"] = train_df["user_id"].astype(int)
    train_df["item_id"] = train_df["item_id"].astype(int)
    
    # 2. 加载测试数据（用于构建完整的item映射）
    log("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)
    test_df["user_id"] = test_df["user_id"].astype(int)
    
    # 3. 构建item映射和文本特征
    log("Building item mapping with text features...")
    item2idx, idx2item, text_features = build_item_mapping_with_text(train_df, test_df)
    num_items = len(item2idx)
    log(f"Number of unique items: {num_items}")
    
    # 4. 划分 train / valid
    train_ratio = 0.9
    n_total = len(train_df)
    n_train = int(n_total * train_ratio)
    train_data = train_df.iloc[:n_train].reset_index(drop=True)
    valid_data = train_df.iloc[n_train:].reset_index(drop=True)
    
    # 5. 构建 Dataset & DataLoader
    log("Building datasets...")
    train_dataset = TrainSeqDatasetWithText(train_data, item2idx, text_features)
    valid_dataset = EvalSeqDatasetWithText(valid_data, item2idx, text_features, is_train=True)
    
    def collate_fn(batch):
        seq_idx_batch = torch.stack([item[0] for item in batch])
        text_seq_batch = torch.stack([item[1] for item in batch])
        target_batch = torch.stack([item[2] for item in batch])
        return seq_idx_batch, text_seq_batch, target_batch
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    
    # 6. 模型 + 优化器
    log("Building Transformer model with text features...")
    model = TransformerRecModelWithText(
        num_items=num_items,
        text_features=text_features,
        embed_dim=EMBED_DIM,
        text_embed_dim=TEXT_EMBED_DIM,
        combined_dim=COMBINED_DIM,
        nhead=N_HEAD,
        num_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=FFN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total parameters: {total_params:,}")
    log(f"Trainable parameters: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LR, 
        weight_decay=WEIGHT_DECAY
    )
    
    # 7. 训练 + Early Stopping
    best_mrr = 0.0
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        mrr10 = evaluate_mrr_at_k(model, valid_loader, k=10)
        
        if mrr10 > best_mrr + 1e-5:
            best_mrr = mrr10
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log(f"New best model at epoch {epoch}, MRR@10={mrr10:.6f}")
        else:
            epochs_no_improve += 1
            log(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= PATIENCE:
                log(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best epoch: {best_epoch}, Best MRR@10={best_mrr:.6f}"
                )
                break
    
    # 8. 保存最优模型
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        best_model_path = run_dir / "best_model.pth"
        torch.save({
            'model_state_dict': best_state,
            'item2idx': item2idx,
            'idx2item': idx2item,
            'text_features': text_features,
            'config': {
                'embed_dim': EMBED_DIM,
                'text_embed_dim': TEXT_EMBED_DIM,
                'combined_dim': COMBINED_DIM,
            }
        }, best_model_path)
        log(f"Best model saved to: {best_model_path}")
    else:
        log("Warning: best_state is None, model not saved!")
    
    # 9. Test 预测 & 生成 submission
    log("Making predictions on test data...")
    test_dataset = EvalSeqDatasetWithText(test_df, item2idx, text_features, is_train=False)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.stack([item[1] for item in batch]),
            torch.tensor([item[2] for item in batch])
        )
    )
    
    model.eval()
    all_user_ids = []
    all_pred_items = []
    
    with torch.no_grad():
        for seq_idx, text_seq, user_ids in test_loader:
            seq_idx = seq_idx.to(DEVICE)
            text_seq = text_seq.to(DEVICE)
            
            logits = model(seq_idx, text_seq)
            logits[:, 0] = -1e9
            
            _, topk_idx = torch.topk(logits, k=10, dim=1)
            topk_idx = topk_idx.cpu().numpy().tolist()
            
            for idx_list in topk_idx:
                clean_idx = [i for i in idx_list if i != 0]
                clean_idx = clean_idx[:10]
                pred_items = [idx2item[int(i)] for i in clean_idx]
                all_pred_items.append(str(pred_items))
            
            all_user_ids.extend([int(u) for u in user_ids])
    
    submission = pd.DataFrame({"user_id": all_user_ids, "item_id": all_pred_items})
    submission_path = run_dir / SUBMISSION_PATH
    submission.to_csv(submission_path, index=False)
    log(f"Submission file saved to: {submission_path}")
    log("Done!")


if __name__ == "__main__":
    main()