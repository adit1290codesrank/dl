import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import struct
import array
import os
import time

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, heads, depth, is_causal):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.is_causal = is_causal
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_model*4, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        # x shape: [B, seq_len]
        emb = self.embedding(x)
        emb = self.pos_encoder(emb)
        if self.is_causal:
            seq_len = x.size(1)
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
            out = self.transformer(emb, mask=mask, is_causal=True)
        else:
            out = self.transformer(emb)
        return out
        
    def forward_pooled(self, x_groups):
        # x_groups shape: [B, S, max_schema_toks]
        B, S, T = x_groups.shape
        flat_x = x_groups.view(B, S * T)
        emb = self.embedding(flat_x) # [B, S*T, d_model]
        
        # Mean pool every T tokens
        emb = emb.view(B, S, T, -1)
        # Create mask for valid tokens (not pad 0)
        mask = (x_groups != 0).float().unsqueeze(-1)
        sum_emb = (emb * mask).sum(dim=2)
        count = mask.sum(dim=2).clamp(min=1.0)
        pooled = sum_emb / count
        return pooled # [B, S, d_model]

class PointerAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, heads, batch_first=True)
        
    def forward(self, query, schema, schema_mask):
        # query: [B, seq_len, d_model]
        # schema: [B, S, d_model]
        # schema_mask: [B, S] (1 for valid, 0 for pad)
        
        # PyTorch MHA expects boolean key_padding_mask where True = ignore
        key_padding_mask = (schema_mask == 0)
        
        attn_output, attn_weights = self.mha(query, schema, schema, key_padding_mask=key_padding_mask, average_attn_weights=True)
        return attn_output, attn_weights

class SchemaRAGNet(nn.Module):
    def __init__(self, vocab_size, d_model, heads, depth):
        super().__init__()
        self.vocab_size = vocab_size
        self.query_encoder = TextEncoder(vocab_size, d_model, heads, depth, is_causal=True)
        self.schema_encoder = TextEncoder(vocab_size, d_model, heads, depth, is_causal=False)
        self.pointer_layer = PointerAttention(d_model, heads)
        self.final_ln = nn.LayerNorm(d_model)
        self.p_gen_proj = nn.Linear(d_model, 1)
        
        # Initialize p_gen bias to encourage generation initially
        nn.init.constant_(self.p_gen_proj.bias, 2.5)

    def forward(self, query_tokens, schema_tokens, schema_mask, schema_vocab_ids):
        B, seq_len = query_tokens.shape
        
        Q_emb = self.query_encoder(query_tokens)
        K_emb = self.schema_encoder.forward_pooled(schema_tokens)
        
        context, attn_weights = self.pointer_layer(Q_emb, K_emb, schema_mask)
        # Residual connection
        context = context + Q_emb
        
        ln_out = self.final_ln(context) # [B, seq_len, d_model]
        
        # Generator branch
        # Tie embeddings
        vocab_logits = F.linear(ln_out, self.query_encoder.embedding.weight)
        
        # Mask out schema tokens from vocab generation (id >= 50000)
        # Assuming schema tokens start at 50000
        vocab_logits[:, :, 50000:] = -1e9
        
        P_vocab = F.softmax(vocab_logits, dim=-1) # [B, seq_len, V]
        
        # Copy gate
        p_gen = torch.sigmoid(self.p_gen_proj(ln_out)) # [B, seq_len, 1]
        
        # Pointer branch
        # attn_weights is [B, seq_len, S]
        P_schema = torch.zeros_like(P_vocab)
        
        # Scatter attention weights into P_schema
        # schema_vocab_ids is [S]
        # We want to add attn_weights[:, :, s] to P_schema[:, :, schema_vocab_ids[s]]
        S = schema_vocab_ids.shape[0]
        vocab_indices = schema_vocab_ids.view(1, 1, S).expand(B, seq_len, S).long()
        P_schema.scatter_add_(2, vocab_indices, attn_weights)
        
        # Blend
        P_final = p_gen * P_vocab + (1.0 - p_gen) * P_schema
        
        # Return log probabilities for NLLLoss
        # Clamp to avoid log(0)
        P_final = P_final.clamp(min=1e-7)
        return torch.log(P_final)

def load_dataset(bin_path):
    with open(bin_path, "rb") as f:
        n_train = struct.unpack("i", f.read(4))[0]
        n_val = struct.unpack("i", f.read(4))[0]
        seq_len = struct.unpack("i", f.read(4))[0]
        vocab_size = struct.unpack("i", f.read(4))[0]
        schema_size = struct.unpack("i", f.read(4))[0]
        max_schema_toks = struct.unpack("i", f.read(4))[0]
        
        schema_vocab_ids = array.array('f')
        schema_vocab_ids.fromfile(f, schema_size)
        schema_vocab_ids = torch.tensor(schema_vocab_ids, dtype=torch.long)
        
        def read_set(n_samples):
            X = array.array('f')
            X.fromfile(f, n_samples * seq_len)
            X = torch.tensor(X, dtype=torch.long).view(n_samples, seq_len)
            
            Schema = array.array('f')
            Schema.fromfile(f, n_samples * schema_size * max_schema_toks)
            Schema = torch.tensor(Schema, dtype=torch.long).view(n_samples, schema_size, max_schema_toks)
            
            Y = array.array('f')
            Y.fromfile(f, n_samples * seq_len)
            Y = torch.tensor(Y, dtype=torch.long).view(n_samples, seq_len)
            
            return X, Schema, Y

        train_X, train_Schema, train_Y = read_set(n_train)
        val_X, val_Schema, val_Y = read_set(n_val)
        
        return {
            "n_train": n_train, "n_val": n_val, "seq_len": seq_len, 
            "vocab_size": vocab_size, "schema_size": schema_size, "max_schema_toks": max_schema_toks,
            "schema_vocab_ids": schema_vocab_ids,
            "train": (train_X, train_Schema, train_Y),
            "val": (val_X, val_Schema, val_Y)
        }

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    bin_path = os.path.join(os.path.dirname(__file__), "..", "data", "breakwalls.bin")
    if not os.path.exists(bin_path):
        print("Dataset not found. Please run prepare_breakwalls.py first.")
        return
        
    data = load_dataset(bin_path)
    
    train_dataset = torch.utils.data.TensorDataset(data["train"][0], data["train"][1], data["train"][2])
    val_dataset = torch.utils.data.TensorDataset(data["val"][0], data["val"][1], data["val"][2])
    
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    model = SchemaRAGNet(
        vocab_size=data["vocab_size"],
        d_model=256,
        heads=8,
        depth=4
    ).to(device)
    
    schema_vocab_ids = data["schema_vocab_ids"].to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-6)
    criterion = nn.NLLLoss(ignore_index=-100)
    
    epochs = 200
    max_lr = 2.5e-4
    warmup_epochs = 20
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        # Simple Linear Warmup
        if epoch <= warmup_epochs:
            lr = 1e-6 + (max_lr - 1e-6) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = max_lr # Keeping it simple for PyTorch
            
        t0 = time.time()
        for batch_idx, (X, Schema, Y) in enumerate(train_loader):
            X, Schema, Y = X.to(device), Schema.to(device), Y.to(device)
            
            schema_mask = (Schema[:, :, 0] != 0).float()
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                log_probs = model(X, Schema, schema_mask, schema_vocab_ids)
                
                # Reshape for NLLLoss
                log_probs = log_probs.view(-1, data["vocab_size"])
                Y_flat = Y.view(-1)
                
                loss = criterion(log_probs, Y_flat)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        correct_top1 = 0
        correct_top5 = 0
        total_valid = 0
        
        with torch.no_grad():
            for X, Schema, Y in val_loader:
                X, Schema, Y = X.to(device), Schema.to(device), Y.to(device)
                schema_mask = (Schema[:, :, 0] != 0).float()
                
                with torch.amp.autocast('cuda'):
                    log_probs = model(X, Schema, schema_mask, schema_vocab_ids)
                    
                    log_probs_flat = log_probs.view(-1, data["vocab_size"])
                    Y_flat = Y.view(-1)
                    
                    loss = criterion(log_probs_flat, Y_flat)
                    val_loss += loss.item()
                    
                    # Accuracies
                    valid_mask = (Y_flat != -100)
                    valid_targets = Y_flat[valid_mask]
                    if len(valid_targets) > 0:
                        valid_log_probs = log_probs_flat[valid_mask]
                        
                        _, top5_preds = valid_log_probs.topk(5, dim=-1)
                        top1_preds = top5_preds[:, 0]
                        
                        correct_top1 += (top1_preds == valid_targets).sum().item()
                        correct_top5 += (top5_preds == valid_targets.unsqueeze(1)).sum().item()
                        total_valid += len(valid_targets)
                        
        val_loss /= len(val_loader)
        top1_acc = (correct_top1 / total_valid) * 100 if total_valid > 0 else 0
        top5_acc = (correct_top5 / total_valid) * 100 if total_valid > 0 else 0
        
        t1 = time.time()
        print(f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Top1: {top1_acc:.2f}% | Top5: {top5_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {t1-t0:.2f}s")

if __name__ == "__main__":
    train()
