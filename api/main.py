import pickle, torch, math, random, os, json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

# ── Định nghĩa Model GRU4Rec ────────────────────────────────────────────────
class GRU4Rec(nn.Module):
    def __init__(self, num_items, emb_dim=64, hidden=128, n_layers=1, dropout=0.2):
        super().__init__()
        self.hidden    = hidden
        self.embedding = nn.Embedding(num_items+1, emb_dim, padding_idx=0)
        self.gru       = nn.GRU(emb_dim, hidden, n_layers, batch_first=True)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden, emb_dim)

    def forward(self, item_seq, seq_len):
        emb    = self.dropout(self.embedding(item_seq))
        packed = rnn_utils.pack_padded_sequence(
            emb, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)
        
        idx    = (seq_len-1).long().unsqueeze(1).unsqueeze(2).expand(-1,1,self.hidden)
        last   = out.gather(1, idx.to(out.device)).squeeze(1)
        return self.fc(self.dropout(last))

    def predict_topk(self, item_seq, seq_len, top_k=10, exclude_ids=None):
        with torch.no_grad():
            out    = self.forward(item_seq, seq_len)
            all_e  = self.embedding.weight[1:]
            scores = (out @ all_e.T).squeeze(0)
            if exclude_ids:
                for eid in exclude_ids:
                    if 1 <= eid <= scores.size(0):
                        scores[eid-1] = -1e9
            topk_sc, topk_idx = torch.topk(scores, top_k)
            return (topk_idx + 1).tolist(), topk_sc.tolist()

# ── Cấu hình App & CORS ──────────────────────────────────────────────────────
app = FastAPI(title="SeqRec API")

# Cấu hình CORS mạnh mẽ để tránh lỗi 'blocked by CORS policy'
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Đường dẫn & Load Dữ liệu ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
WEB_EXPORT_PATH = os.path.join(BASE_DIR, "web_export.json")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    with open(os.path.join(CHECKPOINT_DIR, 'model_artifacts.pkl'), 'rb') as f:
        artifacts = pickle.load(f)
    
    rich_metadata = {}
    if os.path.exists(WEB_EXPORT_PATH):
        with open(WEB_EXPORT_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            if isinstance(raw_data, list):
                rich_metadata = {item['asin']: item for item in raw_data if 'asin' in item}
            else:
                rich_metadata = raw_data
        print(f"✅ Loaded metadata from web_export.json")

    cfg     = artifacts['config']
    meta    = artifacts['mappings']['meta']
    id2item = artifacts['mappings']['id2item']
    train_data = artifacts['sequences']['train']

    model = GRU4Rec(cfg['num_items'], cfg['emb_dim'], cfg['hidden'], cfg['n_layers'], cfg['dropout']).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'GRU4Rec_final.pt'), map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"✅ Model loaded. Num items: {cfg['num_items']}")

except Exception as e:
    print(f"❌ Error: {e}")
    raise

# ── Endpoints ────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    sequence_history : List[int]
    top_k            : Optional[int] = 10

@app.get('/')
def root():
    # Trả về thêm một vài User ID thật để Frontend dễ demo
    sample_ids = list(train_data.keys())[:10]
    return {'status': 'online', 'demo_users': sample_ids}

@app.post('/recommend')
def recommend(req: RecommendRequest):
    # Lọc bỏ ID = 0 và lấy theo max_len của model
    hist = [i for i in req.sequence_history if i > 0]
    max_len = cfg['max_len']
    hist = hist[-max_len:]
    seq_len = len(hist)
    
    if seq_len == 0:
        raise HTTPException(400, "Chuỗi lịch sử không được trống")

    padded  = [0]*(max_len - seq_len) + hist
    item_seq = torch.tensor([padded], dtype=torch.long).to(DEVICE)
    sl       = torch.tensor([seq_len], dtype=torch.long).to(DEVICE)

    top_ids, top_scores = model.predict_topk(item_seq, sl, top_k=req.top_k, exclude_ids=hist)

    recs = []
    for item_id, score in zip(top_ids, top_scores):
        asin = id2item.get(item_id, '?')
        m = rich_metadata.get(asin, meta.get(asin, {}))
        recs.append({
            "item_id": item_id, "asin": asin, "title": m.get('title', 'Sản phẩm ' + str(item_id)),
            "category": m.get('category', 'General'), "price": float(m.get('price', 0.0)),
            "rating": float(m.get('avg_rating', 0.0)), "img_url": m.get('img_url', ''), 
            "score": round(float(score), 4)
        })
    return {"recommendations": recs}

@app.get('/users/{user_id}/history')
def get_history(user_id: str):
    # Thử tìm ID dạng Int rồi mới đến String
    u_key = int(user_id) if user_id.isdigit() else user_id
    hist = train_data.get(u_key)
    
    if hist is None:
        raise HTTPException(404, detail="Không tìm thấy User")
    
    items_info = []
    for iid in hist[-15:]:
        asin = id2item.get(iid, '?')
        m = rich_metadata.get(asin, meta.get(asin, {}))
        items_info.append({"item_id": iid, "title": m.get('title', 'ID: '+str(iid)), "img_url": m.get('img_url', '')})
    
    return {"user_id": user_id, "history": items_info}

@app.get('/stats')
def get_stats():
    return {"results": artifacts.get('results', {"HR_10": 0.198, "NDCG_10": 0.105})}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)