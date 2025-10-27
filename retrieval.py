# -*- coding: utf-8 -*-

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import hashlib
import torch
import csv
from collections import defaultdict

# ============ 可調參 ============
MODEL_SR = 44100
TOPK = 5
USE_GPU = True           # OOM 就改 False
USE_CHUNK = True        # OOM 就改 True（啟用切段編碼）
CHUNK_SEC = 10           # 切段長度（秒，啟用切段時才有用）
HOP_SEC = 10             # 段與段的位移（秒）
READ_CACHE_ONLY = True
# =============================

# 若你很在意完全不讓程式嘗試連網，可保持這兩行：
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "*"

# 路徑（沿用上一個 cell 設定的變數）
PROJECT_DIR = Path(__file__).resolve().parent
REF_DIR   = PROJECT_DIR / "referecne_music_list_60s"     # 參考曲庫資料夾
TGT_DIR   = PROJECT_DIR / "target_music_list_60s"        # 目標曲庫資料夾
CKPT_PATH =  PROJECT_DIR / "checkpoints" / "music2latent.pt"  # 你下載的權重（離線）
OUT_DIR   = PROJECT_DIR / "outputs"                  # 輸出資料夾

from music2latent import EncoderDecoder

def list_audio_files(d):
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    return [str(p) for p in sorted(Path(d).rglob("*")) if p.suffix.lower() in exts]

def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_audio_mono_resample(path, target_sr):
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    maxv = np.max(np.abs(y)) + 1e-12
    y = y / maxv
    return y, sr

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T

def build_encdec_offline(ckpt_path: str | Path, sr: int):
    if USE_GPU and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if ckpt_path and Path(ckpt_path).exists():
        print(f"[Init] 使用本地權重：{ckpt_path}")
        encdec = EncoderDecoder(load_path_inference=str(ckpt_path), device=device)
    else:
        print("[Init] 未提供本地權重，將使用預設（需已在本機快取）")
        encdec = EncoderDecoder(device=device)
    encdec.sr = sr
    return encdec

# ---- 可選：切段推論（遇到 OOM 再啟用） ----
def encode_chunk_avg_latent(wav_1d: np.ndarray, encdec, sr: int,
                            chunk_sec=10, hop_sec=10) -> np.ndarray:
    CHUNK_SAM = int(chunk_sec * sr)
    HOP_SAM = int(hop_sec * sr)
    vecs_torch = []  # ← 收集 torch 向量
    for start in range(0, len(wav_1d), HOP_SAM):
        seg = wav_1d[start:start+CHUNK_SAM]
        if len(seg) < int(0.2 * CHUNK_SAM):  # 太短就丟掉
            break
        seg = np.ascontiguousarray(seg, dtype=np.float32)
        seg_t = torch.tensor(seg, dtype=torch.float32, device=encdec.device)
        with torch.no_grad():
            z = encdec.encode(seg_t)
        # if isinstance(z, torch.Tensor): ##
        #     z = z.detach().cpu().numpy()
        # if z.ndim == 3:
        #     z = z.reshape(-1, z.shape[-2], z.shape[-1])[0]
        # elif z.ndim != 2:
        #     raise RuntimeError(f"Unexpected latent shape: {z.shape}")
        # vec = z.mean(axis=-1)
        # vec = vec / (np.linalg.norm(vec) + 1e-9)
        # vecs.append(vec)
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32, device=encdec.device)
        elif isinstance(z, torch.Tensor):
            z = z.to(encdec.device, dtype=torch.float32)
        else:
            raise RuntimeError(f"Unexpected latent type: {type(z)}")
        if z.ndim == 3:
            z = z.reshape(-1, z.shape[-2], z.shape[-1])[0]
        elif z.ndim != 2:
            raise RuntimeError(f"Unexpected latent shape: {tuple(z.shape)}")
        vec_t = torch.mean(z, dim=-1)                          # (64,)
        vec_t = vec_t / (torch.norm(vec_t) + 1e-9)
        vecs_torch.append(vec_t)

        if encdec.device == "cuda":
            torch.cuda.empty_cache()
    if not vecs_torch:
        raise RuntimeError("No valid chunks to encode.")
    # V = np.stack(vecs, axis=0).mean(axis=0)
    # V = V / (np.linalg.norm(V) + 1e-9)
    # return V
    V_t = torch.stack(vecs_torch, dim=0).mean(dim=0)
    V_t = V_t / (torch.norm(V_t) + 1e-9)

    # 最後一刻才回 numpy
    return V_t.detach().cpu().numpy()

def main():
    safe_mkdir(OUT_DIR)
    latent_cache = Path(OUT_DIR) / "latents"
    safe_mkdir(latent_cache)

    encdec = build_encdec_offline(CKPT_PATH, MODEL_SR)

    ref_files = list_audio_files(REF_DIR)
    tgt_files = list_audio_files(TGT_DIR)
    if not ref_files:
        print(f"[Error] 參考資料夾沒有音檔：{REF_DIR}")
        return
    if not tgt_files:
        print(f"[Error] 目標資料夾沒有音檔：{TGT_DIR}")
        return

    def file_to_vec(path: str) -> np.ndarray:
        key = hashlib.md5(str(Path(path).resolve()).encode("utf-8")).hexdigest()
        npy_path = latent_cache / f"{key}.npy"
        if npy_path.exists():
            return np.load(npy_path)
        if READ_CACHE_ONLY:
            # True：只讀 .npy，沒檔就跳過；False：沒檔就去算(照理來說前面會回傳)
            print(f"[Error] Outputs夾沒有latent：{path}")
            return None

        y, sr = load_audio_mono_resample(path, MODEL_SR)

        if USE_CHUNK:
            vec = encode_chunk_avg_latent(y, encdec, sr, CHUNK_SEC, HOP_SEC)
        else:
            y = np.ascontiguousarray(y, dtype=np.float32)
            y_t = torch.tensor(y, dtype=torch.float32, device=encdec.device)
            with torch.no_grad():
                z = encdec.encode(y_t)
            # if isinstance(z, torch.Tensor):
            #     z = z.cpu().numpy()
            if isinstance(z, np.ndarray):
                z = torch.tensor(z, dtype=torch.float32, device=encdec.device)
            elif isinstance(z, torch.Tensor):
                z = z.to(encdec.device, dtype=torch.float32)
            else:
                raise RuntimeError(f"Unexpected latent type: {type(z)}")
            if z.ndim == 3:
                z = z.reshape(-1, z.shape[-2], z.shape[-1])[0]
            elif z.ndim == 2:
                pass
            else:
                raise RuntimeError(f"Unexpected latent shape: {z.shape}")
            # vec = z.mean(axis=-1)
            # vec = vec / (np.linalg.norm(vec) + 1e-9)
            vec_t = torch.mean(z, dim=-1)
            vec_t = vec_t / (torch.norm(vec_t) + 1e-9)
            vec = vec_t.detach().cpu().numpy()  # 存檔才回 numpy

        np.save(npy_path, vec)
        return vec

    print("[Emb] Reference embeddings ...")
    ref_vecs, ref_names = [], []
    for f in tqdm(ref_files):
        v = file_to_vec(f)
        if v is None: 
            continue
        ref_vecs.append(v)
        ref_names.append(Path(f).name)
    
    np.save(Path(OUT_DIR)/"ref_vectors.npy", np.stack(ref_vecs, 0))
    with open(Path(OUT_DIR)/"ref_filenames.txt", "w") as f:
        f.write("\n".join(ref_names))

    print("[Emb] Target embeddings ...")
    tgt_vecs, tgt_names = [], []
    for f in tqdm(tgt_files):
        v = file_to_vec(f)
        if v is None: 
            continue
        tgt_vecs.append(v)
        tgt_names.append(Path(f).name)

    np.save(Path(OUT_DIR)/"tgt_vectors.npy", np.stack(tgt_vecs, 0))
    with open(Path(OUT_DIR)/"tgt_filenames.txt", "w") as f:
        f.write("\n".join(tgt_names))

    if len(ref_vecs) == 0 or len(tgt_vecs) == 0:
        print("[Error] 沒有可用的快取向量（請先關閉 READ_CACHE_ONLY 跑一次建立 .npy）")
        return

    ref_arr = np.stack(ref_vecs, axis=0)
    tgt_arr = np.stack(tgt_vecs, axis=0)

    S = cosine_sim(tgt_arr, ref_arr)
    topk = min(TOPK, S.shape[1])

    rows = []
    for i, tname in enumerate(tgt_names):
        idx = np.argsort(-S[i])[:topk]
        for rank, j in enumerate(idx, 1):
            cos_ij = float(np.asarray(S[i, j]).item())
            rows.append({
                "target": str(tname),
                "ref": str(ref_names[j]),
                "rank": int(rank),
                "cosine": cos_ij
            })

    for k in ("target", "ref", "rank", "cosine"):
        bad = [type(r.get(k)) for r in rows if not isinstance(r.get(k), (str, int, float))]
    if bad:
        print("[Warn] column", k, "has weird types:", bad[:3])
    csv_path = Path(OUT_DIR) / "retrieval_topk.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "ref", "rank", "cosine"])  # header
        for r in rows:
            # 保險轉成純 Python 標量，避免 numpy scalar/ndarray 之類
            target = "" if r.get("target") is None else str(r.get("target"))
            ref    = "" if r.get("ref")    is None else str(r.get("ref"))
            try:
                rank = int(r.get("rank"))
            except Exception:
                rank = 0
            try:
                cosine = float(r.get("cosine"))
            except Exception:
                cosine = 0.0
            writer.writerow([target, ref, rank, cosine])

    # 2) 整理 JSON（依 target 分組並照 rank 排序）
    by_target = defaultdict(list)
    for r in rows:
        try:
            t = "" if r.get("target") is None else str(r.get("target"))
            ref = "" if r.get("ref") is None else str(r.get("ref"))
            rank = int(r.get("rank"))
            cosine = float(r.get("cosine"))
        except Exception:
            # 若有髒資料就跳過
            continue
        by_target[t].append((rank, ref, cosine))

    # 若想只輸出出現在 tgt_names 的 target，維持原本語意：
    best = {}
    for t in tgt_names:
        lst = by_target.get(t, [])
        lst.sort(key=lambda x: x[0])  # 依 rank 升冪
        best[t] = [{"ref": ref, "cosine": cosine} for _, ref, cosine in lst]

    # 若想輸出所有 rows 裡的 target（不侷限於 tgt_names），改成：
    # best = {t: [{"ref": ref, "cosine": cosine} for _, ref, cosine in sorted(lst)] 
    #         for t, lst in by_target.items()}

    json_path = Path(OUT_DIR) / "retrieval_topk.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print(f"[OK] 已輸出：{csv_path}")
    print(f"[OK] 已輸出：{json_path}")

if __name__ == "__main__":
    main()
