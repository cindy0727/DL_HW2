# pip install -U laion-clap torch torchaudio soundfile librosa
import torch, torchaudio, soundfile as sf
from laion_clap import CLAP_Module
from pathlib import Path
import json
from typing import Optional
import numpy as np

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 讓 PyTorch 看不到任何 GPU
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # debug 時可同步回報錯誤

DEVICE = "cpu"
TARGET_SR = 48000  # CLAP 預設 48k


def _ci_equal(a: str, b: str) -> bool:
    return a.casefold() == b.casefold()

def _search_one(root: Path, targets: list[str], recursive: bool) -> Optional[Path]:
    it = root.rglob("*") if recursive else root.glob("*")
    for p in it:
        if p.is_file() and any(_ci_equal(p.name, t) for t in targets):
            return p
    return None

def find_song(song_name: str, dir_T: str,  recursive: bool = False):

    T_targets = [f"{song_name}.mp3", f"{song_name}.wav"]

    T_path = _search_one(Path(dir_T), T_targets, recursive)

    return T_path

def load_mono_resampled(path, target_sr=TARGET_SR):
    wav, sr = torchaudio.load(str(path))        # (C, T)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:                        # 轉單聲道
        wav = wav.mean(dim=0, keepdim=True)
    # 正規化到 [-1, 1]（若來源非 float）
    wav = wav / max(wav.abs().max().item(), 1e-9)
    return wav, target_sr

def cosine(u, v):
    u = torch.nn.functional.normalize(u, dim=-1)
    v = torch.nn.functional.normalize(v, dim=-1)
    return (u * v).sum(dim=-1).item()

def compute_clap_sims(MODEL, target_audio, ref_audio):

    # 2) 音訊前處理 → 嵌入
    t_wav, _ = load_mono_resampled(target_audio)   # (1, T)
    r_wav, _ = load_mono_resampled(ref_audio)

    with torch.no_grad():
        a_emb_t = MODEL.get_audio_embedding_from_data(
            x=t_wav.to(DEVICE), use_tensor=True
        )  # (1, D)
        a_emb_r = MODEL.get_audio_embedding_from_data(
            x=r_wav.to(DEVICE), use_tensor=True
        )  # (1, D)
        
    # 3) 三種 cosine similarity
    sims = {
        "reference_audio__vs__target_audio":   cosine(a_emb_r, a_emb_t),
    }
    return sims

if __name__ == "__main__":
    # 換成你自己的路徑與文字
    PROJECT_DIR = Path(__file__).resolve().parent
    # REF_DIR   = PROJECT_DIR / "referecne_music_list_60s" 
    TARGET_DIR = PROJECT_DIR / "target_music_list_60s"
    Reference_DIR = PROJECT_DIR / "referecne_music_list_60s"
    SONGPAIR_FILEPATH = PROJECT_DIR / "retrieval_top2.json"
    OUTPUT_JSON = PROJECT_DIR / "CLAP_re_top2.json"

    print("download model")
    model = CLAP_Module(enable_fusion=False, device="cpu")
    model.load_ckpt()  # 下載預訓練權重（laion/clap-htsat-unfused）
    
    with open(SONGPAIR_FILEPATH, "r", encoding="utf-8") as f:
        names = json.load(f)
    print(f"共 {len(names)} 首，開始計算...")

    results = {}
    for tar, ref in names.items():
        tar_name = Path(tar).stem.strip()
        ref_name = Path(ref).stem.strip() 
        TARGET = find_song(tar_name, TARGET_DIR)
        REFERENCE = find_song(ref_name, Reference_DIR)
        
        print(f"Compute {tar_name} vs {ref_name}")
        sims = compute_clap_sims(model, TARGET, REFERENCE)
        print(tar_name, sims)
        results[tar_name] = sims
    
    # 若要存檔：
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {OUTPUT_JSON.resolve()}")
