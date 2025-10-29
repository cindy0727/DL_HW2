from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parent
MUSIC_GEN_DIR = PROJECT_DIR / "task_2_result" / "ttm" / "Prompt_short_gen"  #記得改最後資料夾名稱

# 想要納入的音檔副檔名（可自行增減）
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}

# 輸出成 JSON Lines（每行一個 JSON 物件）
OUT_PATH = PROJECT_DIR / "task_2_result" / "audiobox" /"short_audio_paths.jsonl" #記得改最後檔案名稱

count = 0
with OUT_PATH.open("w", encoding="utf-8") as f:
    # rglob 會遞迴搜尋子資料夾；若不需要遞迴可改用 MUSIC_GEN_DIR.glob("*")
    for p in sorted(MUSIC_GEN_DIR.glob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            rec = {"path": str(p.resolve())}  # 產生絕對路徑；要相對路徑可改成 str(p.relative_to(PROJECT_DIR))
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")
            count += 1

print(f"已寫入 {OUT_PATH}，共 {count} 筆音檔。")