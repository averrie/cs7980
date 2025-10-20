import os
import re
import json
import glob
from collections import defaultdict
from datetime import datetime

DATASET_DIR = "dataset"
OUT_DIR = os.path.join(DATASET_DIR, "merged_by_topic")
GLOBAL_OUT = os.path.join(DATASET_DIR, "merged_all_posts.json")

os.makedirs(OUT_DIR, exist_ok=True)

def is_posts_file(path: str) -> bool:
    base = os.path.basename(path)
    return base.endswith("_posts.json") and not base.endswith("_keyword_counts.json")

def canonical_id(p):
    return p.get("id") or p.get("uri") or p.get("url")

DATE_RE = re.compile(r".*_(\d{8})")
def parse_date_from_folder(folder_name: str) -> str:
    m = DATE_RE.match(folder_name)
    if not m:
        return ""
    ds = m.group(1)
    try:
        return datetime.strptime(ds, "%Y%m%d").date().isoformat()
    except Exception:
        return ""

def main():
    files = [p for p in glob.glob(os.path.join(DATASET_DIR, "**", "*_posts.json"), recursive=True)
             if is_posts_file(p)]
    files.sort()
    print(f"Found {len(files)} topic-level posts files.")

    
    global_seen = set()

    
    by_topic = defaultdict(list)
    all_posts = []

    for path in files:
        folder = os.path.basename(os.path.dirname(path))
        source_date = parse_date_from_folder(folder)
        topic_slug = os.path.basename(path).replace("_posts.json", "")  # 文件名里就是话题 slug

        with open(path, "r", encoding="utf-8") as f:
            posts = json.load(f)

        added = 0
        for p in posts:
            pid = canonical_id(p)
            if not pid or pid in global_seen:
                continue
            global_seen.add(pid)

            
            meta = p.get("__meta__", {}) or {}
            meta["source_folder"] = folder
            if source_date and not meta.get("source_date"):
                meta["source_date"] = source_date
            
            if not meta.get("topic"):
                
                pretty = " ".join(topic_slug.split("_")).capitalize()
                meta["topic"] = meta.get("topic") or pretty

            p["__meta__"] = meta

            by_topic[topic_slug].append(p)
            all_posts.append(p)
            added += 1

        print(f"  merged {added} new posts from {path}")

    
    for topic_slug, items in by_topic.items():
        out_path = os.path.join(OUT_DIR, f"{topic_slug}_posts.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"✅ wrote {len(items):6d} posts -> {out_path}")

    
    with open(GLOBAL_OUT, "w", encoding="utf-8") as f:
        json.dump(all_posts, f, ensure_ascii=False, indent=2)
    print(f"\n✅ wrote GLOBAL {len(all_posts)} posts -> {GLOBAL_OUT}")

if __name__ == "__main__":
    main()
