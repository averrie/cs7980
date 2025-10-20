import os
import json
import glob

DATASET_DIR = "/Users/yukkihsu/Desktop/Assignment/cs7980/truthsocial/dataset/truthsocial_20251013182649"       #dataset root
OUTPUT_PATH = os.path.join("/Users/yukkihsu/Desktop/Assignment/cs7980/truthsocial", "truthsocial_seen_ids.txt")

def canonical_post_id(post):
    return post.get("id") or post.get("uri") or post.get("url")

def collect_seen_ids(dataset_dir):
    seen = set()
    # scan all *_posts.json files recursively (in all timestamp folders)
    pattern = os.path.join(dataset_dir, "**", "*_posts.json")
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                posts = json.load(f)
            for p in posts:
                pid = canonical_post_id(p)
                if pid:
                    seen.add(pid)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
    print(f"Collected {len(seen)} unique post IDs.")
    return seen

def save_seen_ids(ids, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _id in ids:
            f.write(_id + "\n")
    print(f"Saved {len(ids)} IDs to {out_path}")

if __name__ == "__main__":
    ids = collect_seen_ids(DATASET_DIR)
    save_seen_ids(ids, OUTPUT_PATH)
