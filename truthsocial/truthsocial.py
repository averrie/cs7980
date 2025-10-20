# truthsocial.py (English-only annotations, comments, and logs)
import os, json, time, random
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Iterable
from curl_cffi import requests
from dotenv import load_dotenv
import glob

load_dotenv()

TRUTHSOCIAL_TOKEN = os.environ.get("TRUTHSOCIAL_TOKEN")
TRUTHSOCIAL_CF_BM = os.environ.get("TRUTHSOCIAL_CF_BM")
TRUTHSOCIAL_CF_CLEARANCE = os.environ.get("TRUTHSOCIAL_CF_CLEARANCE")
TRUTHSOCIAL_MASTODON_SESSION = os.environ.get("TRUTHSOCIAL_MASTODON_SESSION")

MAX_POSTS = 1000
DATASET_DIR = "dataset"
API_URL = "https://truthsocial.com/api/v2/search"
REQUEST_LIMIT = 20
REQUEST_DELAY = 8

INITIAL_BACKOFF_429 = 30
INITIAL_BACKOFF_5XX = 5
MAX_BACKOFF = 600
MAX_RETRIES = 8
JITTER_RANGE = (0.5, 1.5)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

TOPIC_FRAMES: Dict[str, Dict[str, List[str]]] = {}

def load_topic_frames() -> Dict[str, Dict[str, List[str]]]:
    try:
        from topic_frames import TOPIC_FRAMES as TF
        if isinstance(TF, dict):
            print("Loaded TOPIC_FRAMES from topic_frames.py")
            return TF
    except Exception:
        pass
    try:
        with open("topic_frames.json", "r", encoding="utf-8") as f:
            jf = json.load(f)
        if isinstance(jf, dict):
            print("Loaded TOPIC_FRAMES from topic_frames.json")
            return jf
    except Exception:
        pass
    print("WARNING: TOPIC_FRAMES not found; using a minimal fallback.")
    return {
        "Free Speech and Content Regulation": {
            "left":    ["hate speech", "content moderation", "misinformation"],
            "right":   ["free speech", "censorship", "cancel culture", "Big Tech censorship"],
            "neutral": ["First Amendment", "online speech", "platform policy"]
        }
    }

def make_session() -> requests.Session:
    if not TRUTHSOCIAL_TOKEN:
        raise RuntimeError("TRUTHSOCIAL_TOKEN missing")
    if not TRUTHSOCIAL_MASTODON_SESSION:
        raise RuntimeError("TRUTHSOCIAL_MASTODON_SESSION missing")

    s = requests.Session(impersonate="chrome124", timeout=30)
    s.headers.update({
        "User-Agent": UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://truthsocial.com",
        "Referer": "https://truthsocial.com/search",
        "Authorization": "Bearer " + TRUTHSOCIAL_TOKEN,
    })
    s.cookies.set("_mastodon_session", TRUTHSOCIAL_MASTODON_SESSION, domain="truthsocial.com")
    if TRUTHSOCIAL_CF_BM:
        s.cookies.set("__cf_bm", TRUTHSOCIAL_CF_BM, domain="truthsocial.com")
    if TRUTHSOCIAL_CF_CLEARANCE:
        s.cookies.set("cf_clearance", TRUTHSOCIAL_CF_CLEARANCE, domain="truthsocial.com")
    return s

def parse_retry_after(headers) -> Optional[int]:
    ra = headers.get("Retry-After")
    if not ra:
        return None
    try:
        return int(ra)
    except Exception:
        return None

def with_jitter(seconds: float) -> float:
    return seconds * random.uniform(*JITTER_RANGE)

def get_with_retry(session: requests.Session, url: str, params: dict,
                   out_err_path: str) -> Tuple[Optional[requests.Response], Optional[str]]:
    backoff_429 = INITIAL_BACKOFF_429
    backoff_5xx = INITIAL_BACKOFF_5XX
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, http_version="v2")
        except Exception as e:
            wait = with_jitter(min(backoff_5xx, MAX_BACKOFF))
            print(f"[retry {attempt}/{MAX_RETRIES}] network error: {e} -> sleep {int(wait)}s")
            time.sleep(wait)
            backoff_5xx = min(backoff_5xx * 2, MAX_BACKOFF)
            continue
        if r.status_code == 200:
            return r, None
        if r.status_code == 429:
            ra = parse_retry_after(r.headers) or backoff_429
            wait = with_jitter(min(ra, MAX_BACKOFF))
            print(f"[retry {attempt}/{MAX_RETRIES}] 429 Too Many Requests -> sleep {int(wait)}s")
            time.sleep(wait)
            backoff_429 = min(max(backoff_429 * 2, 60), MAX_BACKOFF)
            continue
        if 500 <= r.status_code <= 599:
            wait = with_jitter(min(backoff_5xx, MAX_BACKOFF))
            print(f"[retry {attempt}/{MAX_RETRIES}] {r.status_code} server error -> sleep {int(wait)}s")
            time.sleep(wait)
            backoff_5xx = min(backoff_5xx * 2, MAX_BACKOFF)
            continue
        try:
            with open(out_err_path, "wb") as f:
                f.write(r.content)
        except Exception:
            pass
        return None, f"HTTP {r.status_code} (body saved to {out_err_path})"
    try:
        if 'r' in locals() and isinstance(r, requests.Response):
            with open(out_err_path, "wb") as f:
                f.write(r.content)
    except Exception:
        pass
    return None, f"Failed after {MAX_RETRIES} retries (last status: {getattr(locals().get('r'), 'status_code', 'N/A')})"

def slugify(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")

def iter_keywords(frames: Dict[str, List[str]]) -> Iterable[str]:
    for side in ("left", "right", "neutral"):
        for kw in frames.get(side, []):
            yield kw

def annotate(post: dict, topic: str, matched_keyword: str) -> dict:
    meta = post.get("__meta__", {})
    meta.update({"topic": topic, "matched_keyword": matched_keyword, "platform": "truthsocial"})
    post["__meta__"] = meta
    return post

def canonical_post_id(post: dict) -> Optional[str]:
    return post.get("id") or post.get("uri") or post.get("url")

def fetch_all_for_keyword(session: requests.Session, topic: str, keyword: str,
                          run_output_dir: str, already_seen_ids: set,
                          remaining_cap_for_topic: int) -> List[dict]:
    collected: List[dict] = []
    offset = 0
    while len(collected) < remaining_cap_for_topic:
        print(f"  - fetching '{keyword}'... (collected this kw: {len(collected)}, offset: {offset})")
        params = {"q": keyword, "limit": REQUEST_LIMIT, "offset": offset, "resolve": "true", "type": "statuses"}
        err_path = os.path.join(run_output_dir, f"err_{slugify(topic)}_{slugify(keyword)}_{offset}.html")
        resp, err = get_with_retry(session, API_URL, params, err_path)
        if err:
            print(f"    request failed for '{keyword}': {err}")
            break
        try:
            data = resp.json()
        except Exception as e:
            print(f"    JSON parse error for '{keyword}': {e}")
            break
        found = data.get("statuses", [])
        if not found:
            print(f"    no more posts for '{keyword}'")
            break
        added_this_page = 0
        for post in found:
            pid = canonical_post_id(post)
            if not pid or pid in already_seen_ids:
                continue
            annotate(post, topic, keyword)
            collected.append(post)
            already_seen_ids.add(pid)
            added_this_page += 1
            if len(collected) >= remaining_cap_for_topic:
                break
        print(f"    added {added_this_page} new; total for this kw: {len(collected)}")
        offset += REQUEST_LIMIT
        time.sleep(REQUEST_DELAY)
    return collected

SEEN_INDEX_PATH = os.path.join("/Users/yukkihsu/Desktop/Assignment/cs7980/truthsocial", "truthsocial_seen_ids.txt")

def canonical_post_id(post: dict) -> Optional[str]:
    return post.get("id") or post.get("uri") or post.get("url")

def load_seen_ids(path: str) -> set:
    seen = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seen.add(line)
    print(f"Loaded {len(seen)} seen ids from index: {path}")
    return seen

def append_seen_ids(path: str, new_ids: set) -> None:
    if not new_ids:
        return
    with open(path, "a", encoding="utf-8") as f:
        for _id in new_ids:
            f.write(_id + "\n")
    print(f"Appended {len(new_ids)} new ids to index: {path}")

def main():
    # ---- 1) env check ----
    missing = []
    if not TRUTHSOCIAL_TOKEN:
        missing.append("TRUTHSOCIAL_TOKEN")
    if not TRUTHSOCIAL_MASTODON_SESSION:
        missing.append("TRUTHSOCIAL_MASTODON_SESSION")
    if missing:
        print("Missing env:", ", ".join(missing))
        return

    # ---- 2) load topic frames ----
    global TOPIC_FRAMES
    TOPIC_FRAMES = load_topic_frames()

    # ---- 3) HTTP session ----
    session = make_session()
    print("requests session initialized (impersonate=chrome124)")
    print("set _mastodon_session cookie (domain=truthsocial.com)")
    if TRUTHSOCIAL_CF_BM:
        print("set __cf_bm cookie (optional)")
    if TRUTHSOCIAL_CF_CLEARANCE:
        print("set cf_clearance cookie (optional)")

    # ---- 4) load global seen ids (for incremental crawl) ----
    global_seen_ids = load_seen_ids(SEEN_INDEX_PATH)

    # ---- 5) run output dir ----
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_output_dir = os.path.join(DATASET_DIR, f"truthsocial_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    # ---- 6) per-topic crawl with dedupe against global + per-run ----
    for topic, frames in TOPIC_FRAMES.items():
        print(f"\n--- topic: '{topic}' ---")
        topic_slug = slugify(topic)
        topic_posts: List[dict] = []

        keywords = list(iter_keywords(frames))
        if not keywords:
            print("  (no keywords defined; skipping)")
            continue

        # Make a working dedupe set starting from global seen (so fetch() skips them)
        # We'll also use this to prevent duplicates across keywords in THIS run.
        dedupe_ids = set(global_seen_ids)

        # Evenly split cap across keywords to avoid one term dominating
        per_kw_cap = max(1, (MAX_POSTS + len(keywords) - 1) // len(keywords))

        for kw in keywords:
            if len(topic_posts) >= MAX_POSTS:
                break
            remaining_cap = MAX_POSTS - len(topic_posts)
            cap_for_kw = min(per_kw_cap, remaining_cap)

            posts_kw = fetch_all_for_keyword(
                session=session,
                topic=topic,
                keyword=kw,
                run_output_dir=run_output_dir,
                already_seen_ids=dedupe_ids,           # <- contains global + this-run ids
                remaining_cap_for_topic=cap_for_kw
            )
            topic_posts.extend(posts_kw)

        if topic_posts:
            # ---- save merged posts for this topic ----
            out_json = os.path.join(run_output_dir, f"{topic_slug}_posts.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(topic_posts, f, ensure_ascii=False, indent=2)
            print(f"saved {len(topic_posts)} posts to '{out_json}'")

            # ---- write counts by matched keyword (quick balance check) ----
            counts = {}
            for p in topic_posts:
                mk = p.get("__meta__", {}).get("matched_keyword", "UNKNOWN")
                counts[mk] = counts.get(mk, 0) + 1
            out_idx = os.path.join(run_output_dir, f"{topic_slug}_keyword_counts.json")
            with open(out_idx, "w", encoding="utf-8") as f:
                json.dump({"topic": topic, "counts_by_keyword": counts}, f, ensure_ascii=False, indent=2)
            print(f"wrote keyword distribution to '{out_idx}'")

            # ---- 7) append NEW ids (this topic) to global seen index ----
            new_ids = {canonical_post_id(p) for p in topic_posts if canonical_post_id(p)}
            # Only the truly new ones (exclude what's already in the index when we started)
            new_ids.difference_update(global_seen_ids)
            append_seen_ids(SEEN_INDEX_PATH, new_ids)

            # Update in-memory set so later topics in this same run also skip them
            global_seen_ids.update(new_ids)
        else:
            print("  no posts saved for this topic")

    print("\n--- finished ---")


if __name__ == "__main__":
    main()
