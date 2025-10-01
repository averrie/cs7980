import os
import json
import time
from datetime import datetime
from curl_cffi import requests
from dotenv import load_dotenv

# config
load_dotenv()

# get the following from browser.
TRUTHSOCIAL_TOKEN = os.environ.get("TRUTHSOCIAL_TOKEN")
TRUTHSOCIAL_CF_CLEARANCE = os.environ.get("TRUTHSOCIAL_CF_CLEARANCE")
TRUTHSOCIAL_CF_BM = os.environ.get("TRUTHSOCIAL_CF_BM")
TRUTHSOCIAL_MASTODON_SESSION = os.environ.get("TRUTHSOCIAL_MASTODON_SESSION")

# 20 political topics (subject to change)
TOPICS = [
    "Abortion",
    "Gun Control",
    "Climate Change",
    "Healthcare",
    "Immigration",
    "LGBTQ+ Rights",
    "Voting Rights",
    "Economic Inequality",
    "Police Reform",
    "Freedom of Speech",
    "Affirmative Action",
    "Drug Policy",
    "Foreign Policy",
    "Tech Regulation",
    "Critical Race Theory",
    "Sex Education",
    "Universal Basic Income",
    "Trade Policy",
    "Social Security",
    "National Security",
]

# max number of posts to retrieve for each topic
MAX_POSTS = 2000
DATASET_DIR = "dataset"
API_URL = "https://truthsocial.com/api/v2/search"
REQUEST_LIMIT = 20
REQUEST_DELAY = 5
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:143.0) Gecko/20100101 Firefox/143.0"
)


def main():
    # validation
    if not TRUTHSOCIAL_TOKEN:
        print("TRUTHSOCIAL_TOKEN is not set")
        return

    if not TRUTHSOCIAL_CF_CLEARANCE:
        print("TRUTHSOCIAL_CF_CLEARANCE is not set")

    if not TRUTHSOCIAL_CF_BM:
        print("TRUTH_SOCIAL_CF_BM is not set")

    if not TRUTHSOCIAL_MASTODON_SESSION:
        print("TRUTH_SOCIAL_MASTODON_SESSION is not set")

    session = requests.Session()
    session.headers.update(
        {"User-Agent": USER_AGENT, "Authorization": "Bearer " + TRUTHSOCIAL_TOKEN}
    )
    print("requests session initialized")

    # set cookies
    if TRUTHSOCIAL_CF_BM:
        session.cookies.set("__cf_bm", TRUTHSOCIAL_CF_BM)
        print("set __cf_bm cookie")
    if TRUTHSOCIAL_MASTODON_SESSION:
        session.cookies.set("_mastodon_session", TRUTHSOCIAL_MASTODON_SESSION)
        print("set _mastodon_session cookie")
    if TRUTHSOCIAL_CF_CLEARANCE:
        session.cookies.set("cf_clearance", TRUTHSOCIAL_CF_CLEARANCE)
        print("set cf_clearance cookie")

    # output folder
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_output_dir = os.path.join(DATASET_DIR, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)

    for topic in TOPICS:
        print(f"\n--- searching for topic: '{topic}' ---")
        posts = []
        offset = 0

        try:
            while len(posts) < MAX_POSTS:
                print(f"fetching posts... (collected: {len(posts)}, offset: {offset})")

                params = {
                    "q": topic,
                    "limit": REQUEST_LIMIT,
                    "offset": offset,
                    "resolve": "true",
                    "type": "statuses",
                }

                response = session.get(API_URL, params=params)

                if response.status_code != 200:
                    print(f"error status code {response.status_code}")
                    print(f"response: {response.text}")
                    break

                data = response.json()
                found_posts = data.get("statuses", [])
                if not found_posts:
                    print("no more posts found for this topic")
                    break

                for post in found_posts:
                    posts.append(post)
                offset += REQUEST_LIMIT
                time.sleep(REQUEST_DELAY)

            if posts:
                filename = f"{topic.replace(' ', '_').lower()}_posts.json"
                filepath = os.path.join(run_output_dir, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(posts, f, indent=4, ensure_ascii=False)

                print(f"saved {len(posts)} posts to '{filepath}'")

        except Exception as e:
            print(f"an error occurred while searching for '{topic}': {e}")
            continue  # move to the next topic

    print("\n--- finished ---")


if __name__ == "__main__":
    main()
