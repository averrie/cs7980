import os
import json
import time
from datetime import datetime
from atproto import Client, models

BLUESKY_HANDLE = os.environ.get("BLUESKY_HANDLE")
BLUESKY_PASSWORD = os.environ.get("BLUESKY_PASSWORD")

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

REQUEST_DELAY = 2


def main():
    if not BLUESKY_HANDLE or not BLUESKY_PASSWORD:
        print("BLUESKY_HANDLE and BLUESKY_PASSWORD environment variables are not set")
        return

    try:
        client = Client()
        client.login(BLUESKY_HANDLE, BLUESKY_PASSWORD)
        print(f"logged in as {client.me.handle}")
    except Exception as e:
        print(f"login error: {e}")
        return

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(DATASET_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    for topic in TOPICS:
        print(f"\n--- searching for topic: '{topic}' ---")
        posts = []
        cursor = None

        try:
            while len(posts) < MAX_POSTS:
                print(f"fetching posts... (collected: {len(posts)})")
                params = models.AppBskyFeedSearchPosts.Params(
                    q=topic, cursor=cursor, limit=100
                )
                response = client.app.bsky.feed.search_posts(params)

                if not response.posts:
                    print("no more posts found for this topic")
                    break

                # add fetched posts to posts list
                for post in response.posts:
                    # convert the post object to a dictionary
                    post_data = post.model_dump()

                    # construct url from the post's uri and author handle
                    # uri format: at://<user_id>/app.bsky.feed.post/<post_key>
                    # url format: https://bsky.app/profile/<user_handle>/post/<post_key>
                    if post.uri and post.author and post.author.handle:
                        post_key = post.uri.split("/")[-1]
                        handle = post.author.handle
                        post_data["url"] = (
                            f"https://bsky.app/profile/{handle}/post/{post_key}"
                        )

                    posts.append(post_data)

                cursor = response.cursor
                print(f"cursor updated to: {cursor}")

                # check if reached the end of search
                if not cursor:
                    print("reached the end of the search results")
                    break

                # timeout for prevent rate limited
                time.sleep(REQUEST_DELAY)

            if posts:
                filename = f"{topic.replace(' ', '_').lower()}_posts.json"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(posts, f, indent=4, ensure_ascii=False)

                print(f"saved {len(posts)} posts to '{filepath}'")

        except Exception as e:
            print(f"an error occurred while searching for '{topic}': {e}")
            continue  # move to the next topic

    print("\n--- finished ---")


if __name__ == "__main__":
    main()
