import os
import json
import time
from pathlib import Path
from openai import OpenAI

# config
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_NAME = "google/gemini-2.5-flash"
INPUT_FILE = Path("dataset") / "merged_all_posts.json"
OUTPUT_FILE = Path("dataset") / "labeled_all_posts.json"
REQUEST_DELAY = 1.0
MAX_POSTS_TO_LABEL = None


def get_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


def clean_response(response_text):
    text = response_text.strip().title()

    # check for exact matches
    if text in ["Left", "Right", "Neutral"]:
        return text

    # fallback: check if the word is in the response
    lower_text = text.lower()
    if "left" in lower_text:
        return "Left"
    if "right" in lower_text:
        return "Right"
    if "neutral" in lower_text:
        return "Neutral"

    # if uable to parse model output, mark as unknown
    print(f"could not parse response: '{response_text}'")
    return "error: unknown model output"


def get_classification(client, post_text, topic):
    system_prompt = (
        "You are an expert political analyst. Your task is to classify the "
        f"political stance of a social media post regarding the specific topic of: '{topic}'."
    )

    user_prompt = f"""
    Post text:
    "{post_text}"

    Based on the post text, classify its political stance on the topic of '{topic}'.
    Respond with ONLY one of the following words:
    - Left: The post expresses a clear left-leaning or progressive stance on the topic.
    - Right: The post expresses a clear right-leaning or conservative stance on the topic.
    - Neutral: The post is purely factual, balanced, or does not take a clear political stance on the topic.
    """

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,  # more deterministic output
            max_tokens=10,
        )
        response_text = completion.choices[0].message.content
        return clean_response(response_text)

    except Exception as e:
        print(f"exception during api call: {e}")
        return f"error: {e}"


def main():
    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY environment variable not set.")
        return

    print(f"using model: {MODEL_NAME}")

    # load all posts
    try:
        all_posts = json.loads(INPUT_FILE.read_text("utf-8"))
        print(f"loaded {len(all_posts)} total posts from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"input file not found at {INPUT_FILE}")
        return
    except Exception as e:
        print(f"error loading {INPUT_FILE}: {e}")
        return

    # load already labeled posts to resume progress
    labeled_posts = []
    if OUTPUT_FILE.exists():
        try:
            labeled_posts = json.loads(OUTPUT_FILE.read_text("utf-8"))
            print(f"found {len(labeled_posts)} already-labeled posts in {OUTPUT_FILE}")
        except Exception as e:
            print(f"could not load existing output file, error: {e}")
            labeled_posts = []

    # create a set of seen post IDs for fast lookup
    seen_ids = {p.get("id") or p.get("uri") for p in labeled_posts}

    # filter out posts that are already seen
    posts_to_process = [
        p for p in all_posts if (p.get("id") or p.get("uri")) not in seen_ids
    ]

    print(f"found {len(posts_to_process)} new posts to label.")

    # apply the MAX_POSTS_TO_LABEL limit if set
    if MAX_POSTS_TO_LABEL is not None:
        posts_to_process = posts_to_process[:MAX_POSTS_TO_LABEL]
        print(f"processing {len(posts_to_process)} posts.")

    if not posts_to_process:
        print("processing all posts.")
        return

    client = get_client()
    total_to_process = len(posts_to_process)

    for i, post in enumerate(posts_to_process):
        # extract text (using the fix we found earlier) and topic
        post_text = post.get("record", {}).get("text")
        topic = post.get("__meta__", {}).get("topic", "general politics")

        if not post_text:
            print(f"  skipping post {i+1}/{total_to_process} (no text found)")
            continue

        print(f"--- labeling post {i+1}/{total_to_process} (topic: {topic}) ---")

        # get the model output
        label = get_classification(client, post_text, topic)
        print(f"  result: {label}")

        # add the new label to the post's metadata
        post["__meta__"]["llm_label"] = label

        # add the processed post to our list
        labeled_posts.append(post)

        # save progress every 25 posts
        if (i + 1) % 25 == 0 or (i + 1) == total_to_process:
            print(f"\n... saving progress ({len(labeled_posts)} total labeled) ...\n")
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(labeled_posts, f, indent=2, ensure_ascii=False)

        # wait before the next API call
        time.sleep(REQUEST_DELAY)

    print(f"--- finished ---")
    print(f"saved {len(labeled_posts)} total labeled posts to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
