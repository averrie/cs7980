import os
import json
import time
import re
from datetime import datetime
from atproto import Client, models

BLUESKY_HANDLE = os.environ.get("BLUESKY_HANDLE")
BLUESKY_PASSWORD = os.environ.get("BLUESKY_PASSWORD")

# 20 political topics (subject to change)
TOPICS = {
    "Abortion and Reproductive Policy": {
        "left": [
            "reproductive rights",
            "pro-choice",
            "abortion access",
            "women's rights",
        ],
        "right": ["pro-life", "unborn", "sanctity of life", "heartbeat bill"],
        "neutral": [
            "abortion law",
            "abortion policy",
            "Roe v. Wade",
            "Planned Parenthood",
        ],
    },
    "Gun Policy and Firearms Regulation": {
        "left": ["gun control", "assault weapons ban", "background checks"],
        "right": ["gun rights", "Second Amendment", "2A", "constitutional carry"],
        "neutral": ["firearms policy", "gun laws", "gun ownership"],
    },
    "Climate and Environmental Policy": {
        "left": ["climate crisis", "renewable energy", "green new deal"],
        "right": ["energy independence", "fossil fuels", "climate hoax"],
        "neutral": [
            "climate change",
            "carbon emissions",
            "environmental regulation",
            "Paris Agreement",
        ],
    },
    "Healthcare and Insurance Reform": {
        "left": ["universal healthcare", "Medicare for All", "public option"],
        "right": ["health freedom", "government overreach", "private insurance"],
        "neutral": [
            "healthcare reform",
            "Affordable Care Act",
            "health insurance policy",
        ],
    },
    "Immigration and Border Policy": {
        "left": ["immigrant rights", "DACA", "asylum seekers", "family separation"],
        "right": ["border crisis", "illegal immigration", "build the wall"],
        "neutral": ["immigration policy", "border security", "visa policy"],
    },
    "LGBTQ+ and Civil Rights": {
        "left": ["LGBTQ+ rights", "trans rights", "marriage equality"],
        "right": ["religious freedom", "parental rights", "traditional values"],
        "neutral": ["civil rights", "anti-discrimination", "gender identity policy"],
    },
    "Voting and Election Policy": {
        "left": ["voting rights", "voter suppression", "expand mail-in voting"],
        "right": ["election integrity", "voter fraud", "secure elections"],
        "neutral": ["election reform", "voter ID laws", "ballot access"],
    },
    "Economic Inequality and Taxation": {
        "left": ["wealth tax", "economic justice", "raise minimum wage"],
        "right": ["tax burden", "job creators", "free market"],
        "neutral": ["tax policy", "income inequality", "economic mobility"],
    },
    "Policing and Criminal Justice Reform": {
        "left": ["police reform", "defund the police", "mass incarceration"],
        "right": ["law and order", "back the blue", "crime wave"],
        "neutral": [
            "criminal justice reform",
            "public safety",
            "police accountability",
        ],
    },
    "Free Speech and Content Regulation": {
        "left": ["hate speech", "content moderation", "misinformation"],
        "right": ["free speech", "censorship", "cancel culture"],
        "neutral": ["First Amendment", "online speech", "platform policy"],
    },
    "Affirmative Action and Education Policy": {
        "left": ["affirmative action", "diversity in education", "racial equity"],
        "right": [
            "merit-based admissions",
            "colorblind policy",
            "reverse discrimination",
        ],
        "neutral": ["college admissions", "education policy", "Supreme Court decision"],
    },
    "Drug Policy and Substance Regulation": {
        "left": ["drug decriminalization", "harm reduction", "marijuana legalization"],
        "right": ["war on drugs", "fentanyl crisis", "tough on crime"],
        "neutral": ["drug policy", "opioid epidemic", "controlled substances"],
    },
    "Foreign Policy and National Defense": {
        "left": ["diplomacy", "humanitarian aid", "anti-war"],
        "right": ["military strength", "America First", "peace through strength"],
        "neutral": ["foreign policy", "NATO", "defense spending", "Ukraine war"],
    },
    "Technology and Internet Regulation": {
        "left": ["tech accountability", "data privacy", "AI regulation"],
        "right": ["free market innovation", "anti-censorship", "Big Tech bias"],
        "neutral": ["technology policy", "Section 230", "AI ethics"],
    },
    "Race and Education Curriculum": {
        "left": [
            "racial justice education",
            "anti-racism curriculum",
            "diversity training",
        ],
        "right": [
            "critical race theory",
            "parental rights in education",
            "woke indoctrination",
        ],
        "neutral": ["education curriculum", "teaching history", "school policy"],
    },
    "Sex Education and Family Policy": {
        "left": ["comprehensive sex education", "LGBTQ-inclusive curriculum"],
        "right": ["abstinence education", "parental consent", "family values"],
        "neutral": ["sex education policy", "school curriculum", "health education"],
    },
    "Basic Income and Welfare Programs": {
        "left": ["universal basic income", "social safety net", "poverty relief"],
        "right": ["welfare dependency", "personal responsibility", "work requirements"],
        "neutral": ["welfare policy", "income support", "economic assistance"],
    },
    "Trade and Economic Policy": {
        "left": ["fair trade", "labor rights", "climate-friendly trade"],
        "right": ["free trade", "tariffs", "America First trade"],
        "neutral": ["trade policy", "import/export", "trade agreements"],
    },
    "Social Security and Retirement Policy": {
        "left": ["protect social security", "expand benefits"],
        "right": ["entitlement reform", "reduce spending", "privatize social security"],
        "neutral": ["retirement policy", "social security funding", "aging population"],
    },
    "National Security and Civil Liberties": {
        "left": ["surveillance reform", "privacy rights", "anti-war movement"],
        "right": ["national security", "border protection", "patriotism"],
        "neutral": ["counterterrorism", "cybersecurity", "civil liberties"],
    },
}

# max number of posts to retrieve for each topic
MAX_POSTS = 500

DATASET_DIR = "dataset"

REQUEST_DELAY = 2


def sanitize_filename(name):
    name = name.lower()
    name = re.sub(r"\s+", "_", name)  # replace spaces with underscores
    name = re.sub(r"[^\w\-]", "", name)  # remove all non-word characters
    return name


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

    for topic_name, framings in TOPICS.items():
        print(f"\n--- searching for topic: '{topic_name}' ---")

        topic_dir_name = sanitize_filename(topic_name)
        topic_dir = os.path.join(output_dir, topic_dir_name)
        os.makedirs(topic_dir, exist_ok=True)

        for framing, keywords in framings.items():
            for keyword in keywords:
                search_query = f'"{keyword}"'
                print(
                    f"\n--- searching for framing: '{framing}', keyword: '{keyword}', query: {search_query}) ---"
                )
                posts = []
                cursor = None
                try:
                    while len(posts) < MAX_POSTS:
                        print(f"fetching posts... (collected: {len(posts)})")
                        params = models.AppBskyFeedSearchPosts.Params(
                            q=search_query, cursor=cursor, limit=100
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
                        filename = f"{framing}_{sanitize_filename(keyword)}_posts.json"
                        filepath = os.path.join(topic_dir, filename)

                        with open(filepath, "w", encoding="utf-8") as f:
                            json.dump(posts, f, indent=4, ensure_ascii=False)

                        print(f"saved {len(posts)} posts to '{filepath}'")

                except Exception as e:
                    print(f"an error occurred while searching for '{keyword}': {e}")
                    continue  # move to the next topic

    print("\n--- finished ---")


if __name__ == "__main__":
    main()
