import tarfile
import orjson
import random

import pyarrow as pa
import pyarrow.parquet as pq

BLUESKY_TAR_PATH = "user_posts.tar.gz"
TRUTH_TSV_PATH = "truth_posts.tsv"

SAMPLE_SIZE = 80_000
OUT_PARQUET_BLUESKY = "bluesky_sample.parquet"
OUT_PARQUET_TRUTH = "truth_sample.parquet"

random.seed(0)


def iter_bluesky(tar_path):
    """
    Yields the json object of every post in user_posts.tar.gz.
    """
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                yield obj


def reservoir_sample_from_tar(tar_path, k):
    """
    One-pass reservoir sampling over *posts* (not users).
    Returns a list of k rows and the total number of posts seen.
    """
    reservoir = []
    n = 0  # number of accepted posts seen so far

    for post in iter_bluesky(tar_path):
        text = post.get("text") or ""
        langs = post.get("langs") or []
        reply_to = post.get("reply_to")

        # skip empty texts, non-English, replies
        if not text or "eng" not in langs or reply_to:
            continue

        n += 1
        if n <= k:
            reservoir.append(post)
        else:
            # choose integer in [1, n]; if <= k, replace that slot
            j = random.randint(1, n)
            if j <= k:
                reservoir[j - 1] = post

        # progress logging
        if n % 100_000 == 0:
            print(f"Seen {n:,} posts…", flush=True)

    print(f"Total posts included in sampling process: {n:,}")
    return reservoir, n


def main():
    sample, total = reservoir_sample_from_tar(BLUESKY_TAR_PATH, SAMPLE_SIZE)
    print(
        f"Final sample size: {len(sample):,} (~{len(sample)/max(total,1):.4%} of posts)"
    )

    table = pa.Table.from_pylist(sample)
    pq.write_table(table, OUT_PARQUET_BLUESKY)
    print(f"Wrote Parquet sample to {OUT_PARQUET_BLUESKY}")


if __name__ == "__main__":
    main()
