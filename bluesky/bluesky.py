import requests, json

BASE = "https://api.bsky.app/xrpc"

def get(name, params):
    url = f"{BASE}/{name}"
    r = requests.get(url, params=params, headers={"Accept": "application/json"})
    print("GET", r.url, ":", r.status_code)
    if r.status_code == 200:
        data = r.json()
        print(
            json.dumps(
                {"keys": list(data.keys())[:5], "sample": str(data)[:160]},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(r.text[:200])

get("app.bsky.feed.searchposts", {"q": "abortion or Roe+v+Wade", "limit": 10})
