import requests

def get_wiki_summary(title, lang="zh"):
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "cookie": "WMF-Uniq=0TrjEJhJ7c_CKvXhkgpukgIXAA4EAFvdgHjxyeOOlNqyINiQQPUD34i-tVH0sj2b; WMF-Last-Access-Global=19-Sep-2025; GeoIP=JP:13:Tokyo:35.69:139.69:v4; enwikiBlockID=24895600%21111c3e1608b2a2dc348f5aa231cd6090ef3778f56fbaab1044f06ccc1f342dafe7c10c7cd6b3e3d9f1aeca269112892c61ea1148e718c7b66598b93035abc5c9; WMF-Last-Access=19-Sep-2025; NetworkProbeLimit=0.001; enwikimwuser-sessionId=9c79cc28edacc483fc8f"
    }
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        return {
            "title": data.get("title"),
            "desc": data.get("description"),
            "summary": data.get("extract"),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
            "image": data.get("thumbnail", {}).get("source")
        }
    else:
        print(f"状态码: {resp.status_code}")
    return None

print(get_wiki_summary("地址"))
