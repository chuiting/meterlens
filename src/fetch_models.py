"""Download any CNN weight file missing from models/ (URLs in config.yaml)."""

import os
import re
import shutil
import sys
import urllib.error
import urllib.request

from config import CONFIG, resolve_path

HF_URL = re.compile(r"https://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[^/]+)/(?P<file>.+)$")


def token():
    t = (os.environ.get("HF_TOKEN") or "").strip()
    print(f"[fetch_models] HF_TOKEN {'set (' + t[:3] + '…, ' + str(len(t)) + ' chars)' if t else 'not set'}", flush=True)
    bad = [i for i, c in enumerate(t) if not c.isascii() or c.isspace()]
    if bad:
        raise SystemExit(f"[fetch_models] HF_TOKEN has a stray character at position {bad[0] + 1} -- re-paste the token")
    return t or None


def download(url: str, dest, tok):
    m = HF_URL.match(url)
    if m:
        # Write straight into models/, skipping the hub cache (saves RAM on Cloud Run).
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(m["repo"], m["file"], revision=m["rev"], token=tok, local_dir=dest.parent)
        shutil.move(path, dest)
        return
    req = urllib.request.Request(url)
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def fetch_missing() -> list:
    missing = [(n, u) for n, u in (CONFIG.get("model_downloads") or {}).items() if not resolve_path(f"models/{n}").exists()]
    if not missing:
        return []
    tok = token()
    fetched = []
    for name, url in missing:
        path = resolve_path(f"models/{name}")
        if not url or "<" in url:
            print(f"[fetch_models] {name} is missing and no download URL is configured", file=sys.stderr)
            continue
        print(f"[fetch_models] downloading {name} from {url}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            download(url, tmp, tok)
        except Exception as e:
            code = getattr(e, "code", None) or getattr(getattr(e, "response", None), "status_code", None)
            hint = {401: "not authenticated -- is HF_TOKEN set for this process?",
                    403: "HF_TOKEN cannot read this repo (wrong account, or a fine-grained token without private-repo read)",
                    404: "no such file or repo"}.get(code, str(e))
            raise SystemExit(f"[fetch_models] {name}: HTTP {code} {hint}")
        tmp.rename(path)
        fetched.append(name)
    return fetched


if __name__ == "__main__":
    fetch_missing()
