#!/usr/bin/env python3
"""
format_traces.py
────────────────
all_traces/ 이하 모든 *.json 파일을
  · vocab 한 줄
  · requests 키: "request_n"
  · requests 각 항목 한 줄
형식으로 덮어쓴다. *metrics.json 은 건너뜀.
────────────────
사용:  python format_traces.py [--root all_traces]
"""
import json, argparse, pathlib, sys
from collections import OrderedDict

# ── 인라인 JSON 헬퍼 ──────────────────────────────
inline = lambda obj: json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))

def load(path: pathlib.Path):
    return json.loads(path.read_text(), object_pairs_hook=OrderedDict)

def pretty(obj: OrderedDict) -> str:
    out = ["{"]
    for idx, (k, v) in enumerate(obj.items()):
        comma = "," if idx < len(obj) - 1 else ""

        if k == "vocab":
            out.append(f'  "{k}": {inline(v)}{comma}')

        elif k == "requests":
            out.append('  "requests": {')
            req_items = sorted(v.items(), key=lambda kv: int(kv[0]))
            for i, (_, rv) in enumerate(req_items):
                rc = "," if i < len(req_items) - 1 else ""
                out.append(f'    "request_{i}": {inline(rv)}{rc}')
            out.append(f'  }}{comma}')

        else:
            val = f'"{v}"' if isinstance(v, str) else v
            out.append(f'  "{k}": {val}{comma}')
    out.append("}")
    return "\n".join(out) + "\n"

def process_file(path: pathlib.Path):
    try:
        obj = load(path)
    except Exception as e:
        print(f"✗  {path}: JSON 파싱 실패 ({e})", file=sys.stderr)
        return False
    path.write_text(pretty(obj), encoding="utf-8")
    return True

def main(root: str):
    root = pathlib.Path(root)
    if not root.is_dir():
        sys.exit(f"{root} 디렉터리가 없습니다.")

    json_files = [p for p in root.rglob("*.json")
                  if not p.name.endswith("metrics.json")]

    ok, fail = 0, 0
    for f in json_files:
        if process_file(f):
            ok += 1
        else:
            fail += 1

    print(f"\n완료: {ok}개 성공, {fail}개 실패")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="all_traces",
                        help="탐색을 시작할 최상위 디렉터리 (기본: all_traces)")
    args = parser.parse_args()
    main(args.root)
