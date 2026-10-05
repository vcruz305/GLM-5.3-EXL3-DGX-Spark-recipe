#!/usr/bin/env python3
"""Write the served chat template: GLM-5.3's official template with one line changed so thinking off works.

The official template ends every prompt with `<|assistant|><think>` and ignores `enable_thinking`, so a client that
asks for thinking off still gets a reasoning reply. The fix is GLM-5 / GLM-5.1's thinking-off form on that one line:
`</think>` when enable_thinking is false, `<think>` otherwise. With thinking on (the default) the rendered prompt ids
are the official template's. The pack is never edited: the result goes into the serve view.

  python3 fix_chat_template.py OFFICIAL_TEMPLATE OUT_FILE
Refuses unless the input is the pinned official template and the output has the pinned fixed sha256.
"""
import hashlib
import sys

OFFICIAL_SHA = "3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c"
FIXED_SHA = "2059ad4b073838cebd243d09b6633833e633edd866b95c549be25483909b00d0"
OLD = b"    <|assistant|>{{- '<think>' -}}\n"
NEW = b"    <|assistant|>{{- '</think>' if (enable_thinking is defined and not enable_thinking) else '<think>' -}}\n"


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    data = open(src, "rb").read()
    got = hashlib.sha256(data).hexdigest()
    if got != OFFICIAL_SHA:
        print(f"{src}: sha256 {got} is not the official GLM-5.3 template {OFFICIAL_SHA}", file=sys.stderr)
        return 1
    if data.count(OLD) != 1:
        print(f"{src}: the generation-prompt line occurs {data.count(OLD)} times, expected 1", file=sys.stderr)
        return 1
    out = data.replace(OLD, NEW)
    if hashlib.sha256(out).hexdigest() != FIXED_SHA:
        print("fixed template sha256 mismatch; refusing to write", file=sys.stderr)
        return 1
    with open(dst, "wb") as f:
        f.write(out)
    print(f"{dst}: fixed template, sha256 {FIXED_SHA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
