"""Build the long-prompt test inputs from real Project Gutenberg books (public domain in the US; not model-generated).

usage: python3 bench/make_prompts.py MODEL_DIR [OUT_JSON]
  MODEL_DIR: any directory with the pack's tokenizer.json (the serve view or the pack itself)
  OUT_JSON:  default bench/longctx_prompts.json (git-ignored)

Downloads the two books into bench/books/ (git-ignored) and checks their sha256 against the files the 2026-10-05 run
used, strips the Gutenberg header/footer, trims the body to a target token count with the model's own tokenizer, and
wraps it in one user message with a fixed question. The served prompt length is re-checked with /tokenize at run time
(bench/longctx_run.py). Needs `tokenizers` (the recipe venv has it).
"""
import hashlib
import json
import os
import sys
import urllib.request

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
BOOKS = {   # file: (url, sha256 of the file the measured run used; Gutenberg's copy matched it on 2026-10-05)
    "moby_dick_pg2701.txt": ("https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
                             "907420db6c4b68c70e2988cd2ad9c8cf79138667a01b63376d18dd17fef1a18b"),
    "war_and_peace_pg2600.txt": ("https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
                                 "2d5bb2ad5f422765e714617e21fa31bbaf8958aa79682c86fca6660fcc5d1b2b"),
}
QUESTION = ("\n\n---\nThe text above is the opening of a novel. Write a detailed, chapter-by-chapter summary of what "
            "happens in it, naming the characters involved.")
SPECS = [("book_128k", "moby_dick_pg2701.txt", "Moby-Dick; or, The Whale (Herman Melville), Project Gutenberg #2701",
          127_500),
         ("book_200k", "war_and_peace_pg2600.txt", "War and Peace (Leo Tolstoy, tr. Maude), Project Gutenberg #2600",
          199_500),
         ("book_162k", "war_and_peace_pg2600.txt", "War and Peace (Leo Tolstoy, tr. Maude), Project Gutenberg #2600",
          162_500)]          # the largest that fits --context 163840 with 512 new tokens


def fetch(fn: str) -> str:
    url, sha = BOOKS[fn]
    path = os.path.join(HERE, "books", fn)
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(url, timeout=300) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != sha:
        print(f"warning: {path} sha256 {got} != the measured file's {sha} (Gutenberg updated the text?): prompts "
              f"will differ from the 2026-10-05 run", file=sys.stderr)
    return path


def body(path):
    t = open(path, encoding="utf-8-sig").read().replace("\r\n", "\n")
    a = t.find("*** START OF")
    a = t.find("\n", a) + 1 if a >= 0 else 0
    e = t.find("*** END OF")
    return t[a:e if e > 0 else len(t)].strip()


def main() -> int:
    meta = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "longctx_prompts.json")
    tok = Tokenizer.from_file(os.path.join(meta, "tokenizer.json"))
    res = []
    for name, fn, src, target in SPECS:
        text = body(fetch(fn))
        enc = tok.encode(text, add_special_tokens=False)
        if len(enc) < target:
            raise SystemExit(f"{fn}: only {len(enc)} tokens < {target}")
        cut = enc.offsets[target - 1][1]
        doc = text[:cut]
        content = doc + QUESTION
        n = len(tok.encode(content, add_special_tokens=False).ids)
        res.append({"name": name, "source": src, "file": fn, "doc_chars": len(doc), "doc_tokens_target": target,
                    "content_tokens_local": n, "messages": [{"role": "user", "content": content}]})
        print(name, src, "chars", len(doc), "content tokens (local tokenizer, no template)", n, "of", len(enc), "in book")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"prompts": res}, f)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
