# Benchmark viewer

A static page over [`benchmark-data.json`](benchmark-data.json): the decode-per-prompt bars, the long-prompt table, the
profile comparison and the speed sweep, each labelled with the harness that produced it. It displays saved results and
runs no inference; every number in the JSON is in the root README and traceable to
[`bench/records/`](../bench/records/README.md).

## View

Serve the folder over HTTP (the page fetches the JSON), or publish `docs/` with GitHub Pages once the repo is public:

```sh
python3 -m http.server -d docs 8000      # then http://127.0.0.1:8000/
```

## Render a PNG locally

```sh
python -m pip install playwright
python -m playwright install chromium
python docs/render_benchmark.py          # benchmark-renders/benchmark.png; fails on JS errors or outside requests
```

## Measurement boundaries

- Decode rates are the engine's own (tokens after the first over decode time), greedy, 512 new tokens.
- The 6-prompt and long-prompt rows ran through the `/v1` server; the sweep rows ran in-process (no HTTP).
- SixCat decode is a synthetic, highly draftable workload: a ceiling, not the speed a user sees.
- `int4-262k` is pending validation and has no figures.
