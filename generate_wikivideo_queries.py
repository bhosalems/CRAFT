"""Generate persona-augmented queries.jsonl + topic mapping for the WikiVideo
MultiVENT subset, so the existing query branch can run end-to-end on it.

The MAGMaR2026 dev set ships hand-written persona/background per query. WikiVideo
only has a wiki article title, a short event-level query string, claims, and a
gold human-written article. This script uses a local Qwen vLLM model to fabricate
a (persona_title, background, query) for each event in two flavors -- biased and
unbiased -- mirroring the dev set shape, with a couple of dev examples passed in
as few-shot context so the generator stays on-format.

Inputs (read-only):
  - /data_local2/pyan4_shared/wikivideo/annotations/final_data_2015-2025.json
        per-event: claims, original_article, article (gold), videos
  - /data_local2/pyan4_shared/wikivideo/annotations/multivent1_matched_queries_videos.json
        per-event: short query, event_type, video_id list, wiki_title
  - /a2il/data/mbhosale/MAGMaR2026_test/MAGMaR2026_queries_dev.jsonl
        used as few-shot examples

Outputs:
  - data/wikivideo_queries.jsonl                  (queries for run_query.sh)
  - data/topic_video_mapping_wikivideo.json       (mapping for run_query.sh)

The 4 WikiVideo25 events that already exist in the dev set are skipped; only the
~52 MultiVENT2.0 events are emitted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

WIKIVIDEO_ROOT = Path("/data_local2/pyan4_shared/wikivideo")
ANNOT_DIR = WIKIVIDEO_ROOT / "annotations"
FINAL_DATA = ANNOT_DIR / "final_data_2015-2025.json"
MATCHED = ANNOT_DIR / "multivent1_matched_queries_videos.json"
DEV_QUERIES = Path("/a2il/data/mbhosale/MAGMaR2026_test/MAGMaR2026_queries_dev.jsonl")

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_OUT_QUERIES = Path("data/wikivideo_queries.jsonl")
DEFAULT_OUT_MAPPING = Path("data/topic_video_mapping_wikivideo.json")

ARTICLE_CHAR_BUDGET = 1800
CLAIMS_PER_EVENT = 12

WIKIVIDEO25_TITLES = {
    "2025_Myanmar_earthquake",
    "2025_Canadian_federal_election",
    "Blue_Ghost_Mission_1",
    "Liberation_Day_Tariffs",
}


def _norm_title(s: str) -> str:
    s = s.replace("_", " ").replace("-", " ").replace("–", " ").replace("—", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def load_dev_examples() -> list[tuple[dict, dict]]:
    """Pick (unbiased, biased) pairs from the dev jsonl for few-shot."""
    by_title: dict[str, dict[str, dict]] = {}
    for line in DEV_QUERIES.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        by_title.setdefault(rec["title"], {})[rec.get("query_type", "")] = rec
    pairs: list[tuple[dict, dict]] = []
    for title, byt in by_title.items():
        u, b = byt.get("unbiased"), byt.get("biased")
        if u and b:
            pairs.append((u, b))
    return pairs


def collect_events(skip_wikivideo25: bool = True) -> list[dict]:
    """Join final_data + matched on wiki_title; produce per-event records."""
    final_data = json.loads(FINAL_DATA.read_text())
    matched = json.loads(MATCHED.read_text())

    matched_by_norm: dict[str, dict] = {}
    for mid, mrec in matched.items():
        wt = mrec.get("wiki_title", "")
        if wt:
            matched_by_norm[_norm_title(wt)] = {"matched_id": mid, **mrec}

    events: list[dict] = []
    for title, frec in final_data.items():
        if skip_wikivideo25 and title in WIKIVIDEO25_TITLES:
            continue
        m = matched_by_norm.get(_norm_title(title))
        if not m:
            continue

        videos = list(frec.get("videos", {}).keys()) or list(m.get("videos", {}).keys())
        if not videos:
            continue

        # Flatten claims to a single list, sample first N for prompt context.
        flat_claims: list[str] = []
        for group in frec.get("claims", []):
            if isinstance(group, list):
                flat_claims.extend(c for c in group if isinstance(c, str))
            elif isinstance(group, str):
                flat_claims.append(group)

        events.append(
            {
                "title": title,
                "wiki_title_human": m.get("wiki_title", title),
                "wikivideo_query_id": m["matched_id"],
                "short_query": m.get("query") or title.replace("_", " "),
                "event_type": m.get("event_type", ""),
                "videos": videos,
                "article": frec.get("article", "") or "",
                "claims": flat_claims,
            }
        )
    return events


SYSTEM_PROMPT = """\
You generate query records for a video-grounded report-writing benchmark. Each
record describes a fictional analyst/journalist who must produce a written report
about a real-world event using video evidence.

For every event you receive, produce TWO records:
  - one with query_type="unbiased": a neutral information-gathering query,
    typically from an analyst whose job is factual reporting.
  - one with query_type="biased": a query that frames the event from a
    particular angle, point of view, or stakeholder's interest. The bias must
    be in the framing (what aspects matter, what perspective is privileged) --
    NOT in asking for false information. Persona and background should make
    the bias plausible (e.g. an industry advocate, a partisan policy shop, a
    victims' rights group).

The two personas MUST be different people with different jobs/backgrounds, and
their queries should NOT be paraphrases of each other -- they should ask about
different facets of the event.

Output ONLY a single JSON object (no prose, no code fences) of this shape:
{
  "unbiased": {
    "query_type": "unbiased",
    "language": "<language of the event's primary source coverage, e.g. english, arabic, russian, spanish>",
    "persona_title": "<short job title>",
    "background": "<3-5 sentence first-person paragraph about the persona's role, what they typically write, and their audience>",
    "query": "<3-6 sentence first-person request describing what they want to learn from the videos>"
  },
  "biased": { ...same fields, query_type="biased"... }
}
"""


def _truncate_words(s: str, n_chars: int) -> str:
    if len(s) <= n_chars:
        return s
    cut = s[:n_chars]
    sp = cut.rfind(" ")
    if sp > 0:
        cut = cut[:sp]
    return cut + "..."


def _format_example(unbiased: dict, biased: dict) -> str:
    """Render one few-shot example as `INPUT ... -> OUTPUT json`."""
    title = unbiased.get("title", "")
    short = unbiased.get("query") or title
    inp = (
        f"EVENT_TITLE: {title}\n"
        f"SHORT_QUERY: {short}\n"
        f"EVENT_TYPE: (unspecified)\n"
        f"ARTICLE_EXCERPT: (omitted in this example)\n"
        f"SAMPLE_CLAIMS: (omitted in this example)\n"
    )
    out = json.dumps(
        {
            "unbiased": {
                "query_type": "unbiased",
                "language": unbiased.get("language", "english"),
                "persona_title": unbiased.get("persona_title", ""),
                "background": unbiased.get("background", ""),
                "query": unbiased.get("query", ""),
            },
            "biased": {
                "query_type": "biased",
                "language": biased.get("language", "english"),
                "persona_title": biased.get("persona_title", ""),
                "background": biased.get("background", ""),
                "query": biased.get("query", ""),
            },
        },
        ensure_ascii=False,
    )
    return f"INPUT:\n{inp}\nOUTPUT:\n{out}\n"


def build_user_prompt(event: dict, examples: list[tuple[dict, dict]]) -> str:
    parts: list[str] = []
    if examples:
        parts.append("Here are example outputs for similar events:\n")
        for unb, bia in examples:
            parts.append(_format_example(unb, bia))
            parts.append("\n")

    article = _truncate_words(event["article"], ARTICLE_CHAR_BUDGET)
    sample_claims = event["claims"][:CLAIMS_PER_EVENT]

    parts.append("Now generate the JSON object for this event:\n\n")
    parts.append("INPUT:\n")
    parts.append(f"EVENT_TITLE: {event['wiki_title_human']}\n")
    parts.append(f"SHORT_QUERY: {event['short_query']}\n")
    parts.append(f"EVENT_TYPE: {event['event_type']}\n")
    parts.append(f"ARTICLE_EXCERPT: {article}\n")
    parts.append("SAMPLE_CLAIMS:\n")
    for c in sample_claims:
        parts.append(f"  - {c}\n")
    parts.append("\nOUTPUT:\n")
    return "".join(parts)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_record(text: str) -> dict | None:
    """Extract the first JSON object in `text` and return as dict, or None."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def validate_pair(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    for key in ("unbiased", "biased"):
        v = obj.get(key)
        if not isinstance(v, dict):
            return False
        for f in ("query_type", "language", "persona_title", "background", "query"):
            if not isinstance(v.get(f), str) or not v[f].strip():
                return False
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out-queries", type=Path, default=DEFAULT_OUT_QUERIES)
    p.add_argument("--out-mapping", type=Path, default=DEFAULT_OUT_MAPPING)
    p.add_argument("--limit", type=int, default=0, help="If >0, only generate N events (smoke test).")
    p.add_argument("--include-wikivideo25", action="store_true", help="Also include the 4 events already in the dev set.")
    p.add_argument("--max-tokens", type=int, default=1600)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and dump them to stdout; don't load vLLM. Useful to sanity-check the prompt template.",
    )
    args = p.parse_args()

    print(f"[1/5] Loading wikivideo annotations + dev examples", file=sys.stderr)
    events = collect_events(skip_wikivideo25=not args.include_wikivideo25)
    examples = load_dev_examples()
    print(f"  events: {len(events)}; few-shot pairs: {len(examples)}", file=sys.stderr)
    if args.limit > 0:
        events = events[: args.limit]
        print(f"  --limit applied: {len(events)} events", file=sys.stderr)

    print(f"[2/5] Writing topic->video mapping -> {args.out_mapping}", file=sys.stderr)
    args.out_mapping.parent.mkdir(parents=True, exist_ok=True)
    mapping = {ev["wiki_title_human"]: ev["videos"] for ev in events}
    args.out_mapping.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))

    print(f"[3/5] Building prompts", file=sys.stderr)
    prompts = [build_user_prompt(ev, examples) for ev in events]

    if args.dry_run:
        for ev, prompt in zip(events, prompts):
            print(f"\n===== {ev['wikivideo_query_id']} | {ev['wiki_title_human']} =====")
            print(prompt)
        return 0

    print(f"[4/5] Loading vLLM model: {args.model}", file=sys.stderr)
    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, Any] = dict(model=args.model, seed=args.seed)
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.tensor_parallel_size and args.tensor_parallel_size > 1:
        llm_kwargs["tensor_parallel_size"] = int(args.tensor_parallel_size)
    llm = LLM(**llm_kwargs)

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    conversations = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]

    print(f"[5/5] Generating {len(conversations)} (event -> biased+unbiased) records", file=sys.stderr)
    outputs = llm.chat(conversations, sampling_params=sp)
    raw_texts = [out.outputs[0].text for out in outputs]

    # Retry any that failed to parse, once, with temperature 0.
    sp_retry = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed)
    parsed: list[dict | None] = []
    for ev, text in zip(events, raw_texts):
        rec = parse_record(text)
        parsed.append(rec if (rec and validate_pair(rec)) else None)

    failed_idx = [i for i, r in enumerate(parsed) if r is None]
    if failed_idx:
        print(f"[retry] {len(failed_idx)} events failed parse/validate; retrying greedy", file=sys.stderr)
        retry_convs = [conversations[i] for i in failed_idx]
        retry_outs = llm.chat(retry_convs, sampling_params=sp_retry)
        for j, out in zip(failed_idx, retry_outs):
            rec = parse_record(out.outputs[0].text)
            if rec and validate_pair(rec):
                parsed[j] = rec

    out_records: list[dict] = []
    skipped: list[str] = []
    for ev, rec in zip(events, parsed):
        if rec is None:
            skipped.append(ev["wikivideo_query_id"])
            continue
        for variant_key in ("unbiased", "biased"):
            v = rec[variant_key]
            out_records.append(
                {
                    "query_type": v["query_type"],
                    "language": v["language"],
                    "title": ev["wiki_title_human"],
                    "persona_title": v["persona_title"],
                    "background": v["background"],
                    "query": v["query"],
                    "wikivideo_query_id": ev["wikivideo_query_id"],
                    "wiki_title_key": ev["title"],
                }
            )

    # Assign sequential string query_ids matching the dev format.
    for i, rec in enumerate(out_records, start=1):
        rec["query_id"] = str(i)
        # Reorder to put query_id first for readability.
    ordered_keys = ["query_id", "query_type", "language", "title", "persona_title", "background", "query", "wikivideo_query_id", "wiki_title_key"]
    out_records = [{k: r[k] for k in ordered_keys} for r in out_records]

    args.out_queries.parent.mkdir(parents=True, exist_ok=True)
    with args.out_queries.open("w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(out_records)} query records ({len(out_records)//2} events x 2) -> {args.out_queries}")
    print(f"Wrote {len(mapping)} topic->videos entries -> {args.out_mapping}")
    if skipped:
        print(f"WARNING: {len(skipped)} events failed generation and were skipped: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
