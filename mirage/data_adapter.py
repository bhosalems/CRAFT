import json
import os
from typing import Any, Dict, List


class MagmarJSONLAdapter:
    """
    Converts MAGMAR-style JSONL (one query per line) into the dict format
    expected by MIRAGE's infof1.py and citef1.py main() routines.

    Input record shape (per line of the .jsonl file):
        {
          "metadata":   {"query_id": str | int, ...},
          "references": ["video_id", ...],                       # used for the reference file
          "responses":  [
              {"text": "atomic claim string", "citations": ["video_id", ...]},
              ...
          ]
        }

    Each response's "text" is treated as a single sentence containing a single
    subclaim (so sentences[i] == text, claims[i] == [text]). Citations are
    converted to "{video_dir}/{video_id}{video_ext}" so they byte-match the
    paths citef1.py builds from the reference's supporting_videos list.
    """

    def __init__(self, video_dir: str, video_ext: str = ".mp4"):
        self.video_dir = video_dir
        self.video_ext = video_ext

    def _cite_path(self, video_id: str) -> str:
        return os.path.join(self.video_dir, f"{video_id}{self.video_ext}")

    @staticmethod
    def load_records(path: str) -> List[Dict[str, Any]]:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def to_prediction_dict(self, path: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for rec in self.load_records(path):
            tid = str(rec["metadata"]["query_id"])
            responses = rec.get("responses", [])
            texts = [r["text"] for r in responses]
            out[tid] = {
                "prediction": " ".join(texts),
                "sentences": texts,
                "claims": [[t] for t in texts],
                "citations": [
                    [self._cite_path(c) for c in r.get("citations", [])]
                    for r in responses
                ],
            }
        return out

    def to_reference_dict(self, path: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for rec in self.load_records(path):
            tid = str(rec["metadata"]["query_id"])
            responses = rec.get("responses", [])
            claims_to_videos: Dict[str, Dict[str, Any]] = {}
            for r in responses:
                citations = r.get("citations", [])
                if not citations:
                    continue
                claims_to_videos[r["text"]] = {
                    "supporting_videos": citations,
                    "videos_modalities": {c: {} for c in citations},
                }
            out[tid] = {
                "article": " ".join(r["text"] for r in responses),
                "videos": rec.get("references", []),
                "claims_to_supporting_videos": claims_to_videos,
            }
        return out


def _looks_like_jsonl(path: str) -> bool:
    """Detect newline-delimited JSON regardless of file extension.

    Returns True if the first non-blank line parses as JSON AND at least one
    more non-blank line exists. This catches files like 'submission_baseline'
    (no extension) that are JSONL in disguise.
    """
    with open(path) as f:
        first = ""
        for line in f:
            if line.strip():
                first = line.strip()
                break
        if not first:
            return False
        try:
            json.loads(first)
        except json.JSONDecodeError:
            return False
        for line in f:
            if line.strip():
                return True
    return False


def load_prediction(path: str, video_dir: str) -> Dict[str, Any]:
    """Load a predictions file, auto-detecting JSONL (MAGMAR) vs JSON (MIRAGE)."""
    if path.endswith(".jsonl") or _looks_like_jsonl(path):
        return MagmarJSONLAdapter(video_dir).to_prediction_dict(path)
    with open(path) as f:
        return json.load(f)


def load_reference(path: str, video_dir: str) -> Dict[str, Any]:
    """Load a reference file, auto-detecting JSONL (MAGMAR) vs JSON (MIRAGE)."""
    if path.endswith(".jsonl") or _looks_like_jsonl(path):
        return MagmarJSONLAdapter(video_dir).to_reference_dict(path)
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Dry-run the MAGMAR→MIRAGE JSONL adapter and dump the converted JSON."
    )
    ap.add_argument("--input", required=True, help="Path to input .jsonl")
    ap.add_argument(
        "--mode", choices=["prediction", "reference"], required=True
    )
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--output", required=True, help="Where to write converted .json")
    args = ap.parse_args()

    adapter = MagmarJSONLAdapter(args.video_dir)
    data = (
        adapter.to_prediction_dict(args.input)
        if args.mode == "prediction"
        else adapter.to_reference_dict(args.input)
    )
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {args.output} with {len(data)} topics")
