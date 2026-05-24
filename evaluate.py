#!/usr/bin/env python3
"""
PR9: Evaluation harness and comparison report.

Runs structural validation, computes extraction-level and report-level metrics
across note-taking and query-based pipelines, and produces aggregate comparison
outputs broken out by system, topic, query type, and language group.

Supports both legacy (observation/grounded) and new (general-note/query-claim) schemas.

Usage:
    # Legacy evaluation
    python note_taking/evaluate.py \
        --observations note_taking/outputs/observations_llm/observation_notes.jsonl \
        --grounded note_taking/outputs/grounded_llm/grounded_notes.jsonl \
        --reports-note-taking note_taking/outputs/reports_note_taking \
        --reports-single-query note_taking/outputs/reports_single_query \
        --reports-expanded-query note_taking/outputs/reports_expanded_query \
        --out-dir note_taking/outputs/evaluation

    # New pipeline evaluation
    python note_taking/evaluate.py \
        --general-notes note_taking/outputs/general_notes/general_notes.jsonl \
        --query-claims note_taking/outputs/query_claims_single/query_conditioned_claims.jsonl \
        --note-packets note_taking/outputs/note_packets \
        --claim-packets note_taking/outputs/claim_packets \
        --inferences-note note_taking/outputs/inferences_note \
        --inferences-query note_taking/outputs/inferences_query \
        --reports-note-based note_taking/outputs/reports_note_based \
        --reports-query-based note_taking/outputs/reports_query_based \
        --out-dir note_taking/outputs/evaluation
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set
from tqdm import tqdm

from contracts import (
    DEFAULT_EXPANDED_QUERIES,
    DEFAULT_QUERIES_JSONL,
    DEFAULT_TOPIC_MAPPING,
    build_query_topic_map,
    load_expanded_queries,
    load_queries,
    load_topic_mapping,
    validate_claim_packet,
    validate_fact,
    validate_general_note,
    validate_grounded_note,
    validate_higher_level_inference,
    validate_note_packet,
    validate_observation_note,
    validate_query_conditioned_claim,
    validate_query_packet,
    validate_report_citation,
)
from run_metadata import build_run_manifest, write_run_manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_iter_jsonl(path: str) -> list:
    """Load JSONL if path exists, else return empty list."""
    if not path or not os.path.exists(path):
        return []
    return list(_iter_jsonl(path))


def _load_reports(reports_dir: str) -> List[dict]:
    if not os.path.isdir(reports_dir):
        return []
    combined = os.path.join(reports_dir, "all_reports.json")
    if os.path.exists(combined):
        with open(combined, "r") as f:
            return json.load(f)
    return []


def _load_packets(packets_dir: str) -> List[dict]:
    if not os.path.isdir(packets_dir):
        return []
    combined = os.path.join(packets_dir, "all_packets.json")
    if os.path.exists(combined):
        with open(combined, "r") as f:
            return json.load(f)
    return []


def _tokenize(text: str) -> Set[str]:
    return set(text.lower().split())


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


def validate_structure(
    queries: List[dict],
    topic_map: Dict[str, List[str]],
    observations: List[dict],
    grounded: List[dict],
    reports_by_pipeline: Dict[str, List[dict]],
    expanded: Dict[str, dict],
    verbose: bool = False,
    packets_by_pipeline: Optional[Dict[str, List[dict]]] = None,
    general_notes: Optional[List[dict]] = None,
    query_claims: Optional[List[dict]] = None,
    note_packets: Optional[List[dict]] = None,
    claim_packets: Optional[List[dict]] = None,
    inferences_note: Optional[List[dict]] = None,
    inferences_query: Optional[List[dict]] = None,
) -> List[str]:
    """Run all automatic structural checks."""
    errors = []

    # 1. Every query maps to exactly one topic
    try:
        qtm = build_query_topic_map(queries, list(topic_map.keys()))
        mapped_ids = set()
        for topic, qs in qtm.items():
            for q in qs:
                qid = q["query_id"]
                if qid in mapped_ids:
                    errors.append(f"query {qid} mapped to multiple topics")
                mapped_ids.add(qid)
        all_ids = {q["query_id"] for q in queries}
        unmapped = all_ids - mapped_ids
        if unmapped:
            errors.append(f"queries not mapped to any topic: {sorted(unmapped)}")
    except ValueError as e:
        errors.append(f"query-topic mapping failed: {e}")

    # 2. Legacy: every grounded note links to valid observation IDs from the same video
    obs_idx = {}
    for obs in observations:
        obs_idx[obs.get("note_id", "")] = obs

    for gn in grounded:
        gn_vid = gn.get("video_id", "")
        for obs_id in gn.get("source_observation_ids", []):
            if obs_id not in obs_idx:
                errors.append(
                    f"grounded note {gn['note_id']}: references missing observation {obs_id}"
                )
            elif obs_idx[obs_id].get("video_id", "") != gn_vid:
                errors.append(
                    f"grounded note {gn['note_id']}: observation {obs_id} is from "
                    f"different video ({obs_idx[obs_id].get('video_id')}) than note ({gn_vid})"
                )

    # 3. Citation resolution across all report pipelines
    for pipeline, reports in reports_by_pipeline.items():
        for report in reports:
            for i, cit in enumerate(report.get("citations", [])):
                cit_errors = validate_report_citation(cit)
                if cit_errors:
                    errors.extend(
                        [f"{pipeline}/query_{report['query_id']}/cit[{i}]: {e}" for e in cit_errors]
                    )
                if cit.get("video_id") == "unknown":
                    errors.append(
                        f"{pipeline}/query_{report['query_id']}/cit[{i}]: unresolved video_id"
                    )

    # 4. Expanded queries map to official IDs
    if expanded:
        official_ids = {q["query_id"] for q in queries}
        for qid in expanded:
            if qid not in official_ids:
                errors.append(f"expanded query ID '{qid}' not in official queries")

    # 5. Legacy schema validation
    for i, obs in enumerate(observations):
        obs_errs = validate_observation_note(obs)
        if obs_errs:
            errors.extend([f"observation[{i}]: {e}" for e in obs_errs])

    for i, gn in enumerate(grounded):
        gn_errs = validate_grounded_note(gn)
        if gn_errs:
            errors.extend([f"grounded[{i}]: {e}" for e in gn_errs])

    # 6. Legacy fact validation on query-based packets
    if packets_by_pipeline:
        for pipeline, packets in packets_by_pipeline.items():
            for packet in packets:
                for j, fact in enumerate(packet.get("facts", [])):
                    fact_errs = validate_fact(fact)
                    if fact_errs:
                        errors.extend(
                            [f"{pipeline}/query_{packet.get('query_id', '?')}/fact[{j}]: {e}"
                             for e in fact_errs]
                        )

    # 7. New schema: general notes
    if general_notes:
        for i, note in enumerate(general_notes):
            errs = validate_general_note(note)
            if errs:
                errors.extend([f"general_note[{i}]: {e}" for e in errs])
            # is_post_grounded must be false
            if note.get("is_post_grounded") is True:
                errors.append(f"general_note[{i}]: is_post_grounded should be false")

    # 8. New schema: query-conditioned claims
    if query_claims:
        for i, claim in enumerate(query_claims):
            errs = validate_query_conditioned_claim(claim)
            if errs:
                errors.extend([f"query_claim[{i}]: {e}" for e in errs])
            if claim.get("is_post_grounded") is True:
                errors.append(f"query_claim[{i}]: is_post_grounded should be false")

    # 9. New schema: note packets
    if note_packets:
        for i, packet in enumerate(note_packets):
            errs = validate_note_packet(packet)
            if errs:
                errors.extend([f"note_packet[{i}]: {e}" for e in errs])

    # 10. New schema: claim packets
    if claim_packets:
        for i, packet in enumerate(claim_packets):
            errs = validate_claim_packet(packet)
            if errs:
                errors.extend([f"claim_packet[{i}]: {e}" for e in errs])

    # 11. New schema: inferences (note stream)
    if inferences_note:
        note_packet_ids = {
            packet.get("query_id", ""): set(packet.get("note_ids", []))
            for packet in (note_packets or [])
        }
        note_ids = {note.get("note_id", "") for note in (general_notes or [])}
        for i, inf in enumerate(inferences_note):
            errs = validate_higher_level_inference(inf)
            if errs:
                errors.extend([f"inference_note[{i}]: {e}" for e in errs])
            if inf.get("is_post_grounded") is not True:
                errors.append(f"inference_note[{i}]: is_post_grounded should be true")
            qid = inf.get("query_id", "")
            allowed = note_packet_ids.get(qid, set())
            for sid in inf.get("source_ids", []):
                if sid not in allowed:
                    errors.append(
                        f"inference_note[{i}]: source_id {sid} not present in note packet for query {qid}"
                    )
                if sid not in note_ids:
                    errors.append(
                        f"inference_note[{i}]: source_id {sid} not found in general notes"
                    )

    # 12. New schema: inferences (query stream)
    if inferences_query:
        claim_packet_ids = {
            packet.get("query_id", ""): set(packet.get("claim_ids", []))
            for packet in (claim_packets or [])
        }
        claim_ids = {claim.get("claim_id", "") for claim in (query_claims or [])}
        for i, inf in enumerate(inferences_query):
            errs = validate_higher_level_inference(inf)
            if errs:
                errors.extend([f"inference_query[{i}]: {e}" for e in errs])
            if inf.get("is_post_grounded") is not True:
                errors.append(f"inference_query[{i}]: is_post_grounded should be true")
            qid = inf.get("query_id", "")
            allowed = claim_packet_ids.get(qid, set())
            for sid in inf.get("source_ids", []):
                if sid not in allowed:
                    errors.append(
                        f"inference_query[{i}]: source_id {sid} not present in claim packet for query {qid}"
                    )
                if sid not in claim_ids:
                    errors.append(
                        f"inference_query[{i}]: source_id {sid} not found in query-conditioned claims"
                    )

    if verbose:
        print(f"  Structural checks: {len(errors)} error(s)")

    return errors


# ---------------------------------------------------------------------------
# Extraction-level metrics
# ---------------------------------------------------------------------------


def compute_extraction_metrics(
    observations: List[dict],
    grounded: List[dict],
    topic_map: Dict[str, List[str]],
    general_notes: Optional[List[dict]] = None,
    query_claims: Optional[List[dict]] = None,
) -> dict:
    """Compute extraction-level quality metrics."""
    # Group by topic and video
    obs_by_video = defaultdict(list)
    for obs in observations:
        obs_by_video[obs.get("video_id", "")].append(obs)

    gn_by_video = defaultdict(list)
    gn_by_topic = defaultdict(list)
    for gn in grounded:
        gn_by_video[gn.get("video_id", "")].append(gn)
        gn_by_topic[gn.get("topic", "")].append(gn)

    # Note density: notes per video
    videos_with_notes = [v for v, notes in gn_by_video.items() if notes]
    note_counts = [len(gn_by_video[v]) for v in videos_with_notes]
    avg_density = sum(note_counts) / len(note_counts) if note_counts else 0.0

    # Groundedness: fraction of grounded notes with valid source_observation_ids
    obs_id_set = {obs.get("note_id", "") for obs in observations}
    n_grounded_valid = 0
    for gn in grounded:
        src_ids = gn.get("source_observation_ids", [])
        if src_ids and all(sid in obs_id_set for sid in src_ids):
            n_grounded_valid += 1
    groundedness = n_grounded_valid / len(grounded) if grounded else 0.0

    # Redundancy: within each video, token-set Jaccard between pairs of notes
    redundancy_scores = []
    for vid, notes in gn_by_video.items():
        if len(notes) < 2:
            continue
        claims = [_tokenize(n.get("claim", "")) for n in notes]
        pair_jaccards = []
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                inter = len(claims[i] & claims[j])
                union = len(claims[i] | claims[j])
                if union > 0:
                    pair_jaccards.append(inter / union)
        if pair_jaccards:
            redundancy_scores.append(sum(pair_jaccards) / len(pair_jaccards))

    avg_redundancy = sum(redundancy_scores) / len(redundancy_scores) if redundancy_scores else 0.0

    # Per-topic note density
    topic_density = {}
    for topic, vids in topic_map.items():
        topic_notes = gn_by_topic.get(topic, [])
        vids_with = len(set(gn.get("video_id") for gn in topic_notes))
        topic_density[topic] = {
            "total_notes": len(topic_notes),
            "videos_with_notes": vids_with,
            "total_videos": len(vids),
            "avg_notes_per_video": len(topic_notes) / vids_with if vids_with else 0.0,
        }

    metrics = {
        "total_observations": len(observations),
        "total_grounded_notes": len(grounded),
        "videos_with_notes": len(videos_with_notes),
        "avg_notes_per_video": round(avg_density, 2),
        "groundedness": round(groundedness, 4),
        "avg_redundancy": round(avg_redundancy, 4),
        "per_topic": topic_density,
    }

    # New pipeline metrics
    if general_notes:
        gn_by_vid = defaultdict(list)
        for n in general_notes:
            gn_by_vid[n.get("video_id", "")].append(n)
        metrics["total_general_notes"] = len(general_notes)
        metrics["general_notes_videos"] = len(gn_by_vid)
        gn_counts = [len(v) for v in gn_by_vid.values()]
        metrics["general_notes_avg_per_video"] = round(
            sum(gn_counts) / len(gn_counts), 2
        ) if gn_counts else 0.0

    if query_claims:
        metrics["total_query_claims"] = len(query_claims)
        qc_by_query = defaultdict(list)
        for c in query_claims:
            qc_by_query[c.get("query_id", "")].append(c)
        metrics["query_claims_queries"] = len(qc_by_query)

    return metrics


# ---------------------------------------------------------------------------
# Report-level metrics
# ---------------------------------------------------------------------------


def compute_report_metrics(
    reports: List[dict],
    pipeline: str,
    queries: List[dict],
) -> dict:
    """Compute report-level quality metrics for a single pipeline."""
    query_by_id = {q["query_id"]: q for q in queries}

    per_query = {}
    total_sections = 0
    total_citations = 0
    total_valid_citations = 0
    total_unique_videos = set()

    for report in reports:
        qid = report.get("query_id", "")
        query = query_by_id.get(qid, {})
        sections = report.get("sections", [])
        citations = report.get("citations", [])

        total_sections += len(sections)
        total_citations += len(citations)

        valid_cits = [c for c in citations if c.get("video_id", "unknown") != "unknown"]
        total_valid_citations += len(valid_cits)

        cited_videos = set(c.get("video_id") for c in citations if c.get("video_id") != "unknown")
        total_unique_videos.update(cited_videos)

        query_tokens = _tokenize(query.get("query", ""))
        relevance_scores = []
        for sec in sections:
            sec_tokens = _tokenize(sec.get("text", ""))
            if query_tokens and sec_tokens:
                inter = len(query_tokens & sec_tokens)
                union = len(query_tokens | sec_tokens)
                relevance_scores.append(inter / union if union else 0.0)

        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

        section_tokens = [_tokenize(s.get("text", "")) for s in sections]
        pair_jaccards = []
        for i in range(len(section_tokens)):
            for j in range(i + 1, len(section_tokens)):
                inter = len(section_tokens[i] & section_tokens[j])
                union = len(section_tokens[i] | section_tokens[j])
                if union > 0:
                    pair_jaccards.append(inter / union)
        report_redundancy = sum(pair_jaccards) / len(pair_jaccards) if pair_jaccards else 0.0

        per_query[str(qid)] = {
            "n_sections": len(sections),
            "n_citations": len(citations),
            "n_valid_citations": len(valid_cits),
            "n_cited_videos": len(cited_videos),
            "avg_relevance": round(avg_relevance, 4),
            "redundancy": round(report_redundancy, 4),
            "query_type": query.get("query_type", ""),
            "language": query.get("language", ""),
            "topic": report.get("topic", ""),
        }

    citation_validity = total_valid_citations / total_citations if total_citations else 0.0

    return {
        "pipeline": pipeline,
        "total_reports": len(reports),
        "total_sections": total_sections,
        "total_citations": total_citations,
        "citation_validity": round(citation_validity, 4),
        "unique_videos_cited": len(total_unique_videos),
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Cross-pipeline comparison
# ---------------------------------------------------------------------------


def compute_comparison(
    metrics_by_pipeline: Dict[str, dict],
    queries: List[dict],
) -> dict:
    """Compare pipelines along shared dimensions."""
    by_type = defaultdict(list)
    by_lang = defaultdict(list)
    for q in queries:
        by_type[q.get("query_type", "unknown")].append(q["query_id"])
        by_lang[q.get("language", "unknown")].append(q["query_id"])

    comparison = {"pipelines": {}, "by_query_type": {}, "by_language": {}, "by_topic": {}}

    for pipeline, metrics in metrics_by_pipeline.items():
        pq = metrics.get("per_query", {})
        relevance_vals = [v["avg_relevance"] for v in pq.values()]
        redundancy_vals = [v["redundancy"] for v in pq.values()]

        comparison["pipelines"][pipeline] = {
            "total_sections": metrics["total_sections"],
            "total_citations": metrics["total_citations"],
            "citation_validity": metrics["citation_validity"],
            "avg_relevance": round(sum(relevance_vals) / len(relevance_vals), 4) if relevance_vals else 0.0,
            "avg_redundancy": round(sum(redundancy_vals) / len(redundancy_vals), 4) if redundancy_vals else 0.0,
            "avg_sections_per_query": round(metrics["total_sections"] / len(pq), 2) if pq else 0.0,
        }

    for qtype, qids in by_type.items():
        qid_set = set(str(q) for q in qids)
        comparison["by_query_type"][qtype] = {}
        for pipeline, metrics in metrics_by_pipeline.items():
            pq = metrics.get("per_query", {})
            relevant = {k: v for k, v in pq.items() if k in qid_set}
            if relevant:
                secs = sum(v["n_sections"] for v in relevant.values())
                rels = [v["avg_relevance"] for v in relevant.values()]
                comparison["by_query_type"][qtype][pipeline] = {
                    "n_queries": len(relevant),
                    "total_sections": secs,
                    "avg_relevance": round(sum(rels) / len(rels), 4) if rels else 0.0,
                }

    for lang, qids in by_lang.items():
        qid_set = set(str(q) for q in qids)
        comparison["by_language"][lang] = {}
        for pipeline, metrics in metrics_by_pipeline.items():
            pq = metrics.get("per_query", {})
            relevant = {k: v for k, v in pq.items() if k in qid_set}
            if relevant:
                secs = sum(v["n_sections"] for v in relevant.values())
                rels = [v["avg_relevance"] for v in relevant.values()]
                comparison["by_language"][lang][pipeline] = {
                    "n_queries": len(relevant),
                    "total_sections": secs,
                    "avg_relevance": round(sum(rels) / len(rels), 4) if rels else 0.0,
                }

    topic_set = set()
    for pipeline, metrics in metrics_by_pipeline.items():
        for qid, qm in metrics.get("per_query", {}).items():
            topic_set.add(qm.get("topic", ""))

    for topic in sorted(topic_set):
        if not topic:
            continue
        comparison["by_topic"][topic] = {}
        for pipeline, metrics in metrics_by_pipeline.items():
            pq = metrics.get("per_query", {})
            relevant = {k: v for k, v in pq.items() if v.get("topic") == topic}
            if relevant:
                secs = sum(v["n_sections"] for v in relevant.values())
                rels = [v["avg_relevance"] for v in relevant.values()]
                comparison["by_topic"][topic][pipeline] = {
                    "n_queries": len(relevant),
                    "total_sections": secs,
                    "avg_relevance": round(sum(rels) / len(rels), 4) if rels else 0.0,
                }

    return comparison


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_comparison_md(
    structural_errors: List[str],
    extraction_metrics: dict,
    report_metrics: Dict[str, dict],
    comparison: dict,
) -> str:
    """Render evaluation results as markdown."""
    lines = []
    lines.append("# Evaluation Report")
    lines.append("")

    # Structural validation
    lines.append("## Structural Validation")
    lines.append("")
    if structural_errors:
        lines.append(f"**{len(structural_errors)} error(s) found:**")
        lines.append("")
        for e in structural_errors[:20]:
            lines.append(f"- {e}")
    else:
        lines.append("All structural checks passed.")
    lines.append("")

    # Extraction metrics
    lines.append("## Extraction-Level Metrics")
    lines.append("")
    em = extraction_metrics
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total observations | {em['total_observations']} |")
    lines.append(f"| Total grounded notes | {em['total_grounded_notes']} |")
    lines.append(f"| Videos with notes | {em['videos_with_notes']} |")
    lines.append(f"| Avg notes/video | {em['avg_notes_per_video']} |")
    lines.append(f"| Groundedness | {em['groundedness']} |")
    lines.append(f"| Avg redundancy | {em['avg_redundancy']} |")
    if "total_general_notes" in em:
        lines.append(f"| Total general notes | {em['total_general_notes']} |")
        lines.append(f"| General notes videos | {em['general_notes_videos']} |")
        lines.append(f"| General notes avg/video | {em['general_notes_avg_per_video']} |")
    if "total_query_claims" in em:
        lines.append(f"| Total query claims | {em['total_query_claims']} |")
        lines.append(f"| Query claims queries | {em['query_claims_queries']} |")
    lines.append("")

    # Per-topic density
    lines.append("### Per-Topic Note Density")
    lines.append("")
    lines.append("| Topic | Notes | Videos w/ Notes | Total Videos | Avg/Video |")
    lines.append("|-------|-------|-----------------|--------------|-----------|")
    for topic in sorted(em.get("per_topic", {}).keys()):
        td = em["per_topic"][topic]
        lines.append(
            f"| {topic} | {td['total_notes']} | {td['videos_with_notes']} "
            f"| {td['total_videos']} | {td['avg_notes_per_video']:.1f} |"
        )
    lines.append("")

    # Pipeline comparison
    lines.append("## Pipeline Comparison")
    lines.append("")
    pipelines = comparison.get("pipelines", {})
    if pipelines:
        header = "| Metric | " + " | ".join(pipelines.keys()) + " |"
        sep = "|--------|" + "|".join(["-------"] * len(pipelines)) + "|"
        lines.append(header)
        lines.append(sep)

        metrics_to_show = [
            ("Total sections", "total_sections"),
            ("Total citations", "total_citations"),
            ("Citation validity", "citation_validity"),
            ("Avg relevance", "avg_relevance"),
            ("Avg redundancy", "avg_redundancy"),
            ("Avg sections/query", "avg_sections_per_query"),
        ]
        for label, key in metrics_to_show:
            vals = " | ".join(str(pipelines[p].get(key, "N/A")) for p in pipelines)
            lines.append(f"| {label} | {vals} |")
    lines.append("")

    # By query type
    lines.append("### By Query Type")
    lines.append("")
    for qtype, type_data in sorted(comparison.get("by_query_type", {}).items()):
        lines.append(f"**{qtype}:**")
        lines.append("")
        lines.append("| Pipeline | Queries | Sections | Avg Relevance |")
        lines.append("|----------|---------|----------|---------------|")
        for pipeline, pm in sorted(type_data.items()):
            lines.append(
                f"| {pipeline} | {pm['n_queries']} | {pm['total_sections']} | {pm['avg_relevance']} |"
            )
        lines.append("")

    # By language
    lines.append("### By Language")
    lines.append("")
    for lang, lang_data in sorted(comparison.get("by_language", {}).items()):
        lines.append(f"**{lang}:**")
        lines.append("")
        lines.append("| Pipeline | Queries | Sections | Avg Relevance |")
        lines.append("|----------|---------|----------|---------------|")
        for pipeline, pm in sorted(lang_data.items()):
            lines.append(
                f"| {pipeline} | {pm['n_queries']} | {pm['total_sections']} | {pm['avg_relevance']} |"
            )
        lines.append("")

    # By topic
    lines.append("### By Topic")
    lines.append("")
    for topic, topic_data in sorted(comparison.get("by_topic", {}).items()):
        lines.append(f"**{topic}:**")
        lines.append("")
        lines.append("| Pipeline | Queries | Sections | Avg Relevance |")
        lines.append("|----------|---------|----------|---------------|")
        for pipeline, pm in sorted(topic_data.items()):
            lines.append(
                f"| {pipeline} | {pm['n_queries']} | {pm['total_sections']} | {pm['avg_relevance']} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="PR9: Evaluation harness and comparison")
    # Legacy args
    ap.add_argument(
        "--observations",
        default="outputs/observations_heuristic/observation_notes.jsonl",
    )
    ap.add_argument(
        "--grounded",
        default="outputs/grounded_heuristic/grounded_notes.jsonl",
    )
    ap.add_argument("--reports-note-taking", default="outputs/reports_note_taking")
    ap.add_argument("--reports-single-query", default="outputs/reports_single_query")
    ap.add_argument("--reports-expanded-query", default="outputs/reports_expanded_query")
    ap.add_argument("--packets-note-taking", default="outputs/query_packets_note_taking")
    ap.add_argument("--packets-single-query", default="outputs/query_based_single_query")
    ap.add_argument("--packets-expanded-query", default="outputs/query_based_expanded_query")
    # New pipeline args
    ap.add_argument("--general-notes", default=None, help="General notes JSONL")
    ap.add_argument("--query-claims", default=None, help="Query-conditioned claims JSONL")
    ap.add_argument("--note-packets", default=None, help="Note packets directory")
    ap.add_argument("--claim-packets", default=None, help="Claim packets directory")
    ap.add_argument("--inferences-note", default=None, help="Note-stream inferences directory")
    ap.add_argument("--inferences-query", default=None, help="Query-stream inferences directory")
    ap.add_argument("--reports-note-based", default=None, help="Note-based reports directory")
    ap.add_argument("--reports-query-based", default=None, help="Query-based reports directory")
    # Common
    ap.add_argument("--queries-jsonl", default=DEFAULT_QUERIES_JSONL)
    ap.add_argument("--mapping", default=DEFAULT_TOPIC_MAPPING)
    ap.add_argument("--expanded-queries", default=DEFAULT_EXPANDED_QUERIES)
    ap.add_argument("--out-dir", default="outputs/evaluation")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    manifest = build_run_manifest(
        script_name="evaluate.py",
        argv=sys.argv,
        args_dict=vars(args),
        run_config={},
    )
    manifest_path = write_run_manifest(args.out_dir, manifest)

    # Load frozen contracts
    queries = load_queries(args.queries_jsonl)
    topic_map = load_topic_mapping(args.mapping)
    if os.path.isfile(args.expanded_queries):
        expanded = load_expanded_queries(args.expanded_queries)
    else:
        expanded = {}
        print(f"  WARN: expanded queries file not found: {args.expanded_queries}, skipping")

    # Load legacy observations and grounded notes
    observations = _safe_iter_jsonl(args.observations)
    grounded = _safe_iter_jsonl(args.grounded)
    if observations or grounded:
        print(f"Loaded {len(observations)} observations, {len(grounded)} grounded notes")

    # Load new pipeline artifacts
    general_notes = _safe_iter_jsonl(args.general_notes) if args.general_notes else []
    query_claims = _safe_iter_jsonl(args.query_claims) if args.query_claims else []
    note_packets = _load_packets(args.note_packets) if args.note_packets else []
    claim_packets = _load_packets(args.claim_packets) if args.claim_packets else []

    inferences_note = []
    if args.inferences_note:
        inf_path = os.path.join(args.inferences_note, "inferences.jsonl")
        inferences_note = _safe_iter_jsonl(inf_path)

    inferences_query = []
    if args.inferences_query:
        inf_path = os.path.join(args.inferences_query, "inferences.jsonl")
        inferences_query = _safe_iter_jsonl(inf_path)

    if general_notes:
        print(f"Loaded {len(general_notes)} general notes")
    if query_claims:
        print(f"Loaded {len(query_claims)} query-conditioned claims")
    if note_packets:
        print(f"Loaded {len(note_packets)} note packets")
    if claim_packets:
        print(f"Loaded {len(claim_packets)} claim packets")
    if inferences_note:
        print(f"Loaded {len(inferences_note)} note-stream inferences")
    if inferences_query:
        print(f"Loaded {len(inferences_query)} query-stream inferences")

    # Load reports from all pipelines
    reports_by_pipeline = {}
    for name, path in [
        ("note_taking", args.reports_note_taking),
        ("single_query", args.reports_single_query),
        ("expanded_query", args.reports_expanded_query),
    ]:
        if os.path.isdir(path):
            reports = _load_reports(path)
            if reports:
                reports_by_pipeline[name] = reports
                print(f"Loaded {len(reports)} reports from {name}")

    # New stream reports
    if args.reports_note_based:
        reports = _load_reports(args.reports_note_based)
        if reports:
            reports_by_pipeline["general_note"] = reports
            print(f"Loaded {len(reports)} reports from general_note")

    if args.reports_query_based:
        reports = _load_reports(args.reports_query_based)
        if reports:
            reports_by_pipeline["query_based"] = reports
            print(f"Loaded {len(reports)} reports from query_based")

    # Load legacy query-based packets for fact validation
    packets_by_pipeline = {}
    for name, path in [
        ("single_query", args.packets_single_query),
        ("expanded_query", args.packets_expanded_query),
    ]:
        if os.path.isdir(path):
            packets = _load_packets(path)
            if packets:
                packets_by_pipeline[name] = packets
                print(f"Loaded {len(packets)} packets from {name}")

    # 1. Structural validation
    print("\n[check] Structural validation")
    structural_errors = validate_structure(
        queries, topic_map, observations, grounded,
        reports_by_pipeline, expanded, verbose=args.verbose,
        packets_by_pipeline=packets_by_pipeline,
        general_notes=general_notes or None,
        query_claims=query_claims or None,
        note_packets=note_packets or None,
        claim_packets=claim_packets or None,
        inferences_note=inferences_note or None,
        inferences_query=inferences_query or None,
    )
    if structural_errors:
        print(f"  {len(structural_errors)} error(s)")
        if args.verbose:
            for e in structural_errors[:20]:
                print(f"    {e}")
    else:
        print("  PASS")

    # 2. Extraction-level metrics
    print("\n[check] Extraction-level metrics")
    extraction_metrics = compute_extraction_metrics(
        observations, grounded, topic_map,
        general_notes=general_notes or None,
        query_claims=query_claims or None,
    )
    if args.verbose:
        print(f"  Groundedness: {extraction_metrics['groundedness']}")
        print(f"  Avg redundancy: {extraction_metrics['avg_redundancy']}")
        print(f"  Avg notes/video: {extraction_metrics['avg_notes_per_video']}")
        if "total_general_notes" in extraction_metrics:
            print(f"  General notes: {extraction_metrics['total_general_notes']}")
        if "total_query_claims" in extraction_metrics:
            print(f"  Query claims: {extraction_metrics['total_query_claims']}")

    # 3. Report-level metrics per pipeline
    print("\n[check] Report-level metrics")
    report_metrics = {}
    pbar = tqdm(total=len(reports_by_pipeline), desc="Evaluate report pipelines", unit="pipeline")
    for pipeline, reports in reports_by_pipeline.items():
        pbar.set_postfix(pipeline=pipeline)
        rm = compute_report_metrics(reports, pipeline, queries)
        report_metrics[pipeline] = rm
        if args.verbose:
            print(f"  {pipeline}: {rm['total_sections']} sections, "
                  f"citation validity={rm['citation_validity']}")
        pbar.update(1)
    pbar.close()

    # 4. Cross-pipeline comparison
    print("\n[check] Cross-pipeline comparison")
    comparison = compute_comparison(report_metrics, queries)

    # Write outputs
    os.makedirs(args.out_dir, exist_ok=True)

    evaluation = {
        "provenance": {
            "run_id": manifest["run_id"],
            "manifest_path": "run_manifest.json",
        },
        "structural_validation": {
            "passed": len(structural_errors) == 0,
            "n_errors": len(structural_errors),
            "errors": structural_errors,
        },
        "extraction_metrics": extraction_metrics,
        "report_metrics": report_metrics,
        "comparison": comparison,
    }

    eval_path = os.path.join(args.out_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(args.out_dir, "evaluation_report.md")
    md = render_comparison_md(structural_errors, extraction_metrics, report_metrics, comparison)
    with open(md_path, "w") as f:
        f.write(md)

    print(f"\n[ok] Evaluation complete")
    print(f"[ok] -> {eval_path}")
    print(f"[ok] -> {md_path}")
    print(f"[ok] -> {manifest_path}")

    if structural_errors:
        print(f"\n[warn] {len(structural_errors)} structural error(s) found")
        sys.exit(1)


if __name__ == "__main__":
    main()
