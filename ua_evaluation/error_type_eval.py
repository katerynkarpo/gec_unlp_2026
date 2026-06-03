#!/usr/bin/env python3
"""Per error type evaluation using gold M2 + ERRANT hypothesis M2.

Default mode uses ERRANT's native comparison (`errant_compare -cse -cat 3`)
for span-based correction + classification. A legacy overlap matcher is kept
as a fallback for backwards compatibility.

Usage:
    # Step 1: generate hyp M2 (if not already done)
    errant_parallel -orig /tmp/source.tok -cor predictions.eval.txt -out /tmp/hyp.m2

    # Step 2: run this script
    python ua_evaluation/error_type_eval.py --hyp /tmp/hyp.m2 --ref gold.m2

    # Or all-in-one (auto-generates hyp M2):
    python ua_evaluation/error_type_eval.py --pred predictions.eval.txt --ref gold.m2

Options:
    --hyp       Path to hypothesis M2 (from errant_parallel)
    --pred      Path to tokenized predictions (will auto-generate hyp M2)
    --ref       Path to gold M2 file
    --save      Directory to save report (default: same as --pred or cwd)
    --annotator Annotator ID to evaluate against (default: 0)
    --backend   errant (default) or legacy
"""

import argparse
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

# Ensure project root is importable when this script is run as a file path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.tokenizer import tokenize as ua_tokenize


def parse_m2(m2_path: str) -> list[dict]:
    """Parse M2 file into list of {source, edits}."""
    sentences = []
    current_src = None
    current_edits = []

    with open(m2_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("S "):
                if current_src is not None:
                    sentences.append({"source": current_src, "edits": current_edits})
                current_src = line[2:]
                current_edits = []
            elif line.startswith("A "):
                parts = line[2:].split("|||")
                if len(parts) < 6:
                    continue
                span = parts[0].strip()
                try:
                    start, end = span.split()
                    start, end = int(start), int(end)
                except ValueError:
                    continue
                err_type = parts[1].strip()
                correction = parts[2].strip()
                try:
                    annotator = int(parts[5].strip())
                except ValueError:
                    annotator = 0
                current_edits.append({
                    "start": start,
                    "end": end,
                    "type": err_type,
                    "correction": correction,
                    "annotator": annotator,
                })
    if current_src is not None:
        sentences.append({"source": current_src, "edits": current_edits})
    return sentences


def spans_overlap(s1_start, s1_end, s2_start, s2_end) -> bool:
    """Check if two spans overlap or are adjacent (for insertions)."""
    # Handle insertions (start == end): overlap if they touch
    if s1_start == s1_end:
        return s2_start <= s1_start <= s2_end
    if s2_start == s2_end:
        return s1_start <= s2_start <= s1_end
    return s1_start < s2_end and s2_start < s1_end


def edits_match(gold_edit: dict, hyp_edit: dict) -> bool:
    """Check if a hypothesis edit matches a gold edit (same span + same correction)."""
    return (
        gold_edit["start"] == hyp_edit["start"]
        and gold_edit["end"] == hyp_edit["end"]
        and gold_edit["correction"] == hyp_edit["correction"]
    )


def f_score(tp: int, fp: int, fn: int, beta: float = 0.5) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        f = 0.0
    else:
        f = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
    return prec, rec, f


def evaluate_legacy(
    gold_sentences: list[dict],
    hyp_sentences: list[dict],
    annotator_id: int = -1,
    beta: float = 0.5,
):
    """Span-based correction matching; attribute counts to UA gold types.

    If annotator_id == -1, choose the best annotator per sentence using F-beta
    over exact correction matches (same span + same correction), with ERRANT-like
    tie-breaks: higher TP, lower FP, lower FN.
    """
    assert len(gold_sentences) == len(hyp_sentences), (
        f"Gold has {len(gold_sentences)} sentences, hyp has {len(hyp_sentences)}"
    )

    type_tp = Counter()
    type_fp = Counter()
    type_fn = Counter()
    running_tp = 0
    running_fp = 0
    running_fn = 0

    for gold_sent, hyp_sent in zip(gold_sentences, hyp_sentences):
        hyp_edits = [
            e for e in hyp_sent["edits"]
            if e["type"] not in {"noop", "UNK"} and e["start"] != -1
        ]
        raw_gold_edits = list(gold_sent["edits"])
        all_gold_edits = [
            e for e in gold_sent["edits"]
            if e["type"] not in {"noop", "UNK"}
            and e["start"] != -1
        ]

        if annotator_id >= 0:
            gold_edits = [e for e in all_gold_edits if e["annotator"] == annotator_id]
        else:
            # Include annotators that only have noop edits (empty reference),
            # since ERRANT can choose them in multi-reference evaluation.
            by_annotator: dict[int, list[dict]] = {}
            for e in raw_gold_edits:
                by_annotator.setdefault(e["annotator"], [])
            for e in all_gold_edits:
                by_annotator.setdefault(e["annotator"], []).append(e)
            if not by_annotator:
                by_annotator = {0: []}
            if not by_annotator:
                gold_edits = []
            else:
                best_ann = None
                best_tp = best_fp = best_fn = 0
                best_f = -1.0
                hyp_keys = [(e["start"], e["end"], e["correction"]) for e in hyp_edits]
                hyp_counter = Counter(hyp_keys)
                for ann, edits_for_ann in by_annotator.items():
                    gold_keys = [(e["start"], e["end"], e["correction"]) for e in edits_for_ann]
                    gold_counter = Counter(gold_keys)
                    tp = sum(min(hyp_counter[k], gold_counter[k]) for k in gold_counter)
                    fp = len(hyp_keys) - tp
                    fn = len(gold_keys) - tp
                    _, _, f = f_score(
                        running_tp + tp,
                        running_fp + fp,
                        running_fn + fn,
                        beta=beta,
                    )
                    if (
                        (f > best_f)
                        or (f == best_f and tp > best_tp)
                        or (f == best_f and tp == best_tp and fp < best_fp)
                        or (f == best_f and tp == best_tp and fp == best_fp and fn < best_fn)
                    ):
                        best_f = f
                        best_tp = tp
                        best_fp = fp
                        best_fn = fn
                        best_ann = ann
                gold_edits = by_annotator.get(best_ann, [])

        # Track which edits have been matched
        gold_matched = [False] * len(gold_edits)
        hyp_matched = [False] * len(hyp_edits)

        # Pass 1: exact correction match (same span + same correction) = TP
        for gi, ge in enumerate(gold_edits):
            for hi, he in enumerate(hyp_edits):
                if hyp_matched[hi]:
                    continue
                if edits_match(ge, he):
                    type_tp[ge["type"]] += 1
                    gold_matched[gi] = True
                    hyp_matched[hi] = True
                    break

        # Pass 2: same span, wrong correction = FP + FN for the gold type
        for gi, ge in enumerate(gold_edits):
            if gold_matched[gi]:
                continue
            for hi, he in enumerate(hyp_edits):
                if hyp_matched[hi]:
                    continue
                if ge["start"] == he["start"] and ge["end"] == he["end"]:
                    # Same span, wrong correction
                    type_fn[ge["type"]] += 1
                    type_fp[ge["type"]] += 1
                    gold_matched[gi] = True
                    hyp_matched[hi] = True
                    break

        # Unmatched gold edits = FN
        for gi, ge in enumerate(gold_edits):
            if not gold_matched[gi]:
                type_fn[ge["type"]] += 1

        # Unmatched hyp edits = FP.
        # Attribute to same-span or overlapping gold type when possible,
        # otherwise "_uncategorized".
        for hi, he in enumerate(hyp_edits):
            if hyp_matched[hi]:
                continue
            same_span = [
                ge for ge in gold_edits
                if ge["start"] == he["start"] and ge["end"] == he["end"]
            ]
            if same_span:
                type_fp[same_span[0]["type"]] += 1
                continue

            overlapping = [
                ge for ge in gold_edits
                if spans_overlap(ge["start"], ge["end"], he["start"], he["end"])
            ]
            if overlapping:
                type_fp[overlapping[0]["type"]] += 1
                continue

            best_type = None
            best_dist = float("inf")
            for ge in gold_edits:
                dist = abs(ge["start"] - he["start"]) + abs(ge["end"] - he["end"])
                if dist < best_dist:
                    best_dist = dist
                    best_type = ge["type"]
            if best_type is not None:
                type_fp[best_type] += 1
            else:
                type_fp["_uncategorized"] += 1

        sent_tp = sum(1 for matched in hyp_matched if matched)
        sent_fp = len(hyp_edits) - sent_tp
        sent_fn = sum(1 for matched in gold_matched if not matched)
        running_tp += sent_tp
        running_fp += sent_fp
        running_fn += sent_fn

    return type_tp, type_fp, type_fn


def format_report(type_tp: Counter, type_fp: Counter, type_fn: Counter, beta: float = 0.5) -> str:
    all_types = sorted(
        {t for t in list(type_tp) + list(type_fp) + list(type_fn) if not t.startswith("_")},
        key=lambda t: -(type_tp[t] + type_fn[t]),
    )

    lines = []
    lines.append("# Per UA Error Type Metrics")
    lines.append("")
    lines.append(f"| Error Type | Gold | TP | FP | FN | Prec | Rec | F{beta} |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for t in all_types:
        tp = type_tp[t]
        fp = type_fp[t]
        fn = type_fn[t]
        gold = tp + fn
        prec, rec, f05 = f_score(tp, fp, fn, beta=beta)
        lines.append(
            f"| **{t}** | {gold} | {tp} | {fp} | {fn} "
            f"| {prec:.4f} | {rec:.4f} | {f05:.4f} |"
        )

    total_tp = sum(type_tp.values())
    total_fp = sum(type_fp.values())  # include _uncategorized FP
    total_fn = sum(type_fn.values())
    prec, rec, f05 = f_score(total_tp, total_fp, total_fn, beta=beta)
    lines.append(
        f"| **TOTAL** | {total_tp + total_fn} | {total_tp} | {total_fp} | {total_fn} "
        f"| {prec:.4f} | {rec:.4f} | {f05:.4f} |"
    )
    lines.append("")

    uncat = type_fp.get("_uncategorized", 0)
    if uncat:
        lines.append(f"_Uncategorized FP (model edits not near any gold edit): {uncat}_")
        lines.append("")

    return "\n".join(lines)


def filter_reference_by_annotator(ref_path: str, annotator_id: int, out_path: Path) -> Path:
    """Write an M2 with only edits from one annotator (plus all source lines)."""
    blocks: list[list[str]] = []
    current: list[str] = []
    with open(ref_path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if current:
                    blocks.append(current)
                    current = []
                continue
            current.append(line.rstrip("\n"))
    if current:
        blocks.append(current)

    with open(out_path, "w", encoding="utf-8") as out:
        for block in blocks:
            src_line = next((line for line in block if line.startswith("S ")), None)
            if src_line is None:
                continue
            out.write(src_line + "\n")
            for line in block:
                if not line.startswith("A "):
                    continue
                parts = line[2:].split("|||")
                if len(parts) < 6:
                    continue
                try:
                    ann = int(parts[5].strip())
                except ValueError:
                    ann = 0
                if ann == annotator_id:
                    out.write(line + "\n")
            out.write("\n")

    return out_path


def generate_hyp_m2(pred_path: str, ref_m2_path: str, out_dir: Path) -> str:
    """Generate hypothesis M2 using errant_parallel."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract source from gold M2
    source_path = out_dir / "source.tok"
    with open(ref_m2_path, encoding="utf-8") as f, open(source_path, "w", encoding="utf-8") as out:
        for line in f:
            if line.startswith("S "):
                out.write(line[2:])

    hyp_m2_path = out_dir / "hyp.m2"
    subprocess.run(
        ["errant_parallel", "-orig", str(source_path), "-cor", pred_path, "-out", str(hyp_m2_path)],
        check=True,
    )
    return str(hyp_m2_path)


def tokenize_file(input_path: str, output_path: Path) -> Path:
    """Tokenize predictions exactly like ua_evaluation/evaluation.py."""
    with open(input_path, encoding="utf-8") as f, open(output_path, "w", encoding="utf-8") as out:
        for line in f:
            line = line.rstrip("\n")
            out.write(ua_tokenize(line) + "\n")
    return output_path


def parse_errant_compare_output(output_text: str) -> tuple[Counter, Counter, Counter]:
    """Parse per-category TP/FP/FN table from errant_compare output."""
    type_tp = Counter()
    type_fp = Counter()
    type_fn = Counter()

    in_table = False
    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line:
            if in_table:
                break
            continue
        if line.startswith("Category"):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("="):
            break

        parts = re.split(r"\s+", line)
        if len(parts) != 7:
            continue
        err_type, tp_s, fp_s, fn_s, *_ = parts
        if err_type.lower() == "category":
            continue
        try:
            type_tp[err_type] = int(tp_s)
            type_fp[err_type] = int(fp_s)
            type_fn[err_type] = int(fn_s)
        except ValueError:
            continue

    return type_tp, type_fp, type_fn


def evaluate_errant(hyp_m2: str, ref_m2: str, beta: float) -> tuple[Counter, Counter, Counter, str]:
    """Run errant_compare and parse per-type counts."""
    cmd = [
        "errant_compare",
        "-hyp",
        hyp_m2,
        "-ref",
        ref_m2,
        "-cse",
        "-cat",
        "3",
        "-b",
        str(beta),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "errant_compare failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    type_tp, type_fp, type_fn = parse_errant_compare_output(proc.stdout)
    return type_tp, type_fp, type_fn, proc.stdout


def has_compatible_error_types(gold: list[dict], hyp: list[dict]) -> bool:
    """Return True when gold/hyp type vocabularies overlap enough for -cse."""
    gold_types = {
        e["type"]
        for sent in gold
        for e in sent["edits"]
        if e["type"] != "noop" and e["start"] != -1
    }
    hyp_types = {
        e["type"]
        for sent in hyp
        for e in sent["edits"]
        if e["type"] != "noop" and e["start"] != -1
    }
    if not gold_types or not hyp_types:
        return False
    overlap = len(gold_types & hyp_types)
    overlap_ratio = overlap / max(len(gold_types), 1)
    return overlap > 0 and overlap_ratio >= 0.2


def main():
    parser = argparse.ArgumentParser(description="Per UA error type evaluation.")
    parser.add_argument("--hyp", default=None, help="Path to hypothesis M2 (from errant_parallel)")
    parser.add_argument("--pred", default=None, help="Path to tokenized predictions (auto-generates hyp M2)")
    parser.add_argument("--ref", required=True, help="Path to gold M2 file")
    parser.add_argument("--save", default=None, help="Directory to save report")
    parser.add_argument(
        "--no-tokenize",
        action="store_true",
        help="Do not tokenize --pred input (default matches ua_evaluation: tokenize)",
    )
    parser.add_argument(
        "--annotator",
        type=int,
        default=-1,
        help="Annotator ID (default: -1 = best annotator per sentence)",
    )
    parser.add_argument("--beta", type=float, default=0.5, help="F-beta weight (default: 0.5)")
    parser.add_argument(
        "--backend",
        choices=["errant", "legacy"],
        default="errant",
        help="Scoring backend: errant (default) or legacy overlap matcher",
    )
    args = parser.parse_args()

    if not args.hyp and not args.pred:
        parser.error("Provide either --hyp (hypothesis M2) or --pred (predictions file)")

    work_dir = Path(tempfile.mkdtemp(prefix="error_type_eval_"))

    # Generate hyp M2 if needed
    if args.hyp:
        hyp_m2_path = args.hyp
    else:
        pred_for_alignment = args.pred
        if not args.no_tokenize:
            print("Tokenizing predictions...")
            tok_path = work_dir / "pred.tokenized.txt"
            pred_for_alignment = str(tokenize_file(args.pred, tok_path))
            print(f"  -> {pred_for_alignment}")
        print("Generating hypothesis M2 with errant_parallel...")
        hyp_m2_path = generate_hyp_m2(pred_for_alignment, args.ref, work_dir)
        print(f"  -> {hyp_m2_path}")

    if args.annotator >= 0:
        filtered_ref_path = str(
            filter_reference_by_annotator(args.ref, args.annotator, work_dir / "ref.filtered.m2")
        )
    else:
        filtered_ref_path = args.ref

    print("Loading gold M2...")
    gold = parse_m2(filtered_ref_path)
    print(f"  {len(gold)} sentences")

    print("Loading hypothesis M2...")
    hyp = parse_m2(hyp_m2_path)
    print(f"  {len(hyp)} sentences")

    print(f"Evaluating with backend: {args.backend}")
    raw_errant_output = None
    if args.backend == "errant":
        # -cse requires compatible error type taxonomy in both hyp/ref M2.
        # UA gold labels (e.g., "Punctuation") and ERRANT hyp labels
        # (e.g., "R:PUNCT") are often incompatible, so we fallback.
        if has_compatible_error_types(gold, hyp):
            type_tp, type_fp, type_fn, raw_errant_output = evaluate_errant(
                hyp_m2=hyp_m2_path,
                ref_m2=filtered_ref_path,
                beta=args.beta,
            )
        else:
            print(
                "Type taxonomies in gold/hyp M2 are incompatible for errant -cse; "
                "falling back to legacy per-type matching."
            )
            type_tp, type_fp, type_fn = evaluate_legacy(
                gold,
                hyp,
                annotator_id=args.annotator,
                beta=args.beta,
            )
    else:
        type_tp, type_fp, type_fn = evaluate_legacy(
            gold,
            hyp,
            annotator_id=args.annotator,
            beta=args.beta,
        )

    report = format_report(type_tp, type_fp, type_fn, beta=args.beta)
    print()
    print(report)

    # Save
    if args.save:
        save_dir = args.save
    elif args.pred:
        save_dir = str(Path(args.pred).parent)
    else:
        save_dir = "."
    out_path = Path(save_dir) / "error_type_metrics.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved to {out_path}")

    if raw_errant_output is not None:
        raw_out_path = Path(save_dir) / "error_type_metrics.errant.txt"
        with open(raw_out_path, "w", encoding="utf-8") as f:
            f.write(raw_errant_output)
        print(f"Saved raw ERRANT output to {raw_out_path}")


if __name__ == "__main__":
    main()
