"""
Test the MDR SaMD classification prompt line-by-line, BEFORE the real study
run, to flag a rule that may be confusing a given model.

Runs entirely on the local GPU via vLLM, on a small subset of devices, at
temperature 0 (deterministic). Two probes:

  1. Leave-one-out: the full prompt with exactly one rule removed, for each
     of the 7 rules, compared to the full-prompt baseline. A big change in
     parse-success rate, or a classification that flips versus baseline,
     flags that rule as worth reviewing.
  2. Cumulative build-up: rules added back one at a time onto the mandatory
     scaffold (persona, label list, output format). A parse-success drop
     right after a specific rule is added pins the problem to that rule.

This is a debugging aid for the prompt itself, not a study result. Rerun it
whenever the canonical prompt changes, before trusting a development run.

Setup:
    pip install -r ../01_general_information_and_requirements/requirements.txt

Usage (run from the project root):
    python 02_testing/diagnose_prompt.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
        --devices 01_general_information_and_requirements/data/development/devices.csv \
        --phase development \
        --dataset-phase development \
        --num-devices 5
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# prompt.py lives in ../03_evaluation, kept there (not duplicated here) so
# run_eval.py and this script can never drift onto different prompt wording.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_evaluation"))
from prompt import RULES, assemble_system_prompt, build_chat_messages, parse_response  # noqa: E402
from src.datasets import DatasetError, load_inference_dataset, validate_phase_compatibility  # noqa: E402
from src.paths import resolve_project_path  # noqa: E402
from src.prompt import PROMPT_VERSION  # noqa: E402


def build_variants():
    """baseline + one leave-one-out variant per rule + cumulative build-up."""
    n = len(RULES)
    variants = [("baseline_full_prompt", "baseline", None, assemble_system_prompt(RULES))]

    for i in range(n):
        remaining = RULES[:i] + RULES[i + 1 :]
        variants.append((f"leave_out_rule_{i + 1}", "leave_one_out", i + 1, assemble_system_prompt(remaining)))

    variants.append(("cumulative_scaffold_only", "cumulative", 0, assemble_system_prompt([])))
    for i in range(n):
        variants.append((f"cumulative_through_rule_{i + 1}", "cumulative", i + 1, assemble_system_prompt(RULES[: i + 1])))

    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--revision",
        required=True,
        help="Exact cached Hugging Face revision/commit for the diagnostic model",
    )
    parser.add_argument("--devices", required=True)
    parser.add_argument("--phase", required=True, choices=("smoke_test", "development"))
    parser.add_argument("--dataset-phase", required=True, choices=("smoke_test", "development", "final"))
    parser.add_argument("--num-devices", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--gdn-prefill-backend", choices=("flashinfer", "triton", "cutedsl"), default=None)
    parser.add_argument("--out", default="prompt_diagnostics_results.csv")
    parser.add_argument(
        "--parse-rate-drop-threshold", type=float, default=0.20,
        help="Flag a rule if removing/adding it shifts parse-success rate by at least this much",
    )
    args = parser.parse_args()

    if args.dataset_phase == "final":
        raise DatasetError("Prompt diagnostics are forbidden on final-phase datasets")
    validate_phase_compatibility(args.phase, args.dataset_phase)
    device_path = resolve_project_path(args.devices)
    devices = load_inference_dataset(device_path)[: args.num_devices]
    print(f"{len(devices)} device(s) for the diagnostic (from {device_path})")
    output_path = resolve_project_path(args.out)
    raw_path = output_path.with_suffix(".raw.jsonl")
    for existing in (output_path, raw_path):
        if existing.exists():
            raise SystemExit(f"{existing} already exists; choose a different --out (results are never overwritten)")

    from vllm import LLM, SamplingParams

    print(f"Loading {args.model}@{args.revision} ...")
    llm_kwargs = dict(
        model=args.model,
        revision=args.revision,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.gdn_prefill_backend is not None:
        llm_kwargs["gdn_prefill_backend"] = args.gdn_prefill_backend
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)

    variants = build_variants()
    rows = []
    baseline_by_device = {}
    raw_records = []

    for variant_id, kind, rule_number, system_prompt in variants:
        prompts = [
            tokenizer.apply_chat_template(
                build_chat_messages(d.description, system_prompt=system_prompt),
                tokenize=False, add_generation_prompt=True,
            )
            for d in devices
        ]
        outputs = llm.generate(prompts, sampling_params)

        for device, output in zip(devices, outputs):
            raw_text = output.outputs[0].text
            parsed = parse_response(raw_text)
            rows.append({
                "variant_id": variant_id, "kind": kind, "rule_number": rule_number,
                "device_id": device.device_id, "parse_ok": parsed["parse_ok"],
                "classification": parsed["classification"], "parse_error": parsed["parse_error"],
                "prompt_version": PROMPT_VERSION, "model": args.model, "revision": args.revision,
                "max_tokens": args.max_tokens, "finish_reason": output.outputs[0].finish_reason,
                "n_tokens": len(output.outputs[0].token_ids),
            })
            raw_records.append({
                "variant_id": variant_id, "device_id": device.device_id,
                "finish_reason": output.outputs[0].finish_reason, "raw_text": raw_text,
            })
            if variant_id == "baseline_full_prompt":
                baseline_by_device[device.device_id] = parsed["classification"]

    output_path = resolve_project_path(args.out)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant_id", "kind", "rule_number", "device_id", "parse_ok", "classification", "parse_error",
                                               "prompt_version", "model", "revision", "max_tokens", "finish_reason", "n_tokens"])
        writer.writeheader()
        writer.writerows(rows)
    with raw_path.open("w", encoding="utf-8") as f:
        for rec in raw_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Raw model outputs written to {raw_path}")
    print(f"\nRaw per-variant responses written to {output_path}")

    baseline_parse_rate = sum(r["parse_ok"] for r in rows if r["variant_id"] == "baseline_full_prompt") / len(devices)
    print(f"\nBaseline (full prompt) parse-success rate: {baseline_parse_rate:.0%}")

    print("\n--- Leave-one-out: does removing rule N change behavior? ---")
    print(f"{'rule':>4}  {'parse_rate':>10}  {'delta_vs_base':>14}  {'flip_rate':>10}  flagged")
    flagged_rules = []
    for rule_number in range(1, len(RULES) + 1):
        variant_rows = [r for r in rows if r["variant_id"] == f"leave_out_rule_{rule_number}"]
        parse_rate = sum(r["parse_ok"] for r in variant_rows) / len(variant_rows)
        compared = [r for r in variant_rows if r["parse_ok"] and r["device_id"] in baseline_by_device]
        flips = sum(1 for r in compared if r["classification"] != baseline_by_device[r["device_id"]])
        flip_rate = flips / len(compared) if compared else float("nan")
        delta = parse_rate - baseline_parse_rate
        flagged = abs(delta) >= args.parse_rate_drop_threshold or (compared and flip_rate >= 0.25)
        if flagged:
            flagged_rules.append(rule_number)
        print(f"{rule_number:>4}  {parse_rate:>10.0%}  {delta:>+14.0%}  {(flip_rate if compared else float('nan')):>10.0%}  {'<-- FLAGGED' if flagged else ''}")

    print("\n--- Cumulative build-up: parse rate as rules are added back in order ---")
    print(f"{'rules included':>14}  {'parse_rate':>10}  {'delta_from_prev':>16}")
    prev_rate = None
    for i in range(0, len(RULES) + 1):
        variant_id = "cumulative_scaffold_only" if i == 0 else f"cumulative_through_rule_{i}"
        variant_rows = [r for r in rows if r["variant_id"] == variant_id]
        parse_rate = sum(r["parse_ok"] for r in variant_rows) / len(variant_rows)
        delta = "" if prev_rate is None else f"{parse_rate - prev_rate:+.0%}"
        flag = " <-- FLAGGED" if prev_rate is not None and (parse_rate - prev_rate) <= -args.parse_rate_drop_threshold else ""
        print(f"{i:>14}  {parse_rate:>10.0%}  {delta:>16}{flag}")
        prev_rate = parse_rate

    if flagged_rules:
        print(f"\nRule(s) flagged as possibly problematic: {flagged_rules}. Review their wording in ../03_evaluation/prompt.py.")
    else:
        print("\nNo rule was flagged by the leave-one-out probe.")


if __name__ == "__main__":
    main()
