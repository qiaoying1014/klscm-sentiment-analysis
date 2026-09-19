from __future__ import annotations

import argparse
import json
from pathlib import Path

# Dispatch the read-only downstream stage before importing legacy model modules.
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "participant-experience":
        from .participant_experience import main as experience_main
        experience_main(sys.argv[2:])
        raise SystemExit(0)

from .adjudication import (
    analyze_agreement, finalize_gold_standard, import_reviewer_file,
    initialize_adjudication_package,
)
from .pipeline import Pipeline
from .storage import read_table, write_table
from .single_researcher_audit import (
    AUDIT_SEED, CURRENT_AUDIT_BLIND, CURRENT_AUDIT_FINAL, CURRENT_AUDIT_KEY,
    estimate_v8_production_cost, finalize_architecture_audit,
    finalize_audit, finalize_single_audit, run_v8_production, write_architecture_audit_package,
    write_audit_package,
)
from .cost_conservative import (
    batch_status, create_batch, finalize_production, finalize_topic_gate, import_batch,
    replay_calibration, submit_batch, write_production_report,
)
from .validation import make_validation_sample
from .disagreement_audit import create_disagreement_package
from .final_taxonomy import finalize_taxonomy, prepare_final_taxonomy
from .topic_discovery import (
    prepare_c1_topic_review, prepare_model_selection_review, prepare_topic_corpus,
    run_c1_refined, run_topic_discovery, topic_review_summary,
    write_model_selection_summary,
)
from .absa_v1 import batch_status as absa_batch_status, import_batch as import_absa_batch
from .absa_v1 import cost_estimate as absa_v1_cost_estimate, create_batch_requests, finalize_batch as finalize_absa_batch, submit_batch as submit_absa_batch
from .absa_v1 import create_single_researcher_audit, evaluate_validation, freeze_gold
from .absa_fp_diagnostic import create_fp_diagnostic, finalize_fp_diagnostic
from .absa_precision_development import (
    create_experiment as create_precision_experiment,
    experiment_status as precision_experiment_status,
    finalize_experiment as finalize_precision_experiment,
    import_experiment as import_precision_experiment,
    submit_experiment as submit_precision_experiment,
)
from .absa_residual_diagnostic import create_residual_diagnostic
from .absa_aspect_ownership_development import (
    create_experiment as create_ownership_experiment,
    experiment_status as ownership_experiment_status,
    finalize_experiment as finalize_ownership_experiment,
    import_experiment as import_ownership_experiment,
    submit_experiment as submit_ownership_experiment,
)
from .absa_production import (
    audit_pre_submission as audit_absa_production,
    create_retry_package as create_absa_production_retry,
    create_production as create_absa_production,
    finalize_production as finalize_absa_production,
    import_production as import_absa_production,
    production_status as absa_production_status,
    submit_production as submit_absa_production,
)
from .local_absa import (
    baseline_evaluate as local_absa_baseline_evaluate,
    baseline_train as local_absa_baseline_train,
    create_audit as create_local_absa_audit,
    format_gpu_preflight as format_local_gpu_preflight,
    gpu_preflight as local_gpu_preflight,
)
from .absa_descriptive_analysis import create_descriptive_analysis as create_absa_descriptive_analysis
from .absa_inferential_analysis import create_inferential_analysis as create_absa_inferential_analysis
from .absa_dashboard_data import create_dashboard_data as create_absa_dashboard_data
from .absa_final_outputs import create_thesis_outputs as create_absa_thesis_outputs
from .aspect_level_themes import create_aspect_level_themes
from .aspect_level_themes_review import (
    finalize_reviewed_taxonomy, generate_review_package, validate_review_package,
)
from .blog_analysis import (
    audit_blog_source, blog_absa_status, create_blog_absa_requests,
    finalize_blog_absa, finalize_theme_mappings, generate_theme_candidates,
    import_blog_absa, submit_blog_absa, verify_frozen_upstream,
)
from .cross_source_analysis import create_cross_source_analysis


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "participant-experience":
        from .participant_experience import main as experience_main
        return experience_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description="KLSCM multilingual topic discovery and ABSA pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download-language-model")
    sub.add_parser("prepare")
    for name in ["language-review", "relevance", "topics", "absa"]:
        child = sub.add_parser(name)
        child.add_argument("--run-api", action="store_true")
        child.add_argument("--limit", type=int)
        if name == "language-review":
            child.add_argument("--source", choices=["instagram", "blog"])
        if name == "relevance":
            child.add_argument("--pilot-size", type=int)
    sample = sub.add_parser("validation-sample")
    sample.add_argument("--size", type=int, default=300)
    relevance_sample = sub.add_parser("relevance-review-sample")
    relevance_sample.add_argument("--size", type=int, default=300)
    relevance_sample.add_argument("--repeats", type=int, default=30)
    fresh_holdout = sub.add_parser("relevance-fresh-holdout")
    fresh_holdout.add_argument("--size", type=int, default=150)
    fresh_holdout.add_argument("--repeats", type=int, default=15)
    for stage, default_size, default_repeats in [
        ("pilot", 100, 10), ("calibration", 300, 30),
        ("fresh-holdout", 300, 30),
    ]:
        child = sub.add_parser(f"relevance-v7-{stage}-sample")
        child.add_argument("--size", type=int, default=default_size)
        child.add_argument("--repeats", type=int, default=default_repeats)
    v7_finalize = sub.add_parser("relevance-v7-finalize")
    v7_finalize.add_argument(
        "--stage", choices=["pilot", "calibration", "fresh_holdout"], required=True,
    )
    v7_repeat_audit = sub.add_parser("relevance-v7-repeat-audit")
    v7_repeat_audit.add_argument(
        "--stage", choices=["pilot", "calibration", "fresh_holdout"], required=True,
    )
    v7_score = sub.add_parser("relevance-v7-score-calibration")
    v7_score.add_argument("--run-api", action="store_true")
    v8_score = sub.add_parser("relevance-v8-score-calibration")
    v8_score.add_argument("--run-api", action="store_true")
    v8_production = sub.add_parser("relevance-v8-production")
    v8_production.add_argument("--run-api", action="store_true")
    sub.add_parser("relevance-v8-cost-estimate")
    adjudication_init = sub.add_parser("relevance-v8-adjudication-init")
    adjudication_init.add_argument("--seed", type=int, default=8042)
    adjudication_import = sub.add_parser("relevance-v8-adjudication-import")
    adjudication_import.add_argument("--file", type=Path, required=True)
    adjudication_import.add_argument("--reviewer-id", required=True)
    adjudication_import.add_argument("--reviewer-version", required=True)
    adjudication_agreement = sub.add_parser("relevance-v8-adjudication-agreement")
    adjudication_agreement.add_argument("--reviewer-a", type=Path, required=True)
    adjudication_agreement.add_argument("--reviewer-b", type=Path, required=True)
    adjudication_finalize = sub.add_parser("relevance-v8-adjudication-finalize")
    adjudication_finalize.add_argument("--reviewer-a", type=Path, required=True)
    adjudication_finalize.add_argument("--reviewer-b", type=Path, required=True)
    adjudication_finalize.add_argument("--adjudication-report", type=Path, required=True)
    adjudication_finalize.add_argument("--gold-version", required=True)
    sub.add_parser("relevance-policy-alignment-sample")
    sub.add_parser("relevance-policy-alignment-finalize")
    sub.add_parser("relevance-finalize")
    sub.add_parser("relevance-validate")
    rescore = sub.add_parser("relevance-rescore-validation")
    rescore.add_argument("--run-api", action="store_true")
    rescore.add_argument(
        "--split", choices=["calibration", "holdout"], default="calibration",
    )
    fresh_rescore = sub.add_parser("relevance-rescore-fresh-holdout")
    fresh_rescore.add_argument("--run-api", action="store_true")
    single_create = sub.add_parser("relevance-v8-single-audit-create")
    single_create.add_argument("--decisions", type=Path, default=Path("data/processed/relevance_v8_single_researcher_cost_conservative/production_v1/relevance_v8_cost_conservative_decisions_v1.csv"))
    single_create.add_argument("--output-dir", type=Path, default=Path("data/processed/relevance_v8_single_researcher_audit"))
    single_create.add_argument("--per-decision", type=int, default=75)
    single_create.add_argument("--seed", type=int, default=AUDIT_SEED)
    single_finalize = sub.add_parser("relevance-v8-single-audit-finalize")
    single_finalize.add_argument("--audit-file", type=Path, default=CURRENT_AUDIT_BLIND)
    single_finalize.add_argument("--key-file", type=Path, default=CURRENT_AUDIT_KEY)
    single_finalize.add_argument("--output-dir", type=Path, default=CURRENT_AUDIT_FINAL)
    architecture_create = sub.add_parser("relevance-v8-architecture-audit-create")
    architecture_create.add_argument("--decisions", type=Path, default=Path("data/processed/relevance_v8_single_researcher_cost_conservative/production_v1/relevance_v8_cost_conservative_decisions_v1.parquet"))
    architecture_create.add_argument("--output-dir", type=Path, default=Path("data/processed/relevance_v8_single_researcher_architecture_audit_v2"))
    architecture_create.add_argument("--include-sample", type=int, default=75)
    architecture_create.add_argument("--deterministic-exclude-sample", type=int, default=50)
    architecture_create.add_argument("--seed", type=int, default=AUDIT_SEED)
    architecture_finalize = sub.add_parser("relevance-v8-architecture-audit-finalize")
    architecture_finalize.add_argument("--audit-file", type=Path, default=Path("data/processed/relevance_v8_single_researcher_architecture_audit_v2/relevance_v8_architecture_audit_blind_v2.csv"))
    architecture_finalize.add_argument("--key-file", type=Path, default=Path("data/processed/relevance_v8_single_researcher_architecture_audit_v2/relevance_v8_architecture_audit_key_v2.csv"))
    architecture_finalize.add_argument("--output-dir", type=Path, default=Path("data/processed/relevance_v8_single_researcher_architecture_audit_v2/final_v2"))
    sub.add_parser("relevance-v8-cost-conservative-replay")
    sub.add_parser("relevance-v8-cost-conservative-report")
    for stage in ["initial", "verification"]:
        create = sub.add_parser(f"relevance-v8-batch-{stage}-create")
        create.add_argument("--max-estimated-tokens", type=int, default=1_250_000)
        submit = sub.add_parser(f"relevance-v8-batch-{stage}-submit")
        submit.add_argument("--chunk", required=True); submit.add_argument("--run-api", action="store_true")
        status = sub.add_parser(f"relevance-v8-batch-{stage}-status")
        status.add_argument("--chunk", required=True)
        imp = sub.add_parser(f"relevance-v8-batch-{stage}-import")
        imp.add_argument("--chunk", required=True); imp.add_argument("--allow-partial", action="store_true")
    sub.add_parser("relevance-v8-cost-conservative-finalize")
    sub.add_parser("relevance-v8-cost-conservative-topic-gate")
    sub.add_parser("relevance-v8-architecture-disagreement-audit")
    sub.add_parser("topic-discovery-prepare")
    sub.add_parser("topic-discovery-run")
    sub.add_parser("topic-discovery-review-prepare")
    sub.add_parser("topic-discovery-review-summary")
    sub.add_parser("topic-discovery-refine-c1")
    sub.add_parser("topic-discovery-model-selection-prepare")
    sub.add_parser("topic-discovery-model-selection-summary")
    sub.add_parser("topic-taxonomy-prepare")
    sub.add_parser("topic-taxonomy-finalize")
    sub.add_parser("absa-v1-audit-create")
    sub.add_parser("absa-v1-freeze-gold")
    sub.add_parser("absa-v1-cost-estimate")
    sub.add_parser("absa-v1-validation-evaluate")
    sub.add_parser("absa-v1-fp-diagnostic-create")
    sub.add_parser("absa-v1-fp-diagnostic-finalize")
    sub.add_parser("absa-v1-precision-development-create")
    precision_submit = sub.add_parser("absa-v1-precision-development-batch-submit")
    precision_submit.add_argument("--run-api", action="store_true")
    precision_status = sub.add_parser("absa-v1-precision-development-batch-status")
    precision_status.add_argument("--run-api", action="store_true")
    precision_import = sub.add_parser("absa-v1-precision-development-batch-import")
    precision_import.add_argument("--run-api", action="store_true")
    sub.add_parser("absa-v1-precision-development-finalize")
    sub.add_parser("absa-v1-precision-residual-diagnostic")
    sub.add_parser("absa-v1-aspect-ownership-development-create")
    ownership_submit = sub.add_parser("absa-v1-aspect-ownership-development-batch-submit")
    ownership_submit.add_argument("--run-api", action="store_true")
    ownership_status = sub.add_parser("absa-v1-aspect-ownership-development-batch-status")
    ownership_status.add_argument("--run-api", action="store_true")
    ownership_import = sub.add_parser("absa-v1-aspect-ownership-development-batch-import")
    ownership_import.add_argument("--run-api", action="store_true")
    sub.add_parser("absa-v1-aspect-ownership-development-finalize")
    sub.add_parser("absa-v1-production-create")
    production_submit = sub.add_parser("absa-v1-production-batch-submit")
    production_submit.add_argument("--run-api", action="store_true")
    production_status = sub.add_parser("absa-v1-production-batch-status")
    production_status.add_argument("--run-api", action="store_true")
    production_import = sub.add_parser("absa-v1-production-batch-import")
    production_import.add_argument("--run-api", action="store_true")
    sub.add_parser("absa-v1-production-finalize")
    sub.add_parser("absa-v1-production-audit")
    sub.add_parser("absa-v1-production-retry-create")
    sub.add_parser("absa-v1-descriptive-analysis")
    sub.add_parser("absa-v1-inferential-analysis")
    sub.add_parser("absa-v1-dashboard-data")
    sub.add_parser("absa-v1-thesis-outputs")
    sub.add_parser("absa-v1-aspect-level-themes")
    sub.add_parser("absa-v1-alta-review-generate")
    review_validate = sub.add_parser("absa-v1-alta-review-validate")
    review_validate.add_argument("--mapping", type=Path)
    review_finalize = sub.add_parser("absa-v1-alta-review-finalize")
    review_finalize.add_argument("--mapping", type=Path)
    sub.add_parser("local-absa-gpu-check")
    sub.add_parser("local-absa-audit")
    sub.add_parser("local-absa-baseline-create")
    sub.add_parser("local-absa-baseline-train")
    sub.add_parser("local-absa-baseline-evaluate")
    sub.add_parser("blog-analysis-audit")
    sub.add_parser("blog-integrity-check")
    sub.add_parser("blog-absa-create")
    blog_submit = sub.add_parser("blog-absa-submit")
    blog_submit.add_argument("--run-api", action="store_true")
    blog_status = sub.add_parser("blog-absa-status")
    blog_status.add_argument("--run-api", action="store_true")
    blog_import = sub.add_parser("blog-absa-import")
    blog_import.add_argument("--run-api", action="store_true")
    sub.add_parser("blog-absa-finalize")
    sub.add_parser("blog-theme-candidates")
    blog_theme_finalize = sub.add_parser("blog-theme-finalize")
    blog_theme_finalize.add_argument("--mapping", type=Path)
    sub.add_parser("blog-emergent-theme-audit")
    sub.add_parser("blog-emergent-theme-finalize")
    sub.add_parser("cross-source-analysis")
    for scope in ["validation", "full"]:
        create = sub.add_parser(f"absa-v1-batch-{scope}-create")
        submit = sub.add_parser(f"absa-v1-batch-{scope}-submit")
        submit.add_argument("--run-api", action="store_true")
        status = sub.add_parser(f"absa-v1-batch-{scope}-status")
        status.add_argument("--run-api", action="store_true")
        imp = sub.add_parser(f"absa-v1-batch-{scope}-import")
        imp.add_argument("--run-api", action="store_true")
        sub.add_parser(f"absa-v1-batch-{scope}-finalize")
    args = parser.parse_args()
    pipeline = Pipeline()

    if args.command == "download-language-model":
        print(f"Downloaded OpenLID-v3 to {pipeline.download_language_model()}")
    elif args.command == "prepare":
        documents, units = pipeline.prepare()
        print(f"Prepared {len(documents):,} documents and {len(units):,} analysis units.")
    elif args.command == "validation-sample":
        documents = read_table(pipeline.settings.output_dir / "documents.parquet")
        output = make_validation_sample(documents, args.size, pipeline.settings.random_seed)
        write_table(output, pipeline.settings.output_dir / "validation_sample.parquet")
        print(f"Created validation sample with {len(output):,} records.")
    elif args.command == "relevance-review-sample":
        output = pipeline.relevance_review_sample(args.size, args.repeats)
        print(f"Created blind relevance review file with {len(output):,} rows.")
    elif args.command == "relevance-fresh-holdout":
        output = pipeline.relevance_fresh_holdout(args.size, args.repeats)
        print(f"Created fresh blind holdout with {len(output):,} reviewer rows.")
    elif args.command.startswith("relevance-v7-") and args.command.endswith("-sample"):
        stage = args.command.removeprefix("relevance-v7-").removesuffix("-sample")
        stage = stage.replace("-", "_")
        output = pipeline.relevance_v7_sample(stage, args.size, args.repeats)
        print(f"Created v7 {stage} blind sample with {len(output):,} reviewer rows.")
    elif args.command == "relevance-v7-finalize":
        print(pipeline.relevance_v7_finalize(args.stage))
    elif args.command == "relevance-v7-repeat-audit":
        output = pipeline.relevance_v7_repeat_audit(args.stage)
        print(f"Created repeat audit with {len(output):,} disagreements.")
    elif args.command == "relevance-v7-score-calibration":
        if not args.run_api:
            parser.error("v7 calibration scoring uses paid OpenAI API calls; pass --run-api explicitly")
        print(pipeline.relevance_v7_score_calibration())
    elif args.command == "relevance-v8-score-calibration":
        if not args.run_api:
            parser.error("v8 calibration scoring uses paid OpenAI API calls; pass --run-api explicitly")
        print(pipeline.relevance_v8_score_calibration())
    elif args.command == "relevance-v8-production":
        if not args.run_api:
            parser.error("v8 production classification uses paid OpenAI API calls; pass --run-api explicitly")
        output = run_v8_production(pipeline.settings)
        print(f"Created frozen v8_op1 production decisions for {len(output):,} records.")
    elif args.command == "relevance-v8-cost-estimate":
        report = estimate_v8_production_cost(pipeline.settings)
        population, estimates = report["population"], report["estimates"]
        print(f"Production population: {population['eligible']:,}")
        print(f"Deterministic exclusions: {population['deterministic_exclusions']:,}")
        print(f"Reusable cached initial results: {population['reusable_initial']:,}")
        print(f"New initial calls required: {population['new_initial']:,}")
        print(f"Expected verification calls: {population['estimated_remaining_verification']:,}")
        for label, key in [("Expected", "expected_remaining"), ("Conservative", "conservative_p95"), ("Worst practical", "worst_practical_all_remaining_verify_p95"), ("No-cache", "no_cache_expected")]:
            value = estimates[key]
            print(f"{label}: input {value['input_tokens']:,}, output {value['output_tokens']:,}, ${value['cost_usd']:.2f}")
        print(f"Estimated cache savings: ${estimates['estimated_cache_savings_usd']:.2f}")
        variant = estimates["cost_conservative_expected"]
        print(f"Cost-conservative: {population['cost_conservative_estimated_remaining_terra']:,} Terra calls, ${variant['cost_usd']:.2f} synchronous, ${estimates['cost_conservative_batch_expected']['cost_usd']:.2f} Batch")
        print(f"Projected remaining manual review: v8_op1 {population['current_projected_remaining_manual_review']:,}; cost-conservative {population['cost_conservative_projected_remaining_manual_review']:,}")
        print("No API call occurred.")
    elif args.command.startswith("relevance-v8-adjudication-"):
        package_dir = pipeline.settings.output_dir / "relevance_v8_adjudication"
        candidates = pipeline.settings.output_dir / "relevance_v8_calibration_results" / "relevance_v8_adjudication_candidates.csv"
        if args.command == "relevance-v8-adjudication-init":
            print(initialize_adjudication_package(candidates, package_dir, args.seed))
        elif args.command == "relevance-v8-adjudication-import":
            print(import_reviewer_file(args.file, args.reviewer_id, args.reviewer_version, package_dir))
        elif args.command == "relevance-v8-adjudication-agreement":
            print(analyze_agreement(args.reviewer_a, args.reviewer_b, candidates, package_dir / "agreement"))
        else:
            print(finalize_gold_standard(args.reviewer_a, args.reviewer_b, candidates, args.adjudication_report, package_dir / "gold", args.gold_version))
    elif args.command == "relevance-policy-alignment-sample":
        output = pipeline.relevance_policy_alignment_sample()
        print(f"Created blind policy-alignment file with {len(output):,} records.")
    elif args.command == "relevance-policy-alignment-finalize":
        output = pipeline.relevance_policy_alignment_finalize()
        print(f"Policy alignment finalized; {len(output):,} disagreements require adjudication.")
    elif args.command == "relevance-finalize":
        output = pipeline.relevance_finalize()
        print(f"Finalized relevance decisions; {output.review_status.eq('pending').sum():,} remain pending.")
    elif args.command == "relevance-validate":
        print(pipeline.relevance_validate())
    elif args.command == "relevance-rescore-validation":
        if not args.run_api:
            parser.error("relevance-rescore-validation uses paid OpenAI API calls; pass --run-api explicitly")
        print(pipeline.relevance_rescore_validation(args.split))
    elif args.command == "relevance-rescore-fresh-holdout":
        if not args.run_api:
            parser.error("relevance-rescore-fresh-holdout uses paid OpenAI API calls; pass --run-api explicitly")
        print(pipeline.relevance_rescore_fresh_holdout())
    elif args.command == "relevance-v8-single-audit-create":
        print(write_audit_package(args.decisions, args.output_dir, args.per_decision, args.seed))
    elif args.command == "relevance-v8-single-audit-finalize":
        print(finalize_single_audit(args.audit_file, args.key_file, args.output_dir))
    elif args.command == "relevance-v8-architecture-audit-create":
        print(write_architecture_audit_package(args.decisions, args.output_dir, args.include_sample, args.deterministic_exclude_sample, args.seed))
    elif args.command == "relevance-v8-architecture-audit-finalize":
        print(finalize_architecture_audit(args.audit_file, args.key_file, args.output_dir))
    elif args.command == "relevance-v8-cost-conservative-replay":
        print(replay_calibration(pipeline.settings))
    elif args.command == "relevance-v8-cost-conservative-report":
        print(write_production_report(pipeline.settings))
    elif args.command.startswith("relevance-v8-batch-"):
        stage = "initial" if "-initial-" in args.command else "verification"
        action = args.command.rsplit("-", 1)[-1]
        if action == "create": print(create_batch(stage, pipeline.settings, args.max_estimated_tokens))
        elif action == "submit":
            if not args.run_api: parser.error("Batch submission requires --run-api")
            batch_root = pipeline.settings.output_dir / "relevance_v8_single_researcher_cost_conservative" / "batch" / stage
            request_count = sum(1 for line in (batch_root / args.chunk).read_text(encoding="utf-8").split("\n") if line.strip())
            print(f"Submitting {request_count:,} {stage} requests from {args.chunk}. Review the package manifest cost estimate before continuing.")
            print(submit_batch(stage, args.chunk, pipeline.settings))
        elif action == "status": print(batch_status(stage, args.chunk, pipeline.settings))
        else: print(import_batch(stage, args.chunk, pipeline.settings, allow_partial=args.allow_partial))
    elif args.command == "relevance-v8-cost-conservative-finalize":
        print(finalize_production(pipeline.settings))
    elif args.command == "relevance-v8-cost-conservative-topic-gate":
        print(finalize_topic_gate(pipeline.settings))
    elif args.command == "relevance-v8-architecture-disagreement-audit":
        print(create_disagreement_package(pipeline.settings))
    elif args.command == "topic-discovery-prepare":
        print(json.dumps(prepare_topic_corpus(), indent=2))
    elif args.command == "topic-discovery-run":
        run_topic_discovery()
        print("Completed local BERTopic candidate comparison; no paid API was used.")
    elif args.command == "topic-discovery-review-prepare":
        print(json.dumps(prepare_c1_topic_review(), indent=2))
    elif args.command == "topic-discovery-review-summary":
        print(json.dumps(topic_review_summary(), indent=2))
    elif args.command == "topic-discovery-refine-c1":
        print(json.dumps(run_c1_refined(), indent=2))
    elif args.command == "topic-discovery-model-selection-prepare":
        print(json.dumps(prepare_model_selection_review(), indent=2))
    elif args.command == "topic-discovery-model-selection-summary":
        print(json.dumps(write_model_selection_summary(), indent=2))
    elif args.command == "topic-taxonomy-prepare":
        print(json.dumps(prepare_final_taxonomy(), indent=2))
    elif args.command == "topic-taxonomy-finalize":
        print(json.dumps(finalize_taxonomy(), indent=2))
    elif args.command == "absa-v1-audit-create":
        output = create_single_researcher_audit()
        print(f"Created frozen blind ABSA single-researcher audit with {len(output):,} documents.")
    elif args.command == "absa-v1-freeze-gold":
        print(freeze_gold())
    elif args.command == "absa-v1-cost-estimate":
        print(json.dumps(absa_v1_cost_estimate(), indent=2))
        print("No API call occurred.")
    elif args.command == "absa-v1-validation-evaluate":
        print(evaluate_validation())
        print("No API call occurred; existing frozen gold and predictions were evaluated locally.")
    elif args.command == "absa-v1-fp-diagnostic-create":
        result = create_fp_diagnostic()
        for name in result["artifacts"]: print(f"{result['protocol_name']}: {name}")
        baseline = json.loads((Path("data/processed/absa_v1/development") / result["protocol_name"] /
                               "absa_v1_fp_diagnostic_summary_v1.json").read_text())["baseline_reconciliation"]
        print(f"Frozen evaluation reconciliation\nTP: {baseline['tp']}\nFP: {baseline['fp']}\nFN: {baseline['fn']}\n"
              f"Predicted mentions: {baseline['predicted_mentions']}\nGold mentions: {baseline['gold_mentions']}")
        print(f"Development FP audit sample\nSelected FP mentions: {result['manual_fp_sample_size']}\nSeed: {result['manual_sample_seed']}")
        print("No prompt/model/ontology/gold changes were made.\nNo API calls were performed.")
    elif args.command == "absa-v1-fp-diagnostic-finalize":
        result = finalize_fp_diagnostic()
        print(json.dumps({"reviewed_fp_rows": result["reviewed_fp_rows"],
                          "dominant_error_mechanisms": result["dominant_error_mechanisms"]}, indent=2))
        print("No prompt was generated. No API calls were performed.")
    elif args.command == "absa-v1-precision-development-create":
        result = create_precision_experiment()
        print(json.dumps(result, indent=2))
        print("Development evidence: true; confirmatory evidence: false.")
        print("Candidate Batch requests were created locally. No API call occurred.")
    elif args.command == "absa-v1-precision-development-batch-submit":
        if not args.run_api:
            parser.error("Candidate Batch submission is a paid API action; pass --run-api explicitly")
        print(json.dumps(submit_precision_experiment(), indent=2))
    elif args.command == "absa-v1-precision-development-batch-status":
        if not args.run_api:
            parser.error("Candidate Batch status uses the OpenAI API; pass --run-api explicitly")
        print(json.dumps(precision_experiment_status(), indent=2))
    elif args.command == "absa-v1-precision-development-batch-import":
        if not args.run_api:
            parser.error("Candidate Batch import uses the OpenAI API; pass --run-api explicitly")
        print(import_precision_experiment())
    elif args.command == "absa-v1-precision-development-finalize":
        print(finalize_precision_experiment())
        print("Development evidence: true; confirmatory evidence: false. No API call occurred.")
    elif args.command == "absa-v1-precision-residual-diagnostic":
        result = create_residual_diagnostic()
        print(json.dumps({"fp_reduction": result["fp_reduction"],
                          "diagnostic_conclusion": result["diagnostic_conclusion"],
                          "artifacts": result["artifacts"]}, indent=2))
        print("Development evidence: true; confirmatory evidence: false. No API call or inference occurred.")
    elif args.command == "absa-v1-aspect-ownership-development-create":
        result = create_ownership_experiment()
        print(json.dumps(result, indent=2))
        print("Development evidence: true; confirmatory evidence: false. V3 requests were created locally; no API call occurred.")
    elif args.command == "absa-v1-aspect-ownership-development-batch-submit":
        if not args.run_api:
            parser.error("V3 Batch submission is a paid API action; pass --run-api explicitly")
        print(json.dumps(submit_ownership_experiment(), indent=2))
    elif args.command == "absa-v1-aspect-ownership-development-batch-status":
        if not args.run_api:
            parser.error("V3 Batch status uses the OpenAI API; pass --run-api explicitly")
        print(json.dumps(ownership_experiment_status(), indent=2))
    elif args.command == "absa-v1-aspect-ownership-development-batch-import":
        if not args.run_api:
            parser.error("V3 Batch import uses the OpenAI API; pass --run-api explicitly")
        print(import_ownership_experiment())
    elif args.command == "absa-v1-aspect-ownership-development-finalize":
        print(finalize_ownership_experiment())
        print("Development evidence: true; confirmatory evidence: false. No API call occurred; no V4 was created.")
    elif args.command == "local-absa-gpu-check":
        result = local_gpu_preflight(require_cuda=False)
        print(format_local_gpu_preflight(result))
        if not result["passed"]:
            raise SystemExit(2)
    elif args.command in {"local-absa-audit", "local-absa-baseline-create"}:
        result = create_local_absa_audit()
        print(json.dumps(result, indent=2))
        print("Local audit prepared. No neural execution or API call occurred.")
    elif args.command == "local-absa-baseline-train":
        local_absa_baseline_train()
    elif args.command == "local-absa-baseline-evaluate":
        print(json.dumps(local_absa_baseline_evaluate(), indent=2))
    elif args.command == "blog-analysis-audit":
        print(json.dumps(audit_blog_source(), indent=2)); print("Read-only source audit complete. API calls = 0.")
    elif args.command == "blog-integrity-check":
        print(json.dumps(verify_frozen_upstream(), indent=2)); print("Frozen upstream quantities and hashes verified. API calls = 0.")
    elif args.command == "blog-absa-create":
        print(json.dumps(create_blog_absa_requests(), indent=2)); print("Blog ABSA request package prepared locally; submitted = false; API calls = 0.")
    elif args.command == "blog-absa-submit":
        if not args.run_api: parser.error("Blog ABSA Batch submission is a paid API action; pass --run-api explicitly")
        print(json.dumps(submit_blog_absa(), indent=2))
    elif args.command == "blog-absa-status":
        if not args.run_api: parser.error("Blog ABSA Batch status uses the OpenAI API; pass --run-api explicitly")
        print(json.dumps(blog_absa_status(), indent=2))
    elif args.command == "blog-absa-import":
        if not args.run_api: parser.error("Blog ABSA Batch import uses the OpenAI API; pass --run-api explicitly")
        print(json.dumps(import_blog_absa(), indent=2))
    elif args.command == "blog-absa-finalize":
        print(json.dumps(finalize_blog_absa(), indent=2)); print("Offline fail-closed finalization complete. API calls = 0.")
    elif args.command == "blog-theme-candidates":
        print(json.dumps(generate_theme_candidates(), indent=2)); print("Candidate similarities require researcher decisions; automatic assignments = 0.")
    elif args.command == "blog-theme-finalize":
        print(json.dumps(finalize_theme_mappings(args.mapping or Path("data/processed/blog_analysis_v1/blog_theme_mapping_candidates_v1.csv")), indent=2))
    elif args.command == "blog-emergent-theme-audit":
        from .blog_emergent_themes import audit
        print(json.dumps(audit(), indent=2))
    elif args.command == "blog-emergent-theme-finalize":
        from .blog_emergent_themes import finalize
        print(json.dumps(finalize(), indent=2))
    elif args.command == "cross-source-analysis":
        print(json.dumps(create_cross_source_analysis(), indent=2)); print("Descriptive triangulation only; pooled prevalence = false; inferential tests = 0.")
    elif args.command == "absa-v1-descriptive-analysis":
        result = create_absa_descriptive_analysis()
        print(json.dumps(result, indent=2))
        print("Offline descriptive analysis complete. API calls = 0.")
    elif args.command == "absa-v1-inferential-analysis":
        result = create_absa_inferential_analysis()
        print(json.dumps(result, indent=2))
        print("Offline inferential analysis complete. API calls = 0; ABSA inference = 0.")
    elif args.command == "absa-v1-dashboard-data":
        result = create_absa_dashboard_data()
        print(json.dumps(result, indent=2))
        print("Frozen dashboard data mart complete. API calls = 0; new statistical tests = 0.")
    elif args.command == "absa-v1-thesis-outputs":
        result = create_absa_thesis_outputs()
        print(json.dumps(result, indent=2))
        print("Thesis-ready output package complete. API calls = 0; new statistical tests = 0.")
    elif args.command == "absa-v1-aspect-level-themes":
        result = create_aspect_level_themes()
        print(json.dumps(result, indent=2))
        print("ALTA complete. Network calls = 0; API calls = 0; paid inference = 0; upstream writes = 0.")
    elif args.command == "absa-v1-alta-review-generate":
        print(json.dumps(generate_review_package(), indent=2))
        print("Researcher review package generated; finalization has not occurred. Network/API calls = 0.")
    elif args.command == "absa-v1-alta-review-validate":
        print(json.dumps(validate_review_package(args.mapping), indent=2))
    elif args.command == "absa-v1-alta-review-finalize":
        print(json.dumps(finalize_reviewed_taxonomy(args.mapping), indent=2))
        print("Reviewed taxonomy finalized from completed researcher decisions. Network/API calls = 0.")
    elif args.command == "absa-v1-production-audit":
        result = audit_absa_production()
        print(json.dumps(result, indent=2))
        print("Zero API calls. Frozen production requests were not modified.")
    elif args.command == "absa-v1-production-retry-create":
        print(create_absa_production_retry())
        print("Affected-document retry requests were copied from the frozen package. submitted = false; API calls = 0.")
    elif args.command == "absa-v1-production-create":
        result = create_absa_production()
        print(json.dumps(result, indent=2))
        print("Production requests were prepared locally. submitted = false. No API call occurred.")
    elif args.command == "absa-v1-production-batch-submit":
        if not args.run_api:
            parser.error("Full-corpus ABSA Batch submission is a paid API action; pass --run-api explicitly")
        print(json.dumps(submit_absa_production(), indent=2))
    elif args.command == "absa-v1-production-batch-status":
        if not args.run_api:
            parser.error("Production Batch status uses the OpenAI API; pass --run-api explicitly")
        print(json.dumps(absa_production_status(), indent=2))
    elif args.command == "absa-v1-production-batch-import":
        if not args.run_api:
            parser.error("Production Batch import uses the OpenAI API; pass --run-api explicitly")
        print(json.dumps(import_absa_production(), indent=2))
    elif args.command == "absa-v1-production-finalize":
        print(finalize_absa_production())
        print("Production outputs are descriptive inference artifacts, not validation evidence. No API call occurred.")
    elif args.command.startswith("absa-v1-batch-") and args.command.endswith("-create"):
        scope = "validation" if "-validation-" in args.command else "full"
        print(create_batch_requests(scope))
    elif args.command.startswith("absa-v1-batch-") and args.command.endswith("-submit"):
        if not args.run_api:
            parser.error("ABSA Batch submission uses paid OpenAI API calls; pass --run-api explicitly")
        scope = "validation" if "-validation-" in args.command else "full"
        print(json.dumps(submit_absa_batch(scope), indent=2))
    elif args.command.startswith("absa-v1-batch-") and args.command.endswith("-status"):
        if not args.run_api: parser.error("ABSA Batch status uses the OpenAI API; pass --run-api explicitly")
        scope = "validation" if "-validation-" in args.command else "full"
        print(json.dumps(absa_batch_status(scope), indent=2))
    elif args.command.startswith("absa-v1-batch-") and args.command.endswith("-import"):
        if not args.run_api: parser.error("ABSA Batch import uses the OpenAI API; pass --run-api explicitly")
        scope = "validation" if "-validation-" in args.command else "full"
        print(import_absa_batch(scope))
    elif args.command.startswith("absa-v1-batch-") and args.command.endswith("-finalize"):
        scope = "validation" if "-validation-" in args.command else "full"
        print(finalize_absa_batch(scope))
    else:
        if not args.run_api:
            parser.error(f"{args.command} uses paid OpenAI API calls; pass --run-api explicitly")
        method = args.command.replace("-", "_")
        if method == "language_review":
            result = pipeline.language_review(args.limit, args.source)
        elif method == "relevance":
            result = pipeline.relevance(args.limit, args.pilot_size)
        else:
            result = getattr(pipeline, method)(args.limit)
        size = len(result[0]) if isinstance(result, tuple) else len(result)
        print(f"Completed {args.command} for {size:,} output records.")

if __name__ == "__main__":
    main()




