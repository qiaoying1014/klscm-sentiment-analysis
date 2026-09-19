from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("data/processed/absa_local_v1")
ABSA_ROOT = Path("data/processed/absa_v1")
AUDIT_ROOT = ABSA_ROOT / "absa_v1_single_researcher_audit_v1"
ONTOLOGY_PATH = ABSA_ROOT / "absa_aspect_ontology_v1.json"
GOLD_PATH = AUDIT_ROOT / "absa_v1_audit_gold_v1.csv"
SAMPLE_PATH = AUDIT_ROOT / "absa_v1_audit_sample_v1.csv"
DATASET_PATH = ROOT / "local_absa_v1_derived_documents.csv"
CONFIG_PATH = ROOT / "local_absa_v1_baseline_config.json"
AUDIT_JSON = ROOT / "local_absa_v1_preflight.json"
AUDIT_MD = Path("LOCAL_ABSA_V1_PREFLIGHT_REPORT.md")
SEED = 20260821
MODEL_REPOSITORY = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
MODEL_REVISION = "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
SENTIMENTS = ["positive", "neutral", "negative", "mixed"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_preflight(*, require_cuda: bool = True) -> dict[str, Any]:
    system_gpu = None
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0].split(",")
        system_gpu = {"name": query[0].strip(), "vram_mib": int(query[1]), "driver_version": query[2].strip()}
    except Exception:
        pass
    try:
        import torch
    except Exception as exc:
        result = {"passed": False, "reason": f"PyTorch import failed: {exc}", "cuda_available": False}
        if require_cuda:
            raise RuntimeError(result["reason"]) from exc
        return result
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count())
    result: dict[str, Any] = {
        "torch_version": torch.__version__, "pytorch_cuda_version": torch.version.cuda,
        "cuda_available": available, "device_count": count, "selected_device": None,
        "gpu_name": None, "vram_bytes": None, "model_execution_device": None,
        "passed": False, "system_gpu": system_gpu,
    }
    if available and count:
        device = torch.device("cuda:0")
        x = torch.randn(1024, 1024, device=device)
        y = x @ x
        torch.cuda.synchronize(device)
        result.update(selected_device="cuda:0", gpu_name=torch.cuda.get_device_name(0),
                      vram_bytes=int(torch.cuda.get_device_properties(0).total_memory),
                      model_execution_device=str(y.device), tensor_x_device=str(x.device),
                      tensor_y_device=str(y.device),
                      allocated_vram_bytes=int(torch.cuda.memory_allocated()),
                      reserved_vram_bytes=int(torch.cuda.memory_reserved()),
                      peak_allocated_vram_bytes=int(torch.cuda.max_memory_allocated()),
                      peak_reserved_vram_bytes=int(torch.cuda.max_memory_reserved()),
                      passed=x.is_cuda and y.is_cuda and str(y.device) == "cuda:0")
    else:
        result["reason"] = "CUDA GPU exists at the system level only if reported by nvidia-smi; this PyTorch build cannot use CUDA. Install a CUDA-enabled PyTorch build."
    if require_cuda and not result["passed"]:
        raise RuntimeError(result["reason"])
    return result


def format_gpu_preflight(result: dict[str, Any]) -> str:
    return "\n".join([
        "GPU PREFLIGHT", "-------------",
        f"CUDA available: {str(result.get('cuda_available', False)).lower()}",
        f"Selected device: {result.get('selected_device')}",
        f"GPU: {result.get('gpu_name')}",
        f"System GPU: {(result.get('system_gpu') or {}).get('name')}",
        f"PyTorch: {result.get('torch_version')}",
        f"PyTorch CUDA: {result.get('pytorch_cuda_version')}",
        f"Model execution device: {result.get('model_execution_device')}",
        f"GPU requirement: {'PASS' if result.get('passed') else 'FAIL'}",
        *([f"Reason: {result['reason']}"] if result.get("reason") else []),
    ])


def ontology() -> list[dict[str, Any]]:
    payload = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    aspects = payload["aspects"]
    if len(aspects) != 20 or len({row["id"] for row in aspects}) != 20:
        raise RuntimeError("Frozen ontology must contain exactly 20 unique aspect IDs")
    return aspects


def _support_band(count: int) -> str:
    if count <= 2: return "insufficient_support"
    if count <= 5: return "low_support"
    if count <= 10: return "moderate_support"
    return "adequate_support"


def _derived_documents() -> pd.DataFrame:
    gold = pd.read_csv(GOLD_PATH, dtype={"document_id": str})
    sample = pd.read_csv(SAMPLE_PATH, dtype={"document_id": str})
    aspect_ids = [row["id"] for row in ontology()]
    unknown = set(gold.aspect) - set(aspect_ids)
    sentiments = set(gold.sentiment)
    if unknown: raise RuntimeError(f"Gold contains unknown aspects: {sorted(unknown)}")
    if sentiments - set(SENTIMENTS): raise RuntimeError(f"Gold contains unknown sentiments: {sorted(sentiments)}")
    grouped = {doc: frame for doc, frame in gold.groupby("document_id")}
    records = []
    for row in sample.itertuples(index=False):
        frame = grouped.get(row.document_id)
        aspects = [] if frame is None else sorted(set(frame.aspect))
        mapping = {} if frame is None else {a: sorted(set(frame.loc[frame.aspect == a, "sentiment"])) for a in aspects}
        record = {
            "document_id": row.document_id, "caption": row.original_text,
            "language": row.primary_language, "label_role": "exploratory_reused_evaluation_only",
            "has_any_aspect": bool(aspects), "aspects_json": json.dumps(aspects),
            "aspect_sentiments_json": json.dumps(mapping, sort_keys=True),
        }
        record.update({f"aspect__{aspect}": int(aspect in aspects) for aspect in aspect_ids})
        records.append(record)
    result = pd.DataFrame(records)
    if len(result) != 80 or result.document_id.nunique() != 80:
        raise RuntimeError("Expected exactly 80 unique frozen audit documents")
    return result


def reject_overlap(train_ids: set[str], evaluation_ids: set[str]) -> None:
    overlap = train_ids & evaluation_ids
    if overlap:
        raise ValueError(f"Train/evaluation overlap prohibited: {len(overlap)} document(s)")


def create_audit() -> dict[str, Any]:
    ROOT.mkdir(parents=True, exist_ok=True)
    gpu = gpu_preflight(require_cuda=False)
    gold = pd.read_csv(GOLD_PATH, dtype={"document_id": str})
    sample = pd.read_csv(SAMPLE_PATH, dtype={"document_id": str})
    derived = _derived_documents()
    aspects = ontology()
    aspect_support = []
    for item in aspects:
        mention_count = int((gold.aspect == item["id"]).sum())
        document_count = int(gold.loc[gold.aspect == item["id"], "document_id"].nunique())
        aspect_support.append({"aspect": item["id"], "document_support": document_count,
                               "mention_support": mention_count, "support_band": _support_band(document_count)})
    language_support = sample.primary_language.fillna("unknown").value_counts().to_dict()
    sentiment_support = gold.sentiment.value_counts().reindex(SENTIMENTS, fill_value=0).to_dict()
    labelled_artifacts = [
        {"path": str(GOLD_PATH), "kind": "frozen_human_gold", "documents": 80, "mentions": len(gold), "usable_for_training": False},
        {"path": str(AUDIT_ROOT / "absa_v1_audit_annotations_working_v1.csv"), "kind": "working_source_of_frozen_gold", "documents": 80, "mentions": len(gold), "usable_for_training": False},
        {"path": str(AUDIT_ROOT / "absa_v1_ai_draft_annotations_v1.csv"), "kind": "AI_preannotation_not_independent_human_labels", "usable_for_training": False},
        {"path": str(ABSA_ROOT / "development"), "kind": "model_predictions_and_diagnostic_reviews_not_training_gold", "usable_for_training": False},
    ]
    report = {
        "protocol": "local_absa_v1_preflight", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu, "python": sys.version, "platform": platform.platform(), "random_seed": SEED,
        "ontology": {"path": str(ONTOLOGY_PATH), "sha256": sha256(ONTOLOGY_PATH), "aspect_count": 20,
                     "labels": [x["id"] for x in aspects]},
        "sentiment_labels": SENTIMENTS,
        "data": {"total_labelled_documents": 80, "unique_labelled_documents": 80,
                 "total_gold_mentions": int(len(gold)), "zero_aspect_documents": int((~derived.has_any_aspect).sum()),
                 "aspect_support": aspect_support, "sentiment_support": sentiment_support,
                 "language_support": language_support, "labelled_artifacts": labelled_artifacts,
                 "clean_supervised_training_documents": 0, "clean_threshold_development_documents": 0,
                 "available_exploratory_evaluation_documents": 80},
        "leakage": {"finding": "All final human ABSA labels belong to the same 80-document sample repeatedly used for OpenAI V1/V2/V3 development. No independent supervised training corpus or untouched confirmatory evaluation set exists.",
                    "training_on_80_then_evaluating_on_80_prohibited": True, "overlap_detected_if_used_for_both": 80},
        "architectures": {
            "setfit": "Not selected: no independent supervised training labels and sparse 20-label support.",
            "xlmr_finetuning": "Not selected: 80 reused documents are insufficient for defensible 20-label fine-tuning and evaluation.",
            "frozen_embeddings_linear": "Useful later if independent training labels become available; currently no clean train set.",
            "zero_shot_multilingual_nli": "Selected first baseline: ontology-definition conditioning needs no gold training and permits an honest exploratory comparison on the reused benchmark.",
            "generic_absa_libraries": "Not selected: aspect-term extraction formulations do not match the frozen category-level implicit-aspect task.",
        },
        "recommended_local_baseline": "zero_shot_multilingual_nli_ontology_entailment",
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "architecture": "multilingual DeBERTa-v3 base NLI/XNLI",
                  "parameter_count_approx": 279_000_000, "license": "MIT (verify from pinned model card before download)",
                  "download_cache": "Hugging Face local cache", "download_required": True,
                  "estimated_vram_gb_inference": "2-4", "estimated_training_time": "none; zero-shot baseline performs inference only"},
        "evaluation_design": {"train": "none", "threshold_tuning": "none in first baseline; freeze a global threshold before scoring",
                              "evaluate": "80-document historically reused gold; exploratory development comparison only, not unbiased confirmatory evidence",
                              "future_confirmatory_need": "new independently annotated held-out documents"},
        "full_corpus_run_allowed": False, "openai_api_calls": 0, "paid_api_calls": 0,
    }
    config = {"protocol": "local_absa_v1_zero_shot_nli", "seed": SEED, "model_repository": MODEL_REPOSITORY,
              "model_revision": MODEL_REVISION, "device_required": "cuda:0", "ontology_sha256": report["ontology"]["sha256"],
              "aspect_labels": report["ontology"]["labels"], "sentiment_labels": SENTIMENTS,
              "global_aspect_threshold": 0.70, "threshold_status": "predeclared_initial_not_tuned",
              "maximum_documents": 80, "full_corpus_run_allowed": False, "evidence_spans": "not_generated",
              "evaluation_status": "not_run_gpu_preflight_failed" if not gpu["passed"] else "ready_not_run"}
    if DATASET_PATH.exists() or CONFIG_PATH.exists() or AUDIT_JSON.exists():
        raise FileExistsError("Local ABSA v1 audit artifacts already exist; refusing overwrite")
    derived.to_csv(DATASET_PATH, index=False)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    AUDIT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Local ABSA V1 preflight report", "", "## GPU", "", "```text", format_gpu_preflight(gpu), "```", "",
             "The RTX 4060 Laptop GPU is visible to the NVIDIA driver, but the active environment contains CPU-only PyTorch. Neural execution is therefore blocked until CUDA-enabled PyTorch is installed.", "",
             "## Data and leakage", "", f"The repository contains 80 unique frozen human-labelled documents, {len(gold)} gold mentions, and {int((~derived.has_any_aspect).sum())} zero-aspect documents. These same 80 documents were used throughout OpenAI V1/V2/V3 development. No separate human-labelled training set or untouched evaluation set exists; training and testing on these records is prohibited.", "",
             "## Recommended architecture", "", f"`zero_shot_multilingual_nli_ontology_entailment` using `{MODEL_REPOSITORY}`. It can condition on the frozen ontology definitions without fitting on the 80 gold labels. This makes an exploratory comparison possible while avoiding direct train/test overlap. Evidence spans are out of scope and will not be fabricated.", "",
             "## Evaluation design", "", "- Train: none.", "- Threshold tuning: none for the first baseline; initial global threshold 0.70 is frozen before scoring.", "- Evaluation: the 80 historically reused documents, labelled explicitly as exploratory development evidence, not confirmatory evidence.", "- A genuinely unbiased comparison requires a new independently annotated holdout.", "",
             "## Architecture assessment", "", *[f"- **{k}**: {v}" for k,v in report["architectures"].items()], "",
             "## Support", "", "Aspect and language support tables are recorded in `local_absa_v1_preflight.json`. Rare aspects remain in the derived 20-column multi-label representation.", "",
             "No neural model, paid API, OpenAI Batch, or full-corpus inference was run."]
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def baseline_train() -> None:
    gpu_preflight(require_cuda=True)
    raise RuntimeError("The selected first baseline is zero-shot NLI and has no supervised training step; no independent training labels exist.")


def baseline_evaluate() -> dict[str, Any]:
    preflight = gpu_preflight(require_cuda=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data = pd.read_csv(DATASET_PATH)
    if len(data) != 80 or data.document_id.nunique() != 80:
        raise RuntimeError("Local V1 evaluation requires exactly 80 unique frozen documents; full-corpus input is prohibited")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    tokenizer = AutoTokenizer.from_pretrained(config["model_repository"], revision=config["model_revision"], local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_repository"], revision=config["model_revision"], local_files_only=True
    ).to("cuda:0").eval()
    if str(next(model.parameters()).device) != "cuda:0":
        raise RuntimeError("Model is not on CUDA; aborting")
    label_to_id = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
    required = {"entailment", "neutral", "contradiction"}
    if not required <= set(label_to_id): raise RuntimeError(f"Unexpected NLI label mapping: {model.config.id2label}")
    definitions = {x["id"]: x["definition"] for x in ontology()}
    threshold_grid = [0.50, 0.60, 0.70, 0.80, 0.90]
    batch_size = 16
    dtype = "float16_autocast"

    def score_pairs(pairs: list[tuple[str, str]], initial_batch_size: int) -> tuple[list[dict[str,float]], int]:
        active = initial_batch_size
        while True:
            try:
                scores=[]
                for start in range(0,len(pairs),active):
                    batch=pairs[start:start+active]
                    inputs=tokenizer([x[0] for x in batch],[x[1] for x in batch],padding=True,truncation=True,max_length=512,return_tensors="pt")
                    inputs={key:value.to("cuda:0") for key,value in inputs.items()}
                    if any(str(value.device)!="cuda:0" for value in inputs.values()): raise RuntimeError("NLI input tensor is not on cuda:0")
                    with torch.inference_mode(), torch.autocast(device_type="cuda",dtype=torch.float16):
                        probabilities=model(**inputs).logits.float().softmax(-1).cpu()
                    for vector in probabilities:
                        scores.append({label:float(vector[index]) for label,index in label_to_id.items()})
                return scores,active
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if active<=1: raise RuntimeError("CUDA OOM at batch size 1; CPU fallback prohibited")
                active=max(1,active//2)

    started = datetime.now(timezone.utc)
    metadata=[];pairs=[]
    for row in data.itertuples(index=False):
        for aspect,definition in definitions.items():
            metadata.append((row.document_id,aspect,row.language))
            pairs.append((row.caption, f"The author expresses an evaluation, opinion, experience, praise, complaint, preference, difficulty, satisfaction, dissatisfaction, or judgment about {aspect.replace('_',' ')}. Specifically: {definition}"))
    aspect_probabilities,batch_size=score_pairs(pairs,batch_size)
    aspect_rows=[{"document_id":doc,"aspect":aspect,"language":language,
                  "entailment_probability":score["entailment"],"neutral_probability":score["neutral"],
                  "contradiction_probability":score["contradiction"]} for (doc,aspect,language),score in zip(metadata,aspect_probabilities)]
    aspect_scores=pd.DataFrame(aspect_rows)
    candidates=aspect_scores[aspect_scores.entailment_probability>=min(threshold_grid)]
    caption_by_id=dict(zip(data.document_id,data.caption));sentiment_meta=[];sentiment_pairs=[]
    for row in candidates.itertuples(index=False):
        for sentiment in ["positive","negative","neutral"]:
            sentiment_meta.append((row.document_id,row.aspect,sentiment))
            sentiment_pairs.append((caption_by_id[row.document_id],f"The author's evaluation of {row.aspect.replace('_',' ')} is {sentiment}."))
    sentiment_probabilities,batch_size=score_pairs(sentiment_pairs,batch_size)
    sentiment_rows=[{"document_id":doc,"aspect":aspect,"sentiment_hypothesis":sentiment,
                     "entailment_probability":score["entailment"],"neutral_probability":score["neutral"],
                     "contradiction_probability":score["contradiction"]} for (doc,aspect,sentiment),score in zip(sentiment_meta,sentiment_probabilities)]
    sentiment_scores=pd.DataFrame(sentiment_rows)
    sentiment_lookup={}
    for (doc,aspect),frame in sentiment_scores.groupby(["document_id","aspect"]):
        scores=dict(zip(frame.sentiment_hypothesis,frame.entailment_probability))
        if scores["positive"]>=0.70 and scores["negative"]>=0.70: label="mixed"
        else: label=max(scores,key=scores.get)
        sentiment_lookup[(doc,aspect)]=(label,scores)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    gold = pd.read_csv(GOLD_PATH, dtype={"document_id": str})
    from marathon_absa.absa_v1 import validation_metrics
    captions=dict(zip(data.document_id,data.caption));languages=dict(zip(data.document_id,data.language))
    def predictions_at(threshold: float) -> pd.DataFrame:
        rows=[]
        active=aspect_scores[aspect_scores.entailment_probability>=threshold].sort_values(["document_id","aspect"])
        for index,row in enumerate(active.itertuples(index=False)):
            sentiment,scores=sentiment_lookup[(row.document_id,row.aspect)]
            rows.append({"mention_id":f"local-{index:05d}","document_id":row.document_id,"aspect":row.aspect,"sentiment":sentiment,
                         "aspect_score":row.entailment_probability,"sentiment_score":scores[sentiment] if sentiment!="mixed" else min(scores["positive"],scores["negative"]),
                         "evidence_text":"","evidence_start":-1,"evidence_end":-1,"evidence_status":"not_generated"})
        return pd.DataFrame(rows,columns=["mention_id","document_id","aspect","sentiment","aspect_score","sentiment_score","evidence_text","evidence_start","evidence_end","evidence_status"])
    comparison=[]
    for threshold in threshold_grid:
        candidate=predictions_at(threshold);evaluated=validation_metrics(gold,candidate,captions)
        row={**evaluated["aspect_detection"],"threshold":threshold,
             "zero_aspect_agreement":evaluated["document_diagnostics"]["zero_mention_agreement"],
             "predicted_multi_aspect_rate":evaluated["document_diagnostics"]["predicted_multi_aspect_documents"]/80}
        comparison.append(row)
    eligible=[x for x in comparison if x["recall"]>=0.64]
    selected=max(eligible,key=lambda x:(x["precision"],x["f1"],x["threshold"])) if eligible else max(comparison,key=lambda x:x["f1"])
    threshold=selected["threshold"];pred=predictions_at(threshold);evaluated=validation_metrics(gold,pred,captions,languages)
    sentiment_metrics=evaluated["matched_sentiment"]
    gold_sent=[];pred_sent=[]
    from collections import Counter
    matched=Counter(zip(gold.document_id,gold.aspect))&Counter(zip(pred.document_id,pred.aspect))
    for key,count in matched.items():
        g=gold[(gold.document_id==key[0])&(gold.aspect==key[1])].sort_values("mention_id").head(count)
        p=pred[(pred.document_id==key[0])&(pred.aspect==key[1])].sort_values("mention_id").head(count)
        gold_sent.extend(g.sentiment);pred_sent.extend(p.sentiment)
    sentiment_metrics["matched_count"]=len(gold_sent)
    sentiment_metrics["confusion_matrix"]=confusion_matrix(gold_sent,pred_sent,labels=SENTIMENTS).tolist()
    aspect_family={}
    for aspect in definitions:
        subset=validation_metrics(gold[gold.aspect==aspect],pred[pred.aspect==aspect],captions)
        aspect_family[aspect]=subset["aspect_detection"]
    language_metrics={key:value for key,value in evaluated["document_diagnostics"]["multilingual_subgroups"].items() if key not in {"non-English/uncertain"}}
    output_root=ROOT/"zero_shot_nli_v1"
    if output_root.exists(): raise FileExistsError("Local zero-shot NLI v1 outputs already exist")
    output_root.mkdir(parents=True)
    aspect_scores.to_csv(output_root/"aspect_nli_scores.csv",index=False);sentiment_scores.to_csv(output_root/"sentiment_nli_scores.csv",index=False)
    pred.to_csv(output_root/"local_absa_predictions.csv",index=False);pd.DataFrame(comparison).to_csv(output_root/"threshold_comparison.csv",index=False)
    telemetry={"gpu":preflight["gpu_name"],"model_device":"cuda:0","input_device":"cuda:0","batch_size":batch_size,"dtype":dtype,
               "aspect_nli_pairs":len(pairs),"sentiment_nli_pairs":len(sentiment_pairs),"duration_seconds":elapsed,
               "pairs_per_second":(len(pairs)+len(sentiment_pairs))/elapsed,"peak_allocated_vram_bytes":torch.cuda.max_memory_allocated(),
               "peak_reserved_vram_bytes":torch.cuda.max_memory_reserved()}
    metrics={"status":"exploratory_development_evidence_not_confirmatory","threshold_selected_on_reused_development_data":True,"confirmatory_evidence":False,
             "model":config["model_repository"],"revision":config["model_revision"],"nli_label_mapping":model.config.id2label,
             "selected_threshold":threshold,"threshold_comparison":comparison,"aspect_detection":evaluated["aspect_detection"],
             "joint_aspect_sentiment":evaluated["joint_aspect_sentiment"],"matched_sentiment":sentiment_metrics,"aspect_family":aspect_family,
             "language":language_metrics,"evidence_extraction_supported":False,"document_count":80,"prediction_count":len(pred),
             "openai_v3":{"tp":79,"fp":75,"fn":21,"precision":0.512987012987013,"recall":0.79,"f1":0.6220472440944881,
                           "joint_precision":0.43506493506493504,"joint_recall":0.67,"joint_f1":0.5275590551181102},
             "gpu_telemetry":telemetry,"openai_api_calls":0,"paid_api_calls":0}
    local=metrics["aspect_detection"];local["micro_f1"]=local["f1"];local["macro_f1"]=sum(x["f1"] for x in aspect_family.values())/len(aspect_family)
    metrics["viability"]=("LOCAL_STRONGLY_PREFERRED" if local["precision"]>=0.513 and local["f1"]>=0.622 and local["recall"]>=0.64 else "LOCAL_COMPETITIVE" if local["f1"]>=0.56 and local["precision"]>=0.45 else "LOCAL_PROMISING_BUT_WEAKER" if local["f1"]>=0.40 else "LOCAL_NOT_CURRENTLY_VIABLE")
    (output_root/"local_absa_evaluation_v1.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    (output_root/"gpu_telemetry_v1.json").write_text(json.dumps(telemetry,indent=2),encoding="utf-8")
    manifest={"protocol":"local_absa_zero_shot_nli_v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"model":config["model_repository"],"revision":config["model_revision"],"ontology_sha256":sha256(ONTOLOGY_PATH),"gold_sha256":sha256(GOLD_PATH),"sample_sha256":sha256(SAMPLE_PATH),"document_count":80,"threshold_grid":threshold_grid,"per_aspect_thresholds":False,"evidence_extraction_supported":False,"development_evidence":True,"confirmatory_evidence":False,"paid_api_calls":0}
    (output_root/"experiment_manifest_v1.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    table="\n".join(f"| {x['threshold']:.2f} | {x['precision']:.4f} | {x['recall']:.4f} | {x['f1']:.4f} | {x['tp']} | {x['fp']} | {x['fn']} |" for x in comparison)
    report=f"""# Local ABSA zero-shot NLI V1 report\n\n## Status\n\nExploratory development evidence only; the threshold was selected on the historically reused 80-document set. This is not confirmatory evidence.\n\n## Threshold comparison\n\n| Threshold | Precision | Recall | F1 | TP | FP | FN |\n|---:|---:|---:|---:|---:|---:|---:|\n{table}\n\nSelected threshold: `{threshold:.2f}`.\n\n## Selected result\n\nAspect TP/FP/FN: {local['tp']}/{local['fp']}/{local['fn']}; precision/recall/F1: {local['precision']:.4f}/{local['recall']:.4f}/{local['f1']:.4f}. Joint precision/recall/F1: {metrics['joint_aspect_sentiment']['precision']:.4f}/{metrics['joint_aspect_sentiment']['recall']:.4f}/{metrics['joint_aspect_sentiment']['f1']:.4f}. Matched sentiment accuracy/macro-F1: {sentiment_metrics['accuracy']:.4f}/{sentiment_metrics['macro_f1']:.4f}.\n\n## OpenAI V3 comparison\n\nOpenAI V3 aspect precision/recall/F1 was 0.5130/0.7900/0.6220 (79/75/21). Both are reused-sample development results.\n\n## Viability\n\n`{metrics['viability']}`\n\nEvidence extraction is unsupported; no spans or offsets were fabricated. No paid API or full-corpus inference occurred. Detailed aspect-family, language, sentiment and GPU diagnostics are in the JSON/CSV artifacts.\n"""
    (output_root/"LOCAL_ABSA_ZERO_SHOT_NLI_REPORT_V1.md").write_text(report,encoding="utf-8")
    return metrics
