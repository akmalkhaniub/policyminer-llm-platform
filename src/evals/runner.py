"""
Layer 4 SDD: 4-Gate Automated Verification & Regression Evaluation Runner.

Governing Requirements: REQ-FT-01, REQ-FT-02, REQ-RAG-01, REQ-RAG-02, REQ-AG-01, REQ-AG-02, REQ-MLOPS-01
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

from src.agents.langsmith_workflow import LangSmithInsuranceMiningWorkflow
from src.core.schemas import UnslothQLoRAConfig, VectorBackendType
from src.finetuning.long_context_architectures import (
    benchmark_long_context_architectures,
    evaluate_needle_in_policy_haystack,
)
from src.finetuning.mlflow_evaluator import evaluate_and_check_retraining_gate
from src.finetuning.unsloth_trainer import run_unsloth_qlora_experiment
from src.rag.vector_router import UnifiedInsuranceVectorRouter


def run_all_evaluation_gates() -> Dict[str, Any]:
    gates: List[Dict[str, Any]] = []

    # Gate 1: Unsloth QLoRA & VRAM Optimization (INV-03, REQ-FT-01)
    qlora_summary = run_unsloth_qlora_experiment(UnslothQLoRAConfig())
    gate1_passed = (
        qlora_summary.vram_savings_pct >= 60.0
        and qlora_summary.finetuned_perplexity < qlora_summary.baseline_perplexity
    )
    gates.append(
        {
            "gate": "Gate 1: Unsloth 4-bit QLoRA & Perplexity Optimization (REQ-FT-01)",
            "passed": gate1_passed,
            "metrics": {
                "vram_savings_pct": qlora_summary.vram_savings_pct,
                "baseline_ppl": qlora_summary.baseline_perplexity,
                "finetuned_ppl": qlora_summary.finetuned_perplexity,
            },
        }
    )

    # Gate 2: Long-Context Needle-in-a-Policy-Haystack (INV-02, REQ-FT-02)
    arch_reports = benchmark_long_context_architectures()
    needle_eval = evaluate_needle_in_policy_haystack(strategy="long_context_hierarchical")
    gate2_passed = (
        len(arch_reports) == 3
        and needle_eval["endorsement_override_detected"] is True
        and needle_eval["extracted_flood_limit_usd"] == 500_000.0
    )
    gates.append(
        {
            "gate": "Gate 2: Long-Context 128k & Endorsement Precedence (REQ-FT-02)",
            "passed": gate2_passed,
            "metrics": {
                "architectures_benchmarked": len(arch_reports),
                "needle_accuracy": needle_eval["accuracy"],
            },
        }
    )

    # Gate 3: Multi-Backend Hybrid RAG Router (REQ-RAG-01, REQ-RAG-02)
    router = UnifiedInsuranceVectorRouter()
    backends_ok = True
    for backend in VectorBackendType:
        hits = router.search("Cyber Extortion SIR sublimit endorsement", backend=backend, top_k=3)
        if not hits or not any("CP-14-20" in h.section_title for h in hits):
            backends_ok = False
    gates.append(
        {
            "gate": "Gate 3: Multi-Vector Hybrid RAG (FAISS, Azure AI Search, Pinecone, Weaviate)",
            "passed": backends_ok,
            "metrics": {"backends_verified": [b.value for b in VectorBackendType]},
        }
    )

    # Gate 4: LangSmith Agentic Reconciliation & MLflow Retraining Gate (REQ-AG-01, REQ-MLOPS-01)
    workflow = LangSmithInsuranceMiningWorkflow()
    pipeline_out = workflow.run_pipeline()
    rec = pipeline_out["reconciliation"]

    gt_entities = pipeline_out["policy"]["coverages"]
    mlflow_healthy = evaluate_and_check_retraining_gate(
        predicted_entities=gt_entities,
        ground_truth_entities=gt_entities,
        cross_entropy_loss=1.05,
    )
    mlflow_degraded = evaluate_and_check_retraining_gate(
        predicted_entities=gt_entities[:2],
        ground_truth_entities=gt_entities,
        cross_entropy_loss=2.10,
    )
    gate4_passed = (
        len(rec["discrepancies"]) == 5
        and mlflow_healthy.f1_score == 1.0
        and mlflow_healthy.retraining_triggered is False
        and mlflow_degraded.retraining_triggered is True
    )
    gates.append(
        {
            "gate": "Gate 4: LangSmith Binder-vs-Policy Reconciliation & MLflow Auto-Retrain Gate",
            "passed": gate4_passed,
            "metrics": {
                "discrepancies_detected": len(rec["discrepancies"]),
                "healthy_f1": mlflow_healthy.f1_score,
                "degraded_retrain_job": mlflow_degraded.aml_pipeline_job_id,
            },
        }
    )

    all_passed = all(g["passed"] for g in gates)
    return {"all_passed": all_passed, "gates": gates}


if __name__ == "__main__":
    report = run_all_evaluation_gates()
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["all_passed"] else 1)
