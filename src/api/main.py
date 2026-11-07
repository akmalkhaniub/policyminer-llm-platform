"""
FastAPI Application Server for PolicyMiner LLM Platform.
Exposes REST API endpoints for:
- Agentic Binder-vs-Policy Reconciliation & SOV Visualization (`/api/v1/reconcile`)
- Multi-Backend Hybrid RAG Search (`/api/v1/rag/search`)
- Unsloth QLoRA Fine-Tuning & Long-Context Benchmarks (`/api/v1/finetuning/simulate`)
- MLflow Model Performance & Drift Monitoring (`/api/v1/mlops/evaluate`)
- 4-Gate Verification Suite (`/api/v1/evals/run`)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from src.agents.langsmith_workflow import LangSmithInsuranceMiningWorkflow
from src.core.schemas import BaseLLMArchitecture, UnslothQLoRAConfig, VectorBackendType
from src.evals.runner import run_all_evaluation_gates
from src.finetuning.long_context_architectures import (
    benchmark_long_context_architectures,
    evaluate_needle_in_policy_haystack,
)
from src.finetuning.mlflow_evaluator import evaluate_and_check_retraining_gate
from src.finetuning.unsloth_trainer import (
    build_insurance_sft_dataset,
    run_unsloth_qlora_experiment,
)
from src.rag.embeddings import expand_insurance_query
from src.rag.vector_router import UnifiedInsuranceVectorRouter

app = FastAPI(
    title="PolicyMiner — Commercial Insurance Mining & LLM Engineering Platform",
    version="1.0.0",
    description="Commercial P&C Binder/Policy/SOV Extraction, Unsloth QLoRA, Multi-Vector Hybrid RAG, LangSmith & Azure MLOps",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

workflow = LangSmithInsuranceMiningWorkflow()
rag_router = UnifiedInsuranceVectorRouter()
STATIC_DIR = Path(__file__).parent / "static"


class RAGSearchRequest(BaseModel):
    query: str = "What endorsements reduced the Cyber Extortion sublimit or excluded Flood Zone A?"
    backend: VectorBackendType = VectorBackendType.AZURE_COGNITIVE_SEARCH
    top_k: int = 4


class DriftSimulationRequest(BaseModel):
    simulate_degradation: bool = False


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "healthy",
        "service": "policyminer-llm-platform",
        "supported_vector_backends": [b.value for b in VectorBackendType],
        "supported_base_models": [m.value for m in BaseLLMArchitecture],
    }


@app.get("/api/v1/reconcile")
def get_reconciliation_workspace(
    query: str = Query(
        default="What endorsements reduced the Cyber Extortion sublimit or excluded Flood Zone A?"
    ),
    backend: VectorBackendType = Query(default=VectorBackendType.AZURE_COGNITIVE_SEARCH),
) -> dict:
    return workflow.run_pipeline(query=query, vector_backend=backend)


@app.post("/api/v1/rag/search")
def search_policy_corpus(req: RAGSearchRequest) -> dict:
    expanded = expand_insurance_query(req.query)
    chunks = rag_router.search(query=req.query, backend=req.backend, top_k=req.top_k)
    return {
        "raw_query": req.query,
        "expanded_query": expanded,
        "backend_used": req.backend.value,
        "results": [c.model_dump() for c in chunks],
    }


@app.post("/api/v1/finetuning/simulate")
def simulate_unsloth_finetuning(config: Optional[UnslothQLoRAConfig] = None) -> dict:
    cfg = config or UnslothQLoRAConfig()
    summary = run_unsloth_qlora_experiment(cfg)
    long_context = [r.model_dump() for r in benchmark_long_context_architectures()]
    needle_comparison = {
        "naive_rag": evaluate_needle_in_policy_haystack(strategy="naive_fixed_chunk_rag"),
        "hierarchical_long_context": evaluate_needle_in_policy_haystack(
            strategy="long_context_hierarchical"
        ),
    }
    return {
        "qlora_summary": summary.model_dump(),
        "long_context_benchmarks": long_context,
        "needle_in_haystack_comparison": needle_comparison,
        "sample_sft_records": build_insurance_sft_dataset(),
    }


@app.post("/api/v1/mlops/evaluate")
def evaluate_mlops_gate(req: DriftSimulationRequest) -> dict:
    pipeline_out = workflow.run_pipeline()
    gt = pipeline_out["policy"]["coverages"]
    if req.simulate_degradation:
        metrics = evaluate_and_check_retraining_gate(
            predicted_entities=gt[:2],
            ground_truth_entities=gt,
            cross_entropy_loss=2.05,
            ttft_ms=84.0,
            tpot_ms=26.5,
            expected_doc_dist=[0.35, 0.30, 0.20, 0.15],
            actual_doc_dist=[0.12, 0.18, 0.42, 0.28],
            model_version="qwen2.5-14b-qlora-drifted",
        )
    else:
        metrics = evaluate_and_check_retraining_gate(
            predicted_entities=gt,
            ground_truth_entities=gt,
            cross_entropy_loss=1.02,
            ttft_ms=36.4,
            tpot_ms=10.8,
            model_version="qwen2.5-14b-qlora-v2.1",
        )
    return metrics.model_dump()


@app.get("/api/v1/evals/run")
def execute_eval_gates() -> dict:
    return run_all_evaluation_gates()


@app.get("/", response_class=HTMLResponse)
def serve_workbench() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
