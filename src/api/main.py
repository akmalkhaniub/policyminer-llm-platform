"""
FastAPI Application Server for PolicyMiner LLM Platform (v2.0).
Exposes REST API endpoints for:
- Agentic Binder-vs-Policy Reconciliation & SOV Visualization (`/api/v1/reconcile`)
- Live Policy Document Upload & Multi-Page SOV Table Stitcher (`/api/v1/documents/upload-and-audit`)
- Multi-Backend Hybrid RAG Search (`/api/v1/rag/search`)
- Unsloth QLoRA SFT + DPO Alignment & Long-Context Benchmarks (`/api/v1/finetuning/simulate`)
- Azure FinOps Cost Optimizer & Free-GPU Dispatcher (`/api/v1/cloud/finops-and-gpu`)
- MLflow Model Performance & Drift Monitoring (`/api/v1/mlops/evaluate`)
- 4-Gate Verification Suite (`/api/v1/evals/run`)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from src.agents.langsmith_workflow import LangSmithInsuranceMiningWorkflow
from src.agents.tower_analyzer import (
    analyze_insurance_program_tower,
    export_reconciliation_csv,
)
from src.core.schemas import BaseLLMArchitecture, UnslothQLoRAConfig, VectorBackendType
from src.core.security_guard import sanitize_insurance_document
from src.evals.runner import run_all_evaluation_gates
from src.finetuning.azure_finops_dispatcher import (
    compute_azure_finops_comparison,
    get_free_gpu_dispatch_targets,
)
from src.finetuning.long_context_architectures import (
    benchmark_long_context_architectures,
    evaluate_needle_in_policy_haystack,
)
from src.finetuning.mlflow_evaluator import evaluate_and_check_retraining_gate
from src.finetuning.unsloth_trainer import (
    build_insurance_dpo_preference_dataset,
    build_insurance_sft_dataset,
    run_unsloth_dpo_alignment,
    run_unsloth_qlora_experiment,
)
from src.rag.embeddings import expand_insurance_query
from src.rag.table_stitcher import (
    parse_and_audit_custom_policy_text,
    stitch_multipage_insurance_tables,
)
from src.rag.vector_router import UnifiedInsuranceVectorRouter

app = FastAPI(
    title="PolicyMiner — Commercial Insurance Mining & LLM Engineering Platform",
    version="2.0.0",
    description=(
        "Commercial P&C Binder/Policy/SOV Extraction, Unsloth QLoRA + DPO, "
        "Multi-Page Table Stitcher, Multi-Vector Hybrid RAG, LangSmith, Azure FinOps & Free-GPU Dispatcher"
    ),
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


class CustomPolicyAuditRequest(BaseModel):
    raw_policy_text: str = (
        "ENDORSEMENT CP-14-20: Cyber Extortion Sublimit is amended to $2,500,000 subject to "
        "a $250,000 Self-Insured Retention (SIR). ENDORSEMENT FL-10: Restore Flood Zone A coverage."
    )


class FinOpsRequest(BaseModel):
    monthly_policies_processed: int = 25000
    avg_tokens_per_policy: int = 42000


class TowerSimulationRequest(BaseModel):
    simulated_loss_usd: float = 10_000_000.0
    coinsurance_pct: float = 90.0
    carried_limit_usd: float = 35_000_000.0


class SanitizeDocumentRequest(BaseModel):
    raw_text: str


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "healthy",
        "service": "policyminer-llm-platform",
        "version": "3.0.0",
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
    out = workflow.run_pipeline(query=query, vector_backend=backend)
    out["stitched_multipage_sov"] = stitch_multipage_insurance_tables().model_dump()
    return out


@app.post("/api/v1/documents/upload-and-audit")
def upload_and_audit_policy(req: CustomPolicyAuditRequest) -> dict:
    sanitized = sanitize_insurance_document(req.raw_policy_text)
    audit = parse_and_audit_custom_policy_text(sanitized.sanitized_text)
    audit["security_guard"] = sanitized.model_dump()
    return audit


@app.post("/api/v1/documents/upload-file")
async def upload_policy_file(file: UploadFile = File(...)) -> dict:
    content_bytes = await file.read()
    filename = file.filename or "uploaded_policy.txt"
    if filename.lower().endswith(".pdf"):
        try:
            import io
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            raw_text = "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"PDF parsing failed: {exc}")
    else:
        raw_text = content_bytes.decode("utf-8", errors="ignore")

    sanitized = sanitize_insurance_document(raw_text)
    audit = parse_and_audit_custom_policy_text(sanitized.sanitized_text)
    audit["filename"] = filename
    audit["security_guard"] = sanitized.model_dump()
    return audit


@app.post("/api/v1/security/sanitize")
def sanitize_document_endpoint(req: SanitizeDocumentRequest) -> dict:
    return sanitize_insurance_document(req.raw_text).model_dump()


@app.post("/api/v1/tower/analyze")
def analyze_tower_endpoint(req: TowerSimulationRequest) -> dict:
    return analyze_insurance_program_tower(
        simulated_loss_usd=req.simulated_loss_usd,
        coinsurance_pct=req.coinsurance_pct,
        carried_limit_usd=req.carried_limit_usd,
    ).model_dump()


@app.get("/api/v1/export/reconciliation-csv", response_class=PlainTextResponse)
def export_reconciliation_csv_endpoint() -> PlainTextResponse:
    csv_body = export_reconciliation_csv()
    return PlainTextResponse(
        content=csv_body,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=lineslip_reconciliation_report.csv"},
    )


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
    dpo_summary = run_unsloth_dpo_alignment(base_sft_adapter=summary.run_id)
    long_context = [r.model_dump() for r in benchmark_long_context_architectures()]
    needle_comparison = {
        "naive_rag": evaluate_needle_in_policy_haystack(strategy="naive_fixed_chunk_rag"),
        "hierarchical_long_context": evaluate_needle_in_policy_haystack(
            strategy="long_context_hierarchical"
        ),
    }
    return {
        "qlora_summary": summary.model_dump(),
        "dpo_summary": dpo_summary.model_dump(),
        "long_context_benchmarks": long_context,
        "needle_in_haystack_comparison": needle_comparison,
        "sample_sft_records": build_insurance_sft_dataset(),
        "sample_dpo_pairs": build_insurance_dpo_preference_dataset(),
    }


@app.post("/api/v1/cloud/finops-and-gpu")
def get_finops_and_free_gpu(req: FinOpsRequest) -> dict:
    finops = compute_azure_finops_comparison(
        monthly_policies_processed=req.monthly_policies_processed,
        avg_tokens_per_policy=req.avg_tokens_per_policy,
    )
    gpu_targets = [t.model_dump() for t in get_free_gpu_dispatch_targets()]
    return {
        "finops_report": finops.model_dump(),
        "free_gpu_targets": gpu_targets,
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
