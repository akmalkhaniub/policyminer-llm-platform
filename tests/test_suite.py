"""
100% Traceable Pytest Verification Suite for PolicyMiner LLM Platform (v2.0).
Maps directly to REQ-FT-01..03, REQ-DOC-01, REQ-RAG-01..02,
REQ-AG-01..02, REQ-MLOPS-01, REQ-FINOPS-01, REQ-GPU-01, and REQ-AZ-01.
"""
from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from src.agents.langsmith_workflow import (
    LangSmithInsuranceMiningWorkflow,
    get_sample_broker_binder,
    get_sample_issued_policy_jacket,
    reconcile_binder_vs_policy,
)
from src.api.main import app
from src.core.schemas import (
    BaseLLMArchitecture,
    UnslothQLoRAConfig,
    VectorBackendType,
)
from src.evals.runner import run_all_evaluation_gates
from src.finetuning.azure_finops_dispatcher import (
    compute_azure_finops_comparison,
    get_free_gpu_dispatch_targets,
)
from src.finetuning.long_context_architectures import (
    benchmark_long_context_architectures,
    evaluate_needle_in_policy_haystack,
)
from src.finetuning.mlflow_evaluator import (
    compute_entity_f1_score,
    compute_population_stability_index,
    evaluate_and_check_retraining_gate,
)
from src.finetuning.unsloth_trainer import (
    build_insurance_dpo_preference_dataset,
    build_insurance_sft_dataset,
    run_unsloth_dpo_alignment,
    run_unsloth_qlora_experiment,
)
from src.rag.embeddings import (
    InsuranceSentenceTransformerEncoder,
    expand_insurance_query,
)
from src.rag.table_stitcher import (
    parse_and_audit_custom_policy_text,
    stitch_multipage_insurance_tables,
)
from src.rag.vector_router import UnifiedInsuranceVectorRouter

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_req_ft_01_unsloth_qlora_pipeline() -> None:
    """REQ-FT-01 & INV-03: Verify Unsloth 4-bit nf4 QLoRA across Qwen, Llama, Falcon, and OPT."""
    dataset = build_insurance_sft_dataset()
    assert len(dataset) >= 2
    assert "SELF_INSURED_RETENTION" in dataset[0]["output"]

    for arch in BaseLLMArchitecture:
        cfg = UnslothQLoRAConfig(base_model=arch, lora_r=16, lora_alpha=32, max_seq_length=32768)
        summary = run_unsloth_qlora_experiment(cfg)
        assert summary.vram_savings_pct >= 60.0
        assert summary.finetuned_perplexity < summary.baseline_perplexity
        assert "gguf_q4_k_m" in summary.exported_formats
        assert "vllm_merged_16bit" in summary.exported_formats


def test_req_ft_02_long_context_architectures() -> None:
    """REQ-FT-02 & INV-02: Verify RoPE YaRN vs. BigBird vs. Memorizing Transformers & Needle recall."""
    reports = benchmark_long_context_architectures()
    names = [r.architecture_name for r in reports]
    assert len(reports) == 3
    assert any("Qwen2.5" in n for n in names)
    assert any("BigBird" in n for n in names)
    assert any("Memorizing Transformer" in n for n in names)

    naive = evaluate_needle_in_policy_haystack(strategy="naive_fixed_chunk_rag")
    assert naive["endorsement_override_detected"] is False
    assert naive["accuracy"] == 0.0

    hierarchical = evaluate_needle_in_policy_haystack(strategy="long_context_hierarchical")
    assert hierarchical["endorsement_override_detected"] is True
    assert hierarchical["extracted_flood_limit_usd"] == 500_000.0
    assert hierarchical["accuracy"] == 1.0


def test_req_ft_03_unsloth_dpo_preference_alignment() -> None:
    """REQ-FT-03: Verify Stage 2 Direct Preference Optimization (DPO) reduces hallucinations."""
    dpo_pairs = build_insurance_dpo_preference_dataset()
    assert len(dpo_pairs) >= 2
    assert "SELF_INSURED_RETENTION" in dpo_pairs[0]["chosen"]

    dpo_summary = run_unsloth_dpo_alignment()
    assert dpo_summary.reward_margin > 3.0
    assert dpo_summary.post_dpo_hallucination_rate_pct < dpo_summary.pre_dpo_hallucination_rate_pct
    assert dpo_summary.sir_vs_deductible_disambiguation_f1 >= 0.99


def test_req_doc_01_multipage_sov_stitcher_and_live_parser() -> None:
    """REQ-DOC-01 & INV-01: Verify multi-page headerless SOV stitching and live policy parser."""
    stitched = stitch_multipage_insurance_tables()
    assert stitched.pages_spanned == [95, 96]
    assert stitched.orphan_rows_reattached == 2
    assert len(stitched.records) == 4
    assert stitched.deterministic_total_usd == 72_000_000.0

    parsed = parse_and_audit_custom_policy_text(
        "ENDORSEMENT CP-14-20: Cyber Extortion Sublimit is amended to $3,000,000. Restore Flood Zone A."
    )
    # With Flood Zone A restored ($10M) and Cyber sublimit raised to $3M ($2M gap instead of $3.5M),
    # exposure delta drops significantly from $13,926,400 to $2,426,400
    assert parsed["reconciliation"]["total_uninsured_exposure_delta_usd"] == 2_426_400.0


def test_req_rag_01_query_encoding_and_embeddings() -> None:
    """REQ-RAG-01: Verify domain acronym expansion (SIR, SOV, TIV) and SentenceTransformers encoding."""
    expanded = expand_insurance_query("Check SIR and SOV TIV for cyber")
    assert "self-insured retention" in expanded
    assert "schedule of values" in expanded

    encoder = InsuranceSentenceTransformerEncoder()
    v1 = encoder.encode("Cyber extortion sublimit self-insured retention")
    v2 = encoder.encode("Ransomware cyber sublimit SIR endorsement")
    sim = encoder.cosine_similarity(v1, v2)
    assert len(v1) == 128
    assert 0.5 <= sim <= 1.0


def test_req_rag_02_multi_vector_hybrid_rag() -> None:
    """REQ-RAG-02: Verify FAISS, Azure Cognitive Search, Pinecone, and Weaviate hybrid RRF retrieval."""
    router = UnifiedInsuranceVectorRouter()
    for backend in VectorBackendType:
        results = router.search(
            query="Cyber Extortion SIR sublimit endorsement CP-14-20",
            backend=backend,
            top_k=3,
        )
        assert len(results) == 3
        assert results[0].backend_used == backend
        assert any("CP-14-20" in r.section_title for r in results)


def test_req_ag_01_langsmith_agentic_workflow() -> None:
    """REQ-AG-01 & INV-04: Verify 5-node LangSmith agentic pipeline execution & trace spans."""
    wf = LangSmithInsuranceMiningWorkflow()
    out = wf.run_pipeline()
    assert len(out["langsmith_traces"]) == 5
    assert out["visualization_kpis"]["total_insured_value_usd"] == 47_500_000.0


def test_req_ag_02_binder_policy_divergence() -> None:
    """REQ-AG-02 & INV-01: Verify deterministic detection of silent exclusions, SIR shift & sublimits."""
    binder = get_sample_broker_binder()
    policy = get_sample_issued_policy_jacket()
    report = reconcile_binder_vs_policy(binder, policy)

    assert len(report.discrepancies) == 5
    categories = {d.category for d in report.discrepancies}
    assert "SILENT_ENDORSEMENT_EXCLUSION" in categories
    assert "SUBLIMIT_REDUCTION" in categories
    assert "RETENTION_TYPE_SHIFT_TO_SIR" in categories
    assert "COINSURANCE_PENALTY_INCREASE" in categories
    assert "PREMIUM_VARIANCE" in categories
    assert report.total_uninsured_exposure_delta_usd == 13_926_400.0


def test_req_finops_01_and_req_gpu_01_cloud_dispatch() -> None:
    """REQ-FINOPS-01 & REQ-GPU-01: Verify Azure FinOps calculator and Free GPU targets."""
    finops = compute_azure_finops_comparison(monthly_policies_processed=25000, avg_tokens_per_policy=42000)
    assert finops.aks_vllm_qlora_monthly_cost_usd < finops.azure_openai_payg_monthly_cost_usd
    assert finops.annual_cost_savings_usd > 10000.0

    targets = get_free_gpu_dispatch_targets()
    providers = {t.provider_id for t in targets}
    assert providers == {"colab", "kaggle", "modal", "lightning"}


def test_req_mlops_01_evaluation_and_retraining() -> None:
    """REQ-MLOPS-01 & INV-05: Verify F1, Perplexity, PSI drift and automatic Azure ML retraining gate."""
    policy = get_sample_issued_policy_jacket().model_dump()
    gt = policy["coverages"]

    f1_perfect = compute_entity_f1_score(gt, gt)
    assert f1_perfect["f1_score"] == 1.0

    psi_low = compute_population_stability_index([0.4, 0.3, 0.2, 0.1], [0.39, 0.31, 0.20, 0.10])
    assert psi_low < 0.05

    healthy = evaluate_and_check_retraining_gate(gt, gt, cross_entropy_loss=1.02)
    assert healthy.retraining_triggered is False
    assert healthy.aml_pipeline_job_id is None

    degraded = evaluate_and_check_retraining_gate(gt[:1], gt, cross_entropy_loss=2.15)
    assert degraded.retraining_triggered is True
    assert degraded.aml_pipeline_job_id is not None


def test_req_az_01_infrastructure_manifests() -> None:
    """REQ-AZ-01: Verify Azure ML v2, AKS Ray/vLLM, DeepSpeed ZeRO-3, Kaggle & Modal files exist."""
    expected_files = [
        "Dockerfile",
        "docker-compose.yml",
        "azure-pipelines.yml",
        "deploy/aml/aml_finetune_pipeline.yml",
        "deploy/aks/k8s-vllm-ray-deployment.yaml",
        "deploy/deepspeed/ds_config_zero3.json",
        "scripts/train_qlora_unsloth.sh",
        "scripts/deploy_azure_aks_aml.sh",
        "scripts/modal_unsloth_train.py",
        "scripts/dispatch_free_gpu.py",
        "notebooks/unsloth_qlora_insurance_finetuning.ipynb",
        "notebooks/kernel-metadata.json",
    ]
    for rel_path in expected_files:
        assert (REPO_ROOT / rel_path).exists(), f"Missing required artifact: {rel_path}"


def test_fastapi_endpoints_and_4_gate_runner() -> None:
    """Verify all REST API endpoints (including v2.0 upload, DPO, FinOps & GPU) and 4-Gate runner."""
    r_health = client.get("/healthz")
    assert r_health.status_code == 200
    assert r_health.json()["version"] == "2.0.0"

    r_rec = client.get("/api/v1/reconcile")
    assert r_rec.status_code == 200
    assert r_rec.json()["stitched_multipage_sov"]["deterministic_total_usd"] == 72_000_000.0

    r_upload = client.post(
        "/api/v1/documents/upload-and-audit",
        json={"raw_policy_text": "Cyber Extortion Sublimit $2,500,000. Restore Flood Zone A."},
    )
    assert r_upload.status_code == 200
    assert r_upload.json()["reconciliation"]["total_uninsured_exposure_delta_usd"] < 13_926_400.0

    r_rag = client.post(
        "/api/v1/rag/search",
        json={"query": "Flood Zone A exclusion endorsement", "backend": "FAISS_HYBRID_LOCAL", "top_k": 3},
    )
    assert r_rag.status_code == 200
    assert len(r_rag.json()["results"]) == 3

    r_ft = client.post("/api/v1/finetuning/simulate", json={"lora_r": 32, "lora_alpha": 64})
    assert r_ft.status_code == 200
    assert r_ft.json()["qlora_summary"]["vram_savings_pct"] >= 60.0
    assert r_ft.json()["dpo_summary"]["post_dpo_hallucination_rate_pct"] < 1.0

    r_cloud = client.post("/api/v1/cloud/finops-and-gpu", json={"monthly_policies_processed": 25000})
    assert r_cloud.status_code == 200
    assert len(r_cloud.json()["free_gpu_targets"]) == 4

    r_mlops = client.post("/api/v1/mlops/evaluate", json={"simulate_degradation": True})
    assert r_mlops.status_code == 200
    assert r_mlops.json()["retraining_triggered"] is True

    eval_report = run_all_evaluation_gates()
    assert eval_report["all_passed"] is True
