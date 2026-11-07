"""
Layer 3 SDD: Strict Pydantic v2 Contracts for Commercial Insurance Mining,
Unsloth QLoRA Fine-Tuning, Long-Context Attention, Multi-Backend RAG,
LangSmith Agentic Orchestration, and Azure MLOps Monitoring.

Governing Requirements: REQ-FT-01, REQ-FT-02, REQ-RAG-01, REQ-RAG-02, REQ-AG-01, REQ-AG-02, REQ-MLOPS-01
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class DocumentType(str, Enum):
    BINDER = "COMMERCIAL_BINDER"
    POLICY_JACKET = "CARRIER_POLICY_JACKET"
    PROPOSAL = "BROKER_PROPOSAL"
    SOV = "SCHEDULE_OF_VALUES"
    ACORD_25 = "ACORD_25_CERTIFICATE"
    LOSS_RUN = "CARRIER_LOSS_RUN"


class RetentionType(str, Enum):
    DEDUCTIBLE = "DEDUCTIBLE"
    SIR = "SELF_INSURED_RETENTION"


class CoverageLineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coverage_code: str = Field(..., description="Standardized ISO/ACORD coverage code, e.g., PROP_BLDG, CYBER_EXT, FLOOD_Z_A")
    coverage_name: str = Field(..., description="Human-readable line of coverage")
    limit_usd: float = Field(..., ge=0.0, description="Occurrence or sublimit in USD")
    aggregate_limit_usd: float = Field(..., ge=0.0, description="Annual aggregate limit in USD")
    retention_usd: float = Field(..., ge=0.0, description="Deductible or Self-Insured Retention (SIR) amount in USD")
    retention_type: RetentionType = Field(default=RetentionType.DEDUCTIBLE)
    coinsurance_pct: float = Field(default=80.0, ge=0.0, le=100.0)
    is_excluded: bool = Field(default=False, description="True if an endorsement rider explicitly excludes this coverage")
    source_section: str = Field(..., description="Section or Endorsement Rider ID, e.g., 'Endorsement CP-14-20'")
    page_number: int = Field(..., ge=1)


class SovPropertyLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loc_id: str
    address: str
    occupancy: str
    construction_class: str = Field(..., description="ISO 1-6 construction class or Frame/Joisted Masonry/Fire Resistive")
    building_value_usd: float = Field(..., ge=0.0)
    contents_value_usd: float = Field(..., ge=0.0)
    business_income_usd: float = Field(..., ge=0.0)
    flood_zone: str
    total_insured_value_usd: float = Field(..., ge=0.0)


class ExtractedInsuranceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_type: DocumentType
    named_insured: str
    carrier_name: str
    policy_number: str
    effective_date: str
    expiration_date: str
    total_annual_premium_usd: float = Field(..., ge=0.0)
    total_insured_value_usd: float = Field(..., ge=0.0)
    coverages: List[CoverageLineItem]
    endorsement_forms: List[str]
    key_exclusions: List[str]
    sov_locations: List[SovPropertyLocation] = Field(default_factory=list)


class DivergenceSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    INFO = "INFO"


class CoverageDiscrepancy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discrepancy_id: str
    coverage_code: str
    coverage_name: str
    category: Literal[
        "SUBLIMIT_REDUCTION",
        "RETENTION_ESCALATION",
        "RETENTION_TYPE_SHIFT_TO_SIR",
        "SILENT_ENDORSEMENT_EXCLUSION",
        "COINSURANCE_PENALTY_INCREASE",
        "PREMIUM_VARIANCE",
    ]
    severity: DivergenceSeverity
    binder_value: str
    policy_value: str
    financial_exposure_delta_usd: float = Field(..., ge=0.0)
    policy_citation: str
    remediation_action: str


class BinderPolicyReconciliationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reconciliation_id: str
    named_insured: str
    binder_doc_id: str
    policy_doc_id: str
    alignment_score_pct: float = Field(..., ge=0.0, le=100.0)
    total_uninsured_exposure_delta_usd: float = Field(..., ge=0.0)
    discrepancies: List[CoverageDiscrepancy]
    executive_summary: str


class BaseLLMArchitecture(str, Enum):
    QWEN_2_5_14B = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
    LLAMA_3_1_8B = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    FALCON_11B = "tiiuae/falcon-11B"
    OPT_6_7B = "facebook/opt-6.7b"


class UnslothQLoRAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model: BaseLLMArchitecture = Field(default=BaseLLMArchitecture.QWEN_2_5_14B)
    max_seq_length: int = Field(default=32768, ge=2048, le=131072)
    load_in_4bit: bool = Field(default=True)
    quant_type: Literal["nf4", "fp4"] = Field(default="nf4")
    use_double_quant: bool = Field(default=True)
    lora_r: int = Field(default=16, ge=4, le=128)
    lora_alpha: int = Field(default=32, ge=8, le=256)
    lora_dropout: float = Field(default=0.0, ge=0.0, le=0.5)
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_gradient_checkpointing: Literal["unsloth", "true", "false"] = Field(default="unsloth")
    learning_rate: float = Field(default=2e-4, gt=0.0)
    per_device_train_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)


class UnslothTrainingRunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    base_model: str
    total_params_millions: float
    trainable_lora_params_millions: float
    trainable_percentage: float
    standard_fp16_vram_gb: float
    unsloth_qlora_vram_gb: float
    vram_savings_pct: float
    throughput_tokens_per_sec: float
    initial_loss: float
    final_loss: float
    baseline_perplexity: float
    finetuned_perplexity: float
    exported_formats: List[str]
    azure_blob_adapter_uri: str


class LongContextArchitectureReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture_name: str
    attention_mechanism: str
    max_context_tokens: int
    theoretical_complexity: str
    kv_cache_vram_gb_at_64k: float
    needle_in_haystack_recall_pct: float
    cross_section_endorsement_accuracy_pct: float
    p95_inference_latency_ms: float
    strengths_for_insurance: str


class VectorBackendType(str, Enum):
    FAISS = "FAISS_HYBRID_LOCAL"
    AZURE_COGNITIVE_SEARCH = "AZURE_AI_SEARCH_HYBRID"
    PINECONE = "PINECONE_SERVERLESS"
    WEAVIATE = "WEAVIATE_HYBRID_BM25"


class RetrievedPolicyChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    section_title: str
    parent_section_id: str
    page_number: int
    content: str
    dense_cosine_score: float
    sparse_bm25_score: float
    rrf_fused_score: float
    cross_encoder_rerank_score: float
    backend_used: VectorBackendType


class LangSmithTraceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    span_id: str
    node_name: str
    run_type: Literal["chain", "llm", "retriever", "tool", "parser"]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    status: Literal["SUCCESS", "FALLBACK", "ERROR"]
    metadata: Dict[str, Any]


class MLflowEvaluationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    run_id: str
    model_version: str
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    perplexity: float = Field(..., gt=0.0)
    ttft_ms: float = Field(..., gt=0.0)
    tpot_ms: float = Field(..., gt=0.0)
    input_psi_drift: float = Field(..., ge=0.0)
    retraining_triggered: bool
    retraining_reason: str
    aml_pipeline_job_id: Optional[str] = None
