# PolicyMiner LLM Platform — Architectural Specification (`SPECIFICATION.md`)

**Methodology:** 4-Layer Specification-Driven Development (`SDD`)  
**Target Domain:** Commercial Property & Casualty (P&C) Insurance Document Mining (Binders, Policy Jackets, Proposals, Schedules of Values / SOV, ACORD 25/125/140)  
**Target Stack:** Unsloth (LoRA/QLoRA/PEFT), Long-Context LLMs (Qwen 2.5 / Llama-Long / BigBird / Memorizing Transformers), LangChain + LangSmith Agentic Workflows, Multi-Backend Hybrid RAG (FAISS, Azure Cognitive Search, Pinecone, Weaviate), Azure Machine Learning (AML) + AKS + DeepSpeed + Ray + MLflow.

---

## Layer 1: Architectural Constitution (Non-Negotiable Invariants)

- **`INV-01` (Zero Unbounded LLM Math on Insurance Limits & TIV):** Total Insured Value (TIV), Self-Insured Retentions (SIR), Coinsurance Penalties, and Binder-vs-Policy Dollar Deltas must be computed and verified by deterministic Python math engines on top of structured Pydantic v2 extractions.
- **`INV-02` (Strict Endorsement Hierarchy Precedence):** Policy Endorsement Riders always supersede Base Policy Jacket clauses. Extraction and RAG pipelines must preserve parent-child section provenance so exclusions in Endorsement Riders are never severed from Insuring Agreements.
- **`INV-03` (Parameter-Efficient Quantized Training & Inference Bounds):** Fine-tuning configurations (`Unsloth`, `QLoRA`) must enforce 4-bit NormalFloat (`nf4`) quantization with double quantization and targeted LoRA adapters (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) to guarantee $\ge 60\%$ VRAM savings.
- **`INV-04` (End-to-End LangSmith Traceability):** Every node in the multi-step insurance mining workflow must emit structured trace spans (input hash, latency, token count, confidence score, and provenance citations) compatible with LangSmith and OpenTelemetry.
- **`INV-05` (Automated MLOps Retraining Gate):** Any model deployment whose field-level Extraction F1-Score drops below `0.90`, Perplexity exceeds `6.50`, or Input Population Stability Index ($\text{PSI}$) exceeds `0.20` must automatically trigger a blocked gate and generate an Azure ML retraining job payload.

---

## Layer 2: Traceable Requirement IDs (`REQ-XX`)

| Requirement ID | Category | Description | Verification Gate |
| :--- | :--- | :--- | :--- |
| **`REQ-FT-01`** | Fine-Tuning (`Unsloth` / `QLoRA`) | Implement parameter-efficient fine-tuning (`LoRA` / `QLoRA` 4-bit `nf4`) via `Unsloth` & `PEFT` for `Qwen2.5`, `Llama-3.1`, `Falcon`, and `OPT` architectures, computing exact trainable parameters, VRAM savings, and GGUF/vLLM export manifests. | `tests/test_suite.py::test_req_ft_01_unsloth_qlora_pipeline` |
| **`REQ-FT-02`** | Long-Context LLMs | Benchmark long-document attention architectures (`Qwen2.5/Llama-Long` RoPE YaRN 128k vs. `BigBird` Block-Sparse $O(n)$ vs. `Memorizing Transformers` kNN-augmented memory) on multi-page insurance policies and "Needle-in-a-Policy-Haystack" endorsement retrieval. | `tests/test_suite.py::test_req_ft_02_long_context_architectures` |
| **`REQ-RAG-01`** | Query Encoding & Embeddings | Implement domain-aware query encoding (`SentenceTransformers` bi-encoder + insurance acronym expansion + cross-encoder re-ranking) to resolve commercial P&C terminology (`SIR`, `ALAE`, `PML`, `SOV`, `ACORD`). | `tests/test_suite.py::test_req_rag_01_query_encoding_and_embeddings` |
| **`REQ-RAG-02`** | Multi-Backend Vector RAG | Implement a unified Hybrid RAG router supporting **FAISS**, **Azure Cognitive Search (Azure AI Search)**, **Pinecone**, and **Weaviate** with Reciprocal Rank Fusion ($\text{RRF}, k=60$) and section-hierarchy metadata filtering. | `tests/test_suite.py::test_req_rag_02_multi_vector_hybrid_rag` |
| **`REQ-AG-01`** | LangChain / LangSmith Agentic Mining | Orchestrate a 5-node stateful agentic workflow (`Classifier` $\rightarrow$ `PolicyExtractor` $\rightarrow$ `SovParser` $\rightarrow$ `BinderPolicyReconciler` $\rightarrow$ `Visualizer`) with LangSmith `@traceable` run trees. | `tests/test_suite.py::test_req_ag_01_langsmith_agentic_workflow` |
| **`REQ-AG-02`** | Binder vs. Policy Divergence | Automatically detect silent coverage drift between a Broker Binder/Proposal and the issued Carrier Policy Jacket (e.g., sublimit reductions, deductible/SIR hikes, newly injected exclusionary riders). | `tests/test_suite.py::test_req_ag_02_binder_policy_divergence` |
| **`REQ-MLOPS-01`** | Model Evaluation & Drift Monitoring | Compute entity-level Precision, Recall, **F1-Score**, **Perplexity ($\text{PPL}$)**, **Inference Time (TTFT / TPOT)**, and **PSI Drift**, logging to `MLflow` and triggering automated Azure ML retraining pipelines when thresholds degrade. | `tests/test_suite.py::test_req_mlops_01_evaluation_and_retraining` |
| **`REQ-AZ-01`** | Azure Cloud, AKS, AML & DeepSpeed | Provide production deployment manifests for **Azure Machine Learning (AML)**, **Azure Kubernetes Service (AKS)** with **Ray Serve + vLLM**, **DeepSpeed ZeRO-3**, **Docker Compose**, and **Azure Pipelines**. | `tests/test_suite.py::test_req_az_01_infrastructure_manifests` |

---

## Layer 3: Contract-First Schemas
Defined in [`src/core/schemas.py`](./src/core/schemas.py) using strict Pydantic v2 models (`extra="forbid"`).

## Layer 4: 4-Gate Automated Verification Runner
Defined in [`src/evals/runner.py`](./src/evals/runner.py) and [`tests/test_suite.py`](./tests/test_suite.py).
