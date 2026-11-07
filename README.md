# PolicyMiner LLM Platform — Commercial Insurance Program Mining, Unsloth QLoRA & Azure MLOps

[![SDD Architecture](https://img.shields.io/badge/Methodology-Spec--Driven%20Development%20(SDD)-4f46e5)](./SPECIFICATION.md)
[![Fine-Tuning](https://img.shields.io/badge/LLM%20Optimization-Unsloth%204--Bit%20QLoRA%20%7C%20PEFT-10b981)](./src/finetuning/unsloth_trainer.py)
[![Hybrid RAG](https://img.shields.io/badge/Vector%20RAG-Azure%20AI%20Search%20%7C%20FAISS%20%7C%20Pinecone%20%7C%20Weaviate-0284c7)](./src/rag/vector_router.py)
[![Cloud & MLOps](https://img.shields.io/badge/Azure%20Cloud-AML%20v2%20%7C%20AKS%20vLLM%20%7C%20DeepSpeed%20%7C%20MLflow-2563eb)](./deploy/aml/aml_finetune_pipeline.yml)

**PolicyMiner** is an enterprise SaaS commercial insurance mining and LLM optimization platform engineered to automatically extract, reconcile, and visualize data locked in **commercial insurance binders, 100+ page policy jackets, proposals, Schedules of Values (SOV), and ACORD certificates**.

---

## 🎯 Executive Mapping to Senior LLM Engineer Requirements

| Core Engineering Pillar | Implemented Modules & Artifacts | Key Capabilities & Metrics |
| :--- | :--- | :--- |
| **1. Model Development & Unsloth QLoRA Fine-Tuning** | [`unsloth_trainer.py`](./src/finetuning/unsloth_trainer.py)<br/>[`long_context_architectures.py`](./src/finetuning/long_context_architectures.py)<br/>[`unsloth_qlora_insurance_finetuning.ipynb`](./notebooks/unsloth_qlora_insurance_finetuning.ipynb) | • **Unsloth 4-bit NormalFloat (`nf4`) QLoRA** (`r=16, α=32`) across **Qwen 2.5 14B**, **Llama 3.1 8B**, **Falcon 11B**, and **OPT 6.7B** (**>71% VRAM reduction**, Perplexity reduced from `6.84 → 2.33`)<br/>• **Long-Context (128k) Architecture Benchmark:** Compares RoPE YaRN (`Qwen2.5` / `Llama-Long`), Block-Sparse Attention (`BigBird` $O(n)$), and kNN External Memory (`Memorizing Transformers`) on 100-page policies |
| **2. LangChain / LangSmith Agentic Pipeline** | [`langsmith_workflow.py`](./src/agents/langsmith_workflow.py) | • 5-node stateful workflow (`DocumentClassifierNode` → `HybridVectorRetrieverNode` → `LongContextExtractorNode` → `SovTabularParserNode` → `BinderPolicyReconcilerNode`)<br/>• Automatically catches **Binder-vs-Policy Divergences** (e.g., silent Flood Zone A exclusions on Page 87, Cyber sublimit reductions on Page 74, and Deductible-to-SIR shifts) |
| **3. Multi-Backend Hybrid RAG & Query Encoding** | [`embeddings.py`](./src/rag/embeddings.py)<br/>[`vector_router.py`](./src/rag/vector_router.py) | • Commercial P&C Acronym/Jargon Query Expansion (`SIR`, `SOV`, `TIV`, `ALAE`, `ACORD`) + `SentenceTransformers` bi-encoder<br/>• Unified Hybrid Router across **Azure Cognitive Search**, **FAISS**, **Pinecone**, and **Weaviate** using **Reciprocal Rank Fusion ($\text{RRF}, k=60$)** and Cross-Encoder endorsement re-ranking |
| **4. Containerization, Azure AKS/AML & Distributed Training** | [`aml_finetune_pipeline.yml`](./deploy/aml/aml_finetune_pipeline.yml)<br/>[`k8s-vllm-ray-deployment.yaml`](./deploy/aks/k8s-vllm-ray-deployment.yaml)<br/>[`ds_config_zero3.json`](./deploy/deepspeed/ds_config_zero3.json) | • **Azure Machine Learning (AML) SDK v2** pipeline connected to **Azure Blob Storage**<br/>• **Azure Kubernetes Service (AKS)** orchestrating **Ray Serve + vLLM** with dynamic multi-LoRA serving & GPU HPA<br/>• **DeepSpeed ZeRO-3** config for distributed training on **Azure Databricks** / AML + Docker & Docker Compose |
| **5. Model Performance, Monitoring & MLOps** | [`mlflow_evaluator.py`](./src/finetuning/mlflow_evaluator.py)<br/>[`azure-pipelines.yml`](./azure-pipelines.yml) | • Tracks entity-level **F1-Score**, **Perplexity ($\text{PPL}$)**, **Inference Time (TTFT / TPOT)**, and **Population Stability Index ($\text{PSI}$)** drift in **MLflow**<br/>• Automated retraining gate triggers Azure ML pipeline jobs when `F1 < 0.90` or `PSI > 0.20` |

---

## 🚀 Quickstart (1-Command Verification & Interactive Workbench)

### 1. Run Automated Pytest Suite & 4-Gate Evaluation Harness
```bash
pytest -v
python -m src.evals.runner
```

### 2. Launch the Commercial Insurance Mining & MLOps Workbench
```bash
uvicorn src.api.main:app --reload --port 8020
```
Open `http://localhost:8020` to explore:
- **Tab 1 (Program Audit & SOV):** Inspect the `$13,926,400` uninsured coverage gap discovered between the Broker Binder and the 100-page Carrier Policy Jacket, plus verified Schedule of Values (`SOV`) TIV cards.
- **Tab 2 (Policy & Endorsement Search):** Run hybrid queries across **Azure Cognitive Search**, **FAISS**, **Pinecone**, and **Weaviate**.
- **Tab 3 (Unsloth QLoRA & Long-Context Lab):** Simulate LoRA rank/alpha/sequence-window trade-offs and inspect the 100-page Needle-in-a-Policy-Haystack benchmark.
- **Tab 4 (Azure MLOps & Quality Gates):** Trigger synthetic surplus-lines distribution drift to watch the **MLflow** monitor automatically dispatch an **Azure ML** retraining pipeline job.
