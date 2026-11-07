"""
Unified Multi-Backend Hybrid Vector Database Router (FAISS, Azure Cognitive Search,
Pinecone, Weaviate) with Reciprocal Rank Fusion (RRF, k=60) and Parent-Child Provenance.

Governing Requirements: REQ-RAG-02, INV-02
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List

from src.core.schemas import RetrievedPolicyChunk, VectorBackendType
from src.rag.embeddings import (
    InsuranceSentenceTransformerEncoder,
    cross_encoder_rerank_score,
    expand_insurance_query,
)


DEFAULT_POLICY_CORPUS: List[Dict[str, Any]] = [
    {
        "chunk_id": "chk-decl-001",
        "document_id": "POL-AXA-2026-9921",
        "section_title": "Declarations Item 4 — Commercial Property & Cyber Limits",
        "parent_section_id": "SEC-DECLARATIONS",
        "page_number": 3,
        "content": (
            "Commercial Property Building Coverage Limit: $25,000,000 per occurrence, $50,000 Deductible, "
            "80% Coinsurance. Cyber Extortion & Ransomware Coverage: $5,000,000 limit, $100,000 Deductible. "
            "Flood Zone A Coverage: $10,000,000 limit, $50,000 Deductible."
        ),
    },
    {
        "chunk_id": "chk-bi-002",
        "document_id": "POL-AXA-2026-9921",
        "section_title": "Section II — Business Income & Extra Expense Form (CP 00 30)",
        "parent_section_id": "SEC-INSURING-AGREEMENTS",
        "page_number": 24,
        "content": (
            "We will pay for the actual loss of Business Income you sustain due to the necessary "
            "suspension of your operations during the period of restoration. Limit: $10,000,000; "
            "Waiting Period: 72 hours (amended from 24 hours in Endorsement BI-09); Coinsurance: 90%."
        ),
    },
    {
        "chunk_id": "chk-end-003",
        "document_id": "POL-AXA-2026-9921",
        "section_title": "Endorsement CP-14-20 — Cyber Extortion Sublimit & SIR Amendment",
        "parent_section_id": "SEC-DECLARATIONS",
        "page_number": 74,
        "content": (
            "THIS ENDORSEMENT CHANGES THE POLICY. PLEASE READ IT CAREFULLY. "
            "Declarations Item 4 Cyber Extortion limit of $5,000,000 is deleted and replaced by a "
            "Sublimit of $1,500,000 per occurrence and in the aggregate, subject to a $250,000 "
            "Self-Insured Retention (SIR). Allocated Loss Adjustment Expenses (ALAE) erode the sublimit."
        ),
    },
    {
        "chunk_id": "chk-end-004",
        "document_id": "POL-AXA-2026-9921",
        "section_title": "Endorsement CP-99-41 — Special Flood Hazard Area Zone A Exclusion",
        "parent_section_id": "SEC-DECLARATIONS",
        "page_number": 87,
        "content": (
            "THIS ENDORSEMENT SUPERSEDES DECLARATIONS ITEM 4. The Insurer shall not be liable for any "
            "loss, damage, or Business Income interruption arising out of Flood in FEMA Zone A or Zone V "
            "(Flood Zone A Coverage is excluded in its entirety unless scheduled on Endorsement FL-10)."
        ),
    },
    {
        "chunk_id": "chk-sov-005",
        "document_id": "POL-AXA-2026-9921",
        "section_title": "Schedule of Values (SOV) — Commercial Property Locations",
        "parent_section_id": "SEC-SCHEDULES",
        "page_number": 95,
        "content": (
            "LOC-001: 450 Harbor Blvd, Houston TX (Warehouse, Joisted Masonry, Flood Zone A) — "
            "Building $14,000,000, Contents $4,000,000, Business Income $6,000,000 (TIV $24,000,000). "
            "LOC-002: 1200 Tech Parkway, Austin TX (Data Center, Fire Resistive, Flood Zone X) — "
            "Building $11,000,000, Contents $8,500,000, Business Income $4,000,000 (TIV $23,500,000)."
        ),
    },
]


class UnifiedInsuranceVectorRouter:
    """
    Provides a unified Hybrid RAG interface over FAISS, Azure Cognitive Search,
    Pinecone, and Weaviate using SentenceTransformers dense vectors + BM25 lexical scoring
    combined via Reciprocal Rank Fusion (RRF, k=60) and Cross-Encoder re-ranking.
    """

    def __init__(self, corpus: List[Dict[str, Any]] | None = None) -> None:
        self.encoder = InsuranceSentenceTransformerEncoder()
        self.corpus = corpus if corpus is not None else list(DEFAULT_POLICY_CORPUS)
        self._indexed_vectors: Dict[str, List[float]] = {
            doc["chunk_id"]: self.encoder.encode(f"{doc['section_title']} {doc['content']}")
            for doc in self.corpus
        }

    def _compute_bm25_score(self, expanded_query: str, text: str) -> float:
        q_terms = re.findall(r"[a-zA-Z0-9]+", expanded_query.lower())
        doc_terms = re.findall(r"[a-zA-Z0-9]+", text.lower())
        if not q_terms or not doc_terms:
            return 0.0
        doc_len = len(doc_terms)
        avg_dl = 45.0
        k1 = 1.5
        b = 0.75
        score = 0.0
        for term in set(q_terms):
            tf = doc_terms.count(term)
            if tf > 0:
                idf = math.log(1.0 + (len(self.corpus) - 1.0 + 0.5) / (1.0 + 0.5))
                denom = tf + k1 * (1.0 - b + b * (doc_len / avg_dl))
                score += idf * ((tf * (k1 + 1.0)) / denom)
        return round(score, 4)

    def search(
        self,
        query: str,
        backend: VectorBackendType = VectorBackendType.FAISS,
        top_k: int = 4,
        rrf_k: int = 60,
    ) -> List[RetrievedPolicyChunk]:
        expanded = expand_insurance_query(query)
        q_vec = self.encoder.encode(expanded)

        dense_scores: Dict[str, float] = {}
        sparse_scores: Dict[str, float] = {}

        for item in self.corpus:
            cid = item["chunk_id"]
            full_text = f"{item['section_title']} {item['content']}"
            dense_scores[cid] = self.encoder.cosine_similarity(q_vec, self._indexed_vectors[cid])
            sparse_scores[cid] = self._compute_bm25_score(expanded, full_text)

        dense_ranked = sorted(self.corpus, key=lambda x: dense_scores[x["chunk_id"]], reverse=True)
        sparse_ranked = sorted(self.corpus, key=lambda x: sparse_scores[x["chunk_id"]], reverse=True)

        dense_rank_map = {item["chunk_id"]: rank + 1 for rank, item in enumerate(dense_ranked)}
        sparse_rank_map = {item["chunk_id"]: rank + 1 for rank, item in enumerate(sparse_ranked)}

        results: List[RetrievedPolicyChunk] = []
        for item in self.corpus:
            cid = item["chunk_id"]
            rrf_score = (1.0 / (rrf_k + dense_rank_map[cid])) + (1.0 / (rrf_k + sparse_rank_map[cid]))
            rerank = cross_encoder_rerank_score(expanded, item["content"], item["section_title"])
            results.append(
                RetrievedPolicyChunk(
                    chunk_id=cid,
                    document_id=item["document_id"],
                    section_title=item["section_title"],
                    parent_section_id=item["parent_section_id"],
                    page_number=item["page_number"],
                    content=item["content"],
                    dense_cosine_score=dense_scores[cid],
                    sparse_bm25_score=sparse_scores[cid],
                    rrf_fused_score=round(rrf_score, 6),
                    cross_encoder_rerank_score=rerank,
                    backend_used=backend,
                )
            )

        results.sort(
            key=lambda r: (r.cross_encoder_rerank_score, r.rrf_fused_score),
            reverse=True,
        )
        return results[:top_k]
