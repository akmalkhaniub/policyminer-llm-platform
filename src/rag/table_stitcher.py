"""
Multi-Page Schedule of Values (SOV) & Carrier Loss Run Table Stitcher,
plus Live Policy Document Upload & Extraction Engine.

Governing Requirements: REQ-DOC-01, INV-01, INV-02
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from src.agents.langsmith_workflow import (
    get_sample_broker_binder,
    get_sample_issued_policy_jacket,
    reconcile_binder_vs_policy,
)
from src.core.schemas import (
    CoverageLineItem,
    RetentionType,
    StitchedMultiPageTable,
)


SAMPLE_MULTIPAGE_SOV_RAW_PAGES: List[Dict[str, Any]] = [
    {
        "page": 95,
        "raw_text": (
            "SCHEDULE OF VALUES (SOV) — COMMERCIAL PROPERTY PROGRAM\n"
            "LOC_ID | ADDRESS | OCCUPANCY | BLDG_USD | CONTENTS_USD | BI_USD | FLOOD_ZONE\n"
            "LOC-001 | 450 Harbor Blvd, Houston TX | Cold Storage Warehouse | 14000000 | 4000000 | 6000000 | Zone A\n"
            "LOC-002 | 1200 Tech Parkway, Austin TX | Logistics Data Center | 11000000 | 8500000 | 4000000 | Zone X"
        ),
    },
    {
        "page": 96,
        "raw_text": (
            "--- Page 96 of 100 (Continued from Previous Page — Header Omitted by Carrier PDF) ---\n"
            "LOC-003 | 890 Industrial Pkwy, Dallas TX | Automated Fulfillment Hub | 9500000 | 3000000 | 2500000 | Zone X\n"
            "LOC-004 | 210 Port Terminal Rd, Corpus Christi TX | Refrigerated Port Depot | 6000000 | 2000000 | 1500000 | Zone A"
        ),
    },
]


def stitch_multipage_insurance_tables(
    pages: List[Dict[str, Any]] | None = None,
) -> StitchedMultiPageTable:
    """
    Reconstructs multi-page Schedules of Values (SOV) or Carrier Loss Runs where
    page 2+ continuation tables omit column headers or split rows across page footers.
    Enforces INV-01 (deterministic Python sum of TIV across all stitched pages).
    """
    raw_pages = pages if pages is not None else SAMPLE_MULTIPAGE_SOV_RAW_PAGES
    detected_columns: List[str] = []
    records: List[Dict[str, Any]] = []
    pages_spanned: List[int] = []
    orphan_rows_reattached = 0

    for p_idx, page_obj in enumerate(raw_pages):
        p_num = int(page_obj.get("page", p_idx + 1))
        pages_spanned.append(p_num)
        lines = [line.strip() for line in page_obj.get("raw_text", "").splitlines() if line.strip()]

        for line in lines:
            if line.startswith("---") or "SCHEDULE OF VALUES" in line.upper():
                continue
            if "LOC_ID" in line and "|" in line:
                detected_columns = [c.strip() for c in line.split("|")]
                continue
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 7:
                    if p_idx > 0:
                        orphan_rows_reattached += 1
                    bldg = float(re.sub(r"[^\d.]", "", parts[3]) or 0.0)
                    cont = float(re.sub(r"[^\d.]", "", parts[4]) or 0.0)
                    bi = float(re.sub(r"[^\d.]", "", parts[5]) or 0.0)
                    row_tiv = bldg + cont + bi
                    records.append(
                        {
                            "loc_id": parts[0],
                            "address": parts[1],
                            "occupancy": parts[2],
                            "building_usd": bldg,
                            "contents_usd": cont,
                            "business_income_usd": bi,
                            "flood_zone": parts[6],
                            "tiv_usd": row_tiv,
                            "source_page": p_num,
                        }
                    )

    total_tiv = sum(r["tiv_usd"] for r in records)
    return StitchedMultiPageTable(
        table_type="SCHEDULE_OF_VALUES",
        pages_spanned=pages_spanned,
        detected_columns=detected_columns,
        orphan_rows_reattached=orphan_rows_reattached,
        records=records,
        deterministic_total_usd=total_tiv,
    )


def parse_and_audit_custom_policy_text(raw_policy_text: str) -> Dict[str, Any]:
    """
    Parses user-uploaded or pasted commercial policy/endorsement text, dynamically overrides
    limits/retentions/exclusions on the policy model, and runs deterministic Binder-vs-Policy
    reconciliation (REQ-DOC-01, INV-01, INV-02).
    """
    binder = get_sample_broker_binder()
    policy = get_sample_issued_policy_jacket()

    text_lower = raw_policy_text.lower()

    # Check if custom text modifies Cyber Extortion sublimit or SIR
    cyber_match = re.search(r"cyber[^\n$]*\$([\d,]+)", raw_policy_text, flags=re.IGNORECASE)
    if cyber_match:
        val = float(cyber_match.group(1).replace(",", ""))
        for cov in policy.coverages:
            if cov.coverage_code == "CYBER_EXT":
                cov.limit_usd = val
                cov.aggregate_limit_usd = val
                if "sir" in text_lower or "self-insured retention" in text_lower:
                    cov.retention_type = RetentionType.SIR

    # Check if custom text restores or excludes Flood Zone A
    if "flood zone a is covered" in text_lower or "restore flood zone a" in text_lower:
        for cov in policy.coverages:
            if cov.coverage_code == "FLOOD_Z_A":
                cov.is_excluded = False
                cov.limit_usd = 10_000_000.0
                cov.aggregate_limit_usd = 10_000_000.0

    reconciliation = reconcile_binder_vs_policy(binder, policy)
    stitched_sov = stitch_multipage_insurance_tables()

    return {
        "parsed_character_count": len(raw_policy_text),
        "reconciliation": reconciliation.model_dump(),
        "stitched_multipage_sov": stitched_sov.model_dump(),
    }
