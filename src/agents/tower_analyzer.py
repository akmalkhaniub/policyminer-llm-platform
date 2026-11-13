"""
Commercial Insurance Program Tower & Coinsurance Penalty Analyzer.
Models Primary, Umbrella, and Excess layers to detect attachment gaps ("Drop-Down Failures")
and computes deterministic Coinsurance Penalties on underinsured SOV properties.

Governing Requirements: REQ-TOWER-01, INV-01
"""
from __future__ import annotations

import csv
import io
from typing import List

from src.agents.langsmith_workflow import (
    get_sample_broker_binder,
    get_sample_issued_policy_jacket,
    reconcile_binder_vs_policy,
)
from src.core.schemas import (
    CoinsuranceSimulationResult,
    DivergenceSeverity,
    ProgramTowerAnalysis,
    TowerAttachmentGap,
    TowerLayer,
)


def simulate_coinsurance_penalty(
    reported_tiv_usd: float = 47_500_000.0,
    carried_limit_usd: float = 35_000_000.0,
    coinsurance_pct: float = 90.0,
    simulated_loss_usd: float = 10_000_000.0,
    deductible_usd: float = 50_000.0,
) -> CoinsuranceSimulationResult:
    """
    Deterministic Commercial Property Coinsurance Penalty formula (INV-01):
        Required Limit = TIV * (Coinsurance% / 100)
        Did/Should Ratio = min(1.0, Carried Limit / Required Limit)
        Payout = max(0.0, min(Carried Limit, (Did/Should Ratio * Loss) - Deductible))
        Coinsurance Penalty = max(0.0, (Loss - Deductible) - Payout)
    """
    required_limit = round(reported_tiv_usd * (coinsurance_pct / 100.0), 2)
    ratio = min(1.0, carried_limit_usd / required_limit) if required_limit > 0 else 1.0
    unpenalized_payout = max(0.0, min(carried_limit_usd, simulated_loss_usd - deductible_usd))
    actual_payout = round(max(0.0, min(carried_limit_usd, (ratio * simulated_loss_usd) - deductible_usd)), 2)
    penalty = round(max(0.0, unpenalized_payout - actual_payout), 2)

    return CoinsuranceSimulationResult(
        reported_tiv_usd=reported_tiv_usd,
        carried_limit_usd=carried_limit_usd,
        coinsurance_pct=coinsurance_pct,
        required_limit_usd=required_limit,
        simulated_loss_usd=simulated_loss_usd,
        deductible_usd=deductible_usd,
        insurer_payout_usd=actual_payout,
        coinsurance_penalty_usd=penalty,
        is_underinsured=carried_limit_usd < required_limit,
    )


def analyze_insurance_program_tower(
    simulated_loss_usd: float = 10_000_000.0,
    coinsurance_pct: float = 90.0,
    carried_limit_usd: float = 35_000_000.0,
) -> ProgramTowerAnalysis:
    """
    Analyzes a 3-layer Commercial P&C Risk Tower (Primary + Umbrella + Excess)
    and detects silent attachment point gaps caused by Endorsement CP-14-20 sublimits.
    """
    layers: List[TowerLayer] = [
        TowerLayer(
            layer_name="Primary Property & Cyber Package",
            carrier="AXA XL / Syndicate 2003",
            policy_number="POL-AXA-2026-9921",
            attachment_point_usd=0.0,
            limit_usd=25_000_000.0,
            underlying_sublimit_usd=1_500_000.0,  # Cyber Extortion sublimited by Endorsement CP-14-20
            annual_premium_usd=438_900.0,
        ),
        TowerLayer(
            layer_name="1st Excess / Umbrella Layer ($15M xs $25M)",
            carrier="Chubb Commercial Excess",
            policy_number="EXC-CHB-2026-4410",
            attachment_point_usd=25_000_000.0,
            limit_usd=15_000_000.0,
            underlying_sublimit_usd=5_000_000.0,  # Excess Cyber attaches only above $5,000,000!
            annual_premium_usd=142_000.0,
        ),
        TowerLayer(
            layer_name="2nd Excess Catastrophe Layer ($25M xs $40M)",
            carrier="Zurich North America",
            policy_number="EXC-ZUR-2026-7782",
            attachment_point_usd=40_000_000.0,
            limit_usd=25_000_000.0,
            underlying_sublimit_usd=None,
            annual_premium_usd=98_500.0,
        ),
    ]

    gaps: List[TowerAttachmentGap] = []
    # Check Cyber Sublimit Attachment Gap between Primary ($1.5M) and 1st Excess ($5.0M attachment)
    primary_cyber_cap = layers[0].underlying_sublimit_usd or layers[0].limit_usd
    excess_cyber_attach = layers[1].underlying_sublimit_usd or layers[1].attachment_point_usd
    if primary_cyber_cap < excess_cyber_attach:
        gap_amt = excess_cyber_attach - primary_cyber_cap
        gaps.append(
            TowerAttachmentGap(
                gap_id="GAP-TOWER-001",
                coverage_line="Cyber Extortion & Ransomware (Excess Attachment)",
                lower_layer_ceiling_usd=primary_cyber_cap,
                upper_layer_attachment_usd=excess_cyber_attach,
                uninsured_gap_usd=gap_amt,
                severity=DivergenceSeverity.CRITICAL,
                explanation=(
                    f"Endorsement CP-14-20 reduced Primary Cyber sublimit to ${primary_cyber_cap:,.0f}, "
                    f"but Chubb 1st Excess (EXC-CHB-2026-4410) does not attach until ${excess_cyber_attach:,.0f}, "
                    f"creating a ${gap_amt:,.0f} uninsured 'drop-down' gap."
                ),
            )
        )

    coinsurance_sim = simulate_coinsurance_penalty(
        reported_tiv_usd=47_500_000.0,
        carried_limit_usd=carried_limit_usd,
        coinsurance_pct=coinsurance_pct,
        simulated_loss_usd=simulated_loss_usd,
    )

    return ProgramTowerAnalysis(
        program_name="Apex Global Logistics — 2026 Commercial P&C Tower",
        total_tower_capacity_usd=sum(layer.limit_usd for layer in layers),
        total_tower_premium_usd=sum(layer.annual_premium_usd for layer in layers),
        layers=layers,
        attachment_gaps=gaps,
        coinsurance_simulation=coinsurance_sim,
    )


def export_reconciliation_csv() -> str:
    """Generates a LineSlip-ready CSV executive report of all Binder-vs-Policy discrepancies."""
    binder = get_sample_broker_binder()
    policy = get_sample_issued_policy_jacket()
    report = reconcile_binder_vs_policy(binder, policy)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "Discrepancy_ID",
            "Severity",
            "Coverage_Code",
            "Coverage_Name",
            "Category",
            "Binder_Value",
            "Policy_Value",
            "Exposure_Delta_USD",
            "Policy_Citation",
            "Remediation_Action",
        ]
    )
    for d in report.discrepancies:
        writer.writerow(
            [
                d.discrepancy_id,
                d.severity.value,
                d.coverage_code,
                d.coverage_name,
                d.category,
                d.binder_value,
                d.policy_value,
                f"{d.financial_exposure_delta_usd:.2f}",
                d.policy_citation,
                d.remediation_action,
            ]
        )
    return buf.getvalue()
