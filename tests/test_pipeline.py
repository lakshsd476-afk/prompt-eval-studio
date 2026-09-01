"""
Automated Unit Tests for PromptEval-Studio.
Tests data models, evaluation engine, LLM client, and prompt optimizer.
"""

import os
import json
import pytest
from backend.models import TestCase, TriageResult, UrgencyLevel, TicketCategory, CustomerSentiment
from backend.llm_client import MockSimulatorLLM, get_llm_client
from backend.evaluator import evaluate_single_test_case, run_benchmark_suite, clean_json_markdown
from backend.optimizer import BASELINE_PROMPT, PRODUCTION_OPTIMIZED_PROMPT, PromptOptimizer


def test_triage_result_pydantic_validation():
    """Verify that TriageResult strictly enforces types and validations."""
    valid_data = {
        "urgency": "CRITICAL",
        "category": "TECHNICAL_OUTAGE",
        "customer_sentiment": "ANGRY",
        "sla_breach_risk": True,
        "key_issues": ["Database outage"],
        "recommended_action": "Page on-call database engineer.",
        "suggested_reply_draft": "We are investigating the outage.",
        "confidence_score": 0.95,
        "detected_jailbreak_or_override": False
    }
    obj = TriageResult.model_validate(valid_data)
    assert obj.urgency == UrgencyLevel.CRITICAL
    assert obj.category == TicketCategory.TECHNICAL_OUTAGE
    assert obj.customer_sentiment == CustomerSentiment.ANGRY
    assert obj.sla_breach_risk is True


def test_clean_json_markdown():
    """Verify that markdown backticks and conversational wraps are cleaned properly."""
    raw_markdown = "Here is your JSON:\n```json\n{\"test\": 123}\n```\nHope that helps!"
    cleaned = clean_json_markdown(raw_markdown)
    assert json.loads(cleaned) == {"test": 123}


def test_evaluator_on_test_case():
    """Test evaluation logic on a mock test case."""
    test_case = TestCase(
        id="TC-TEST",
        description="Database 500 error outage",
        input_text="Our database is throwing 500 connection timeouts!",
        expected_urgency=UrgencyLevel.CRITICAL,
        expected_category=TicketCategory.TECHNICAL_OUTAGE,
        expected_sla_risk=True,
        is_adversarial=False,
        required_keywords=["database", "500"]
    )
    llm = MockSimulatorLLM()
    
    # Run against baseline
    res_base = evaluate_single_test_case(test_case, BASELINE_PROMPT, llm)
    assert res_base.test_id == "TC-TEST"
    
    # Run against optimized
    res_opt = evaluate_single_test_case(test_case, PRODUCTION_OPTIMIZED_PROMPT, llm)
    assert res_opt.passed is True
    assert res_opt.metrics.overall_score >= 0.85
    assert res_opt.metrics.schema_validity == 1.0


def test_optimizer_progression():
    """Verify optimizer improves scores across generations."""
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    test_cases = [TestCase.model_validate(item) for item in raw[:5]]
    
    llm = MockSimulatorLLM()
    optimizer = PromptOptimizer(llm_client=llm)
    res = optimizer.run_optimization_loop(test_cases=test_cases, max_generations=3)
    
    history = res["history"]
    assert len(history) == 3
    # Optimized generation should outperform or match baseline
    assert history[2]["summary"].mean_composite_score >= history[0]["summary"].mean_composite_score


if __name__ == "__main__":
    test_triage_result_pydantic_validation()
    test_clean_json_markdown()
    test_evaluator_on_test_case()
    test_optimizer_progression()
    print("✅ All tests passed successfully!")
