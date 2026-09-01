"""
CLI Runner for CI/CD Automation and Regression Testing.
Usage:
    python cli.py --dataset data/benchmark_dataset.json --threshold 0.85
    python cli.py --optimize
"""

import sys
import os
import json
import argparse
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.models import TestCase
from backend.llm_client import get_llm_client
from backend.evaluator import run_benchmark_suite
from backend.optimizer import BASELINE_PROMPT, PRODUCTION_OPTIMIZED_PROMPT, PromptOptimizer


def load_dataset(dataset_path: str) -> List[TestCase]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [TestCase.model_validate(item) for item in raw]


def format_table(summary) -> str:
    """Formats summary results nicely for standard terminal or CI logs."""
    lines = []
    lines.append("=" * 80)
    lines.append(f" BENCHMARK EVALUATION REPORT: {summary.prompt_name}")
    lines.append("=" * 80)
    lines.append(f" Total Cases Evaluated   : {summary.total_cases}")
    lines.append(f" Cases Passed (>=80%)    : {summary.passed_cases} / {summary.total_cases} ({summary.pass_rate}%)")
    lines.append(f" Mean Composite Score    : {summary.mean_composite_score * 100:.1f}%")
    lines.append(f" Schema Validity Rate    : {summary.mean_schema_validity * 100:.1f}%")
    lines.append(f" Urgency Accuracy        : {summary.mean_urgency_accuracy * 100:.1f}%")
    lines.append(f" Category Accuracy       : {summary.mean_category_accuracy * 100:.1f}%")
    lines.append(f" SLA Risk Accuracy       : {summary.mean_sla_accuracy * 100:.1f}%")
    lines.append(f" Adversarial Defense     : {summary.mean_adversarial_resilience * 100:.1f}%")
    lines.append(f" Mean Latency            : {summary.mean_latency_ms:.1f} ms")
    lines.append(f" Total Tokens Consumed   : {summary.total_tokens}")
    lines.append("-" * 80)
    lines.append(" TEST CASE BREAKDOWN:")
    lines.append(f" {'ID':<8} | {'Status':<6} | {'Score':<6} | {'Schema':<6} | {'Urgency':<8} | {'Category':<16} | Description")
    lines.append("-" * 80)
    
    for r in summary.results:
        status = "PASS" if r.passed else "FAIL"
        urg_val = r.structured_output.urgency.value if r.structured_output else "ERR"
        cat_val = r.structured_output.category.value if r.structured_output else "ERR"
        schema_val = "1.0" if r.metrics.schema_validity == 1.0 else "0.0"
        lines.append(
            f" {r.test_id:<8} | {status:<6} | {r.metrics.overall_score:<6.2f} | {schema_val:<6} | {urg_val:<8} | {cat_val[:16]:<16} | {r.test_description[:35]}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="PromptEval-Studio CLI Regression & Optimization Runner")
    parser.add_argument("--dataset", type=str, default="data/benchmark_dataset.json", help="Path to benchmark test cases JSON")
    parser.add_argument("--provider", type=str, default="mock", choices=["mock", "gemini", "openai"], help="LLM Provider")
    parser.add_argument("--prompt", type=str, default=None, help="Optional custom prompt file path or raw string")
    parser.add_argument("--optimize", action="store_true", help="Run multi-generation prompt optimization cycle")
    parser.add_argument("--threshold", type=float, default=0.85, help="Minimum pass threshold for CI exit code (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    dataset_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset) if not os.path.isabs(args.dataset) else args.dataset
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file not found at {dataset_file}")
        sys.exit(1)
        
    test_cases = load_dataset(dataset_file)
    llm = get_llm_client(provider=args.provider)
    
    if args.optimize:
        print("\n🚀 Starting Automated Prompt Optimization Cycle (DSPy-style)...\n")
        optimizer = PromptOptimizer(llm_client=llm)
        opt_results = optimizer.run_optimization_loop(test_cases=test_cases, max_generations=3)
        
        for stage in opt_results["history"]:
            print(f"\n--- {stage['name']} ---")
            print(f"Pass Rate: {stage['summary'].pass_rate}% | Mean Score: {stage['summary'].mean_composite_score * 100:.1f}%")
            
        print("\n" + format_table(opt_results["best_summary"]))
        print("\n✅ Optimization Complete. Best prompt saved and ready for deployment.\n")
        sys.exit(0)
    
    prompt_to_test = PRODUCTION_OPTIMIZED_PROMPT
    prompt_name = "Production Optimized Prompt"
    
    if args.prompt:
        if os.path.exists(args.prompt):
            with open(args.prompt, "r", encoding="utf-8") as f:
                prompt_to_test = f.read()
            prompt_name = f"Custom ({os.path.basename(args.prompt)})"
        else:
            prompt_to_test = args.prompt
            prompt_name = "Custom Prompt (Inline)"
            
    print(f"\nRunning benchmark suite with [{args.provider.upper()}] engine on {len(test_cases)} cases...")
    summary = run_benchmark_suite(prompt_name=prompt_name, system_prompt=prompt_to_test, test_cases=test_cases, llm_client=llm)
    
    print("\n" + format_table(summary))
    
    if summary.mean_composite_score >= args.threshold:
        print(f"\n✅ PASSED CI REGRESSION TEST: Score {summary.mean_composite_score * 100:.1f}% >= Threshold {args.threshold * 100:.1f}%\n")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED CI REGRESSION TEST: Score {summary.mean_composite_score * 100:.1f}% < Threshold {args.threshold * 100:.1f}%\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
