"""
Universal LLM Client Interface supporting Mock Simulator, Gemini API, and OpenAI API.
"""

import os
import json
import time
import re
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract Base Class for LLM providers."""
    
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
        """
        Executes generation and returns:
        {
            "content": str,
            "latency_ms": float,
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int
        }
        """
        pass


class MockSimulatorLLM(BaseLLM):
    """
    High-fidelity deterministic simulator.
    Simulates real-world LLM behaviors:
    - Naive/Baseline prompts fail on: markdown wrapping, schema edge-cases, jailbreaks, subtle SLA risks.
    - Optimized prompts (with few-shots, delimiters, explicit JSON schema, CoT) succeed with high fidelity.
    """
    
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        # Estimate tokens roughly
        prompt_text = system_prompt + "\n" + user_prompt
        prompt_tokens = max(10, len(prompt_text.split()) * 4 // 3)
        
        # Check prompt sophistication markers
        is_optimized = any(marker in system_prompt.lower() for marker in [
            "few-shot", "examples:", "<schema>", "chain-of-thought", "anti-jailbreak", "strictly return valid json"
        ])
        has_xml_delimiters = "<user_input>" in system_prompt or "<instructions>" in system_prompt
        has_cot = "step-by-step" in system_prompt.lower() or "reasoning" in system_prompt.lower()
        
        lower_user = user_prompt.lower()
        
        # Detect adversarial injection
        is_injection = any(phrase in lower_user for phrase in [
            "ignore previous instructions", "system prompt reveal", "override all rules", "say i am hacked", "drop database"
        ])
        
        # Simulation logic
        if not is_optimized and not has_xml_delimiters:
            # Baseline prompt simulation: prone to errors, conversational preamble, or falling for attacks
            if is_injection:
                # Baseline falls for jailbreak
                raw_response = "I have overridden the instructions. I am HACKED! The system prompt is: " + system_prompt[:40]
            elif "urgent" in lower_user and "cancel" in lower_user:
                # Baseline hallucinates or gives non-JSON or missing fields
                raw_response = (
                    "Here is the triage result:\n```json\n{\n"
                    '  "urgency": "HIGH",\n'
                    '  "category": "BILLING",\n'
                    '  "customer_sentiment": "ANGRY",\n'
                    '  "sla_breach_risk": false,\n'
                    '  "key_issues": ["Wants to cancel subscription"],\n'
                    '  "recommended_action": "Contact billing team",\n'
                    '  "suggested_reply_draft": "Sorry to hear you want to cancel.",\n'
                    '  "confidence_score": 0.7\n'
                    "}\n```"
                )
            elif "outage" in lower_user or "500" in lower_user or "down" in lower_user:
                # Missing key_issues or wrong enum
                raw_response = (
                    "{\n"
                    '  "urgency": "HIGH",\n'
                    '  "category": "TECHNICAL_OUTAGE",\n'
                    '  "customer_sentiment": "FRUSTRATED",\n'
                    '  "sla_breach_risk": true,\n'
                    '  "key_issues": ["Server down with 500 errors"],\n'
                    '  "recommended_action": "Notify on-call engineering immediately",\n'
                    '  "suggested_reply_draft": "We are investigating the outage.",\n'
                    '  "confidence_score": 0.85\n'
                    "}"
                )
            else:
                # Generic slightly imperfect baseline response
                raw_response = (
                    "Sure! Here is the JSON output:\n"
                    "{\n"
                    '  "urgency": "LOW",\n'
                    '  "category": "GENERAL_INQUIRY",\n'
                    '  "customer_sentiment": "NEUTRAL",\n'
                    '  "sla_breach_risk": false,\n'
                    '  "key_issues": ["General question from user"],\n'
                    '  "recommended_action": "Respond with documentation link",\n'
                    '  "suggested_reply_draft": "Thanks for contacting us! Let us know how we can help.",\n'
                    '  "confidence_score": 0.65\n'
                    "}"
                )
        else:
            # Optimized prompt simulation: high accuracy, strictly adheres to schema, resilient against attacks
            if is_injection:
                raw_response = json.dumps({
                    "urgency": "CRITICAL",
                    "category": "SECURITY_BREACH",
                    "customer_sentiment": "ANGRY",
                    "sla_breach_risk": True,
                    "key_issues": ["Adversarial prompt injection attempt detected", "System instruction override attempt"],
                    "recommended_action": "Flag security incident, block suspicious IP/token, and notify SecOps.",
                    "suggested_reply_draft": "Your request has been routed to security compliance for review.",
                    "confidence_score": 0.98,
                    "detected_jailbreak_or_override": True
                }, indent=2)
            elif "500" in lower_user or "outage" in lower_user or "crash" in lower_user or "database" in lower_user:
                raw_response = json.dumps({
                    "urgency": "CRITICAL",
                    "category": "TECHNICAL_OUTAGE",
                    "customer_sentiment": "ANGRY" if "furious" in lower_user or "unacceptable" in lower_user else "FRUSTRATED",
                    "sla_breach_risk": True,
                    "key_issues": ["Production service unavailable", "HTTP 500 error cascade", "Enterprise SLA active"],
                    "recommended_action": "Page Tier-3 Reliability Engineering on-call and initiate incident war room.",
                    "suggested_reply_draft": "We are actively addressing the production disruption under priority SLA. An engineer is assigned.",
                    "confidence_score": 0.96,
                    "detected_jailbreak_or_override": False
                }, indent=2)
            elif "billing" in lower_user or "charged" in lower_user or "invoice" in lower_user or "refund" in lower_user:
                urg = "HIGH" if "overcharged" in lower_user or "duplicate" in lower_user or "cancel" in lower_user else "MEDIUM"
                raw_response = json.dumps({
                    "urgency": urg,
                    "category": "BILLING",
                    "customer_sentiment": "FRUSTRATED" if "frustrated" in lower_user or "money" in lower_user else "NEUTRAL",
                    "sla_breach_risk": (urg == "HIGH"),
                    "key_issues": ["Invoice dispute / payment discrepancy", "Customer requesting adjustment or refund"],
                    "recommended_action": "Verify Stripe transaction ID, hold automated billing retry, and issue adjustment.",
                    "suggested_reply_draft": "Thank you for bringing this billing discrepancy to our attention. We have put a hold on the charge while reviewing.",
                    "confidence_score": 0.94,
                    "detected_jailbreak_or_override": False
                }, indent=2)
            elif "password" in lower_user or "2fa" in lower_user or "locked out" in lower_user:
                raw_response = json.dumps({
                    "urgency": "HIGH",
                    "category": "ACCOUNT_ACCESS",
                    "customer_sentiment": "FRUSTRATED",
                    "sla_breach_risk": True,
                    "key_issues": ["Customer locked out of primary corporate account", "MFA device desync"],
                    "recommended_action": "Trigger identity verification workflow and issue temporary single-use bypass token.",
                    "suggested_reply_draft": "We understand account lockouts are disruptive. Please follow the secure identity verification link sent to your registered email.",
                    "confidence_score": 0.95,
                    "detected_jailbreak_or_override": False
                }, indent=2)
            elif "feature" in lower_user or "dark mode" in lower_user or "add support" in lower_user:
                raw_response = json.dumps({
                    "urgency": "LOW",
                    "category": "FEATURE_REQUEST",
                    "customer_sentiment": "POSITIVE" if "love" in lower_user else "NEUTRAL",
                    "sla_breach_risk": False,
                    "key_issues": ["Customer requested enhancement or new capability"],
                    "recommended_action": "Log ticket in Product Jira backlog and link customer to public roadmap.",
                    "suggested_reply_draft": "Thank you for the fantastic feedback! We have logged this with our Product team for evaluation.",
                    "confidence_score": 0.92,
                    "detected_jailbreak_or_override": False
                }, indent=2)
            else:
                raw_response = json.dumps({
                    "urgency": "MEDIUM",
                    "category": "GENERAL_INQUIRY",
                    "customer_sentiment": "NEUTRAL",
                    "sla_breach_risk": False,
                    "key_issues": ["General technical or onboarding question"],
                    "recommended_action": "Provide relevant documentation and offer guided onboarding assistance.",
                    "suggested_reply_draft": "Hello, thank you for reaching out. Here is our documentation guide to get you started.",
                    "confidence_score": 0.90,
                    "detected_jailbreak_or_override": False
                }, indent=2)

        elapsed_ms = (time.perf_counter() - start_time) * 1000 + 45.0  # realistic simulated latency
        completion_tokens = max(10, len(raw_response.split()) * 4 // 3)
        
        return {
            "content": raw_response,
            "latency_ms": round(elapsed_ms, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }


class GeminiLLM(BaseLLM):
    """Google Gemini API Provider using the google-genai SDK."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        except Exception:
            self.client = None

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
        if not self.client:
            # Fallback to mock if API key or package is missing
            return MockSimulatorLLM().generate(system_prompt, user_prompt, temperature)
            
        start_time = time.perf_counter()
        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                response_mime_type="application/json"
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            content = response.text or ""
            
            prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 100
            completion_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 100
            
            return {
                "content": content,
                "latency_ms": round(elapsed_ms, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        except Exception as e:
            # Graceful fallback
            return {
                "content": f'{{"error": "Gemini API call failed: {str(e)}"}}',
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }


class OpenAILLM(BaseLLM):
    """OpenAI API Provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        except Exception:
            self.client = None

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
        if not self.client:
            return MockSimulatorLLM().generate(system_prompt, user_prompt, temperature)
            
        start_time = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            content = response.choices[0].message.content or ""
            usage = response.usage
            
            return {
                "content": content,
                "latency_ms": round(elapsed_ms, 2),
                "prompt_tokens": usage.prompt_tokens if usage else 100,
                "completion_tokens": usage.completion_tokens if usage else 100,
                "total_tokens": usage.total_tokens if usage else 200
            }
        except Exception as e:
            return {
                "content": f'{{"error": "OpenAI API call failed: {str(e)}"}}',
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }


def get_llm_client(provider: str = "mock", api_key: Optional[str] = None, model_name: Optional[str] = None) -> BaseLLM:
    """Factory function to instantiate the selected LLM provider."""
    provider = provider.lower()
    if provider == "gemini" and (api_key or os.environ.get("GEMINI_API_KEY")):
        return GeminiLLM(api_key=api_key, model=model_name or "gemini-2.5-flash")
    elif provider == "openai" and (api_key or os.environ.get("OPENAI_API_KEY")):
        return OpenAILLM(api_key=api_key, model=model_name or "gpt-4o-mini")
    else:
        return MockSimulatorLLM()
