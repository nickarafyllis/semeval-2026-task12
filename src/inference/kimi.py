"""
Kimi (Moonshot AI) inference implementation

KimiInference: Uses <reasoning> tag extraction with standard prompt format
Similar to DeepSeek but with Kimi-specific response format
"""

import re
from typing import Dict, List, Any
from .base import BaseInference


def _parse_answer_letters(text: str) -> List[str]:
    """
    Extract valid answer letters A/B/C/D from a free-form string.
    Accepts comma/space/newline separated forms like "A", "A,B", "B C", "A\nB".
    """
    if not isinstance(text, str):
        return []
    letters = [
        s.strip().upper()
        for s in re.split(r"[, \n]+", text)
        if s.strip().upper() in {"A", "B", "C", "D"}
    ]
    return letters


class KimiInference(BaseInference):
    """
    Kimi K2 Thinking inference with reasoning extraction.
    Produces predictions, analyses, and reasoning via tag extraction.
    """

    def build_enforce_schema_retry_suffix(self, retry_idx: int) -> str:
        """Add stricter schema enforcement on retries."""
        if retry_idx == 0:
            return ""
        return (
            "\n\nCRITICAL: You MUST output your final answer in EXACTLY this format:\n"
            "<analysis>[Your reasoning for each option]</analysis>\n"
            "<answer>[One or more letters: A, B, C, or D separated by commas]</answer>\n"
            "Do NOT include 'Option' before letters. Just the letter(s)."
        )

    def format_prompt(self, question: Dict, context_docs: List[Dict]) -> str:
        """Plain text prompt for Kimi with analysis/answer tags."""
        context_text = "\n\n".join([doc["content"] for doc in context_docs])

        options_text = "\n".join([
            f"{opt}. {question[f'option_{opt}']}"
            for opt in ["A", "B", "C", "D"]
        ])

        return f"""You are a careful reasoning assistant for abductive event reasoning.

Context:
{context_text}

Target Event: {question["target_event"]}

Options:
{options_text}

Respond in this exact format:

<analysis>
[your analysis of each option and the causal reasoning]
</analysis>

<answer>
[Letter(s): A, B, C, or D; multiple separated by commas]
</answer>
"""

    def call_model(self, prompt: str) -> Dict:
        """
        Call Kimi client.

        Returns dict with 'thinking' and 'answer' keys.
        """
        result = self.chat.generate_isolated(prompt)

        # Ensure it returns a dict
        if isinstance(result, str):
            return {"thinking": "", "answer": result}

        return result

    def parse_response(self, response: Dict) -> Dict:
        """Extract reasoning, analysis, and answer from response."""
        thinking = response.get("thinking", "")
        answer_text = response.get("answer", "")

        if "ERROR::" in answer_text:
            return {
                "answer": ["C"],
                "analysis": answer_text,
                "thinking": thinking,
                "flag": True
            }

        # Extract analysis tag
        analysis = ""
        m = re.search(
            r"<analysis>(.*?)</analysis>",
            answer_text,
            re.DOTALL | re.IGNORECASE
        )
        if m:
            analysis = m.group(1).strip()
        else:
            # Heuristic fallback
            analysis = answer_text.strip()[:400]  # First 400 chars

        # Extract answer tag
        answer_letters = []
        m2 = re.search(
            r"<answer>(.*?)</answer>",
            answer_text,
            re.DOTALL | re.IGNORECASE
        )
        if m2:
            answer_text_inner = m2.group(1).strip()
            answer_letters = _parse_answer_letters(answer_text_inner)

        # Fallback: scan for letters
        if not answer_letters:
            found = re.findall(r"\b([A-D])\b", answer_text)
            answer_letters = list(dict.fromkeys([x.upper() for x in found]))

        if not answer_letters:
            answer_letters = ["C"]  # default fallback

        return {
            "answer": answer_letters,
            "analysis": analysis,
            "thinking": thinking,
            "flag": False
        }

    def supports_thinking(self) -> bool:
        return True

    def get_progress_color(self) -> str:
        return "cyan"

    def get_model_name(self) -> str:
        """Override to show correct name in progress bar."""
        return "Kimi"


def run_kimi_inference(
    chat_client,
    questions: List[Dict],
    docs: List[Dict],
    sleep_seconds: int = 0,
    experiment_path: str = None
) -> Dict[str, Any]:
    """
    Run Kimi inference with automatic retry.
    Uses BaseInference template method pattern.
    """
    print("   → KimiInference")
    inference = KimiInference(chat_client, experiment_path=experiment_path)

    # Run with BaseInference framework (handles retry, progress)
    return inference.run(questions, docs, sleep_seconds)
