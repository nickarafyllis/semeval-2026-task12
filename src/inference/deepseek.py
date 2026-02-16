"""
DeepSeek inference implementations (class-based, fixed)

- DeepSeekR1Inference: R1 with <think> reasoning extraction, XML-structured prompts.
- DeepSeekV31Inference: V3.1 with prompt caching and standard <analysis>/<answer> parsing.

Both implementations:
- format_prompt(...) returns a STRING (DeepSeek expects a single string prompt).
- call_model(...) calls self.chat.generate_isolated(prompt: str).
- parse_response(...) always returns a dict with at least: {'answer': [...], 'analysis': str, 'raw_response': str, 'flag': bool}.
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


class DeepSeekR1Inference(BaseInference):
    """
    DeepSeek R1 inference with pure text parsing (no tool use).
    Produces predictions, analyses, and thinkings via tag extraction.
    """

    def __init__(self, chat_client, experiment_path: str = None):
        super().__init__(chat_client, experiment_path=experiment_path)

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
        """Plain text prompt tailored for R1 with analysis/answer tags."""
        context_text = "\n\n".join([doc["content"] for doc in context_docs])

        options_text = "\n".join([
            f"{opt}. {question[f'option_{opt}']}"
            for opt in ["A", "B", "C", "D"]
        ])

        return f"""You are a careful reasoning assistant. Think step by step in <think> tags, then produce analysis and answer tags.

Context:
{context_text}

Target Event: {question["target_event"]}

Options:
{options_text}

Respond in this exact format:

<think>
[your detailed chain-of-thought]
</think>

<analysis>
[your analysis of each option and the causal reasoning]
</analysis>

<answer>
[Letter(s): A, B, C, or D; multiple separated by commas]
</answer>
"""

    def call_model(self, prompt: str) -> Dict:
        """
        Call DeepSeek R1 pure text client.
        
        Returns dict with 'raw_response' key.
        """
        result = self.chat.generate_isolated(prompt)

        # Ensure it returns a dict (not string)
        if isinstance(result, str):
            return {"raw_response": result}

        return result


    def parse_response(self, response: Dict) -> Dict:
        """Extract thinking, analysis, and answer from raw text."""
        raw = response.get("raw_response", "")
        if raw.startswith("ERROR::"):
            return {
                "answer": ["C"],
                "analysis": raw,
                "thinking": "",
                "flag": True
            }

        # 1) Split thinking vs final
        splitted = self.chat.split_thinking_and_answer(raw)
        thinking = splitted["thinking"]
        final_text = splitted["answer_text"]

        # 2) Extract analysis and answer tags
        # Analysis
        analysis = ""
        m = re.search(r"<analysis>(.*?)</analysis>", final_text, re.DOTALL | re.IGNORECASE)
        if m:
            analysis = m.group(1).strip()
        else:
            # Heuristic fallback
            analysis = final_text.strip()

        # Answer - Try to extract from <answer> tags first
        answer_letters = []
        m2 = re.search(r"<answer>(.*?)</answer>", final_text, re.DOTALL | re.IGNORECASE)
        if m2:
            answer_text = m2.group(1).strip()
            answer_letters = [t.strip().upper() for t in re.split(r"[,\s]+", answer_text) if t.strip().upper() in ["A", "B", "C", "D"]]

        # Check if we got a confident parse (has <answer> tags with valid letters)
        confident_parse = bool(m2 and answer_letters)

        # Fallback: Look for explicit answer patterns (but flag for retry)
        if not answer_letters:
            # Try explicit patterns like "Answer: B" or "The answer is B"
            answer_patterns = [
                r"(?:the\s+)?(?:correct\s+)?answer(?:\s+is)?[:\s]+([A-D](?:\s*[,&]\s*[A-D])*)",
                r"(?:^|\n)\s*([A-D](?:\s*,\s*[A-D])*)\s*$",  # Standalone at end
            ]
            for pattern in answer_patterns:
                match = re.search(pattern, final_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    found_text = match.group(1)
                    answer_letters = [t.strip().upper() for t in re.split(r"[,&\s]+", found_text) if t.strip().upper() in ["A", "B", "C", "D"]]
                    if answer_letters:
                        break

        # If still no answer, set flag to trigger retry with schema enforcement
        if not answer_letters:
            return {
                "answer": ["C"],  # temporary fallback
                "analysis": analysis,
                "thinking": thinking,
                "flag": True  # Signal to retry with stricter schema
            }

        return {
            "answer": answer_letters,
            "analysis": analysis,
            "thinking": thinking,
            "flag": not confident_parse  # Flag if we didn't get clean <answer> tags
        }

    def supports_thinking(self) -> bool:
        return True

    def supports_caching(self) -> bool:
        return True

    def get_progress_color(self) -> str:
        return "yellow"

    def get_model_name(self) -> str:
        """Override to show correct name in progress bar."""
        return "DeepSeekR1"



class DeepSeekV31Inference(BaseInference):
    """
    DeepSeek V3.1 inference with reasoning extraction from reasoningContent.
    Uses force_reasoning=True for thinking, then parses analysis/answer from final text.
    """
    def __init__(self, chat_client, experiment_path: str = None):
        super().__init__(chat_client, experiment_path=experiment_path)
        self.reasoning_effort = "high"

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
        """Build XML-structured prompt (like your context caching)."""
        # Build context (like your context_text)
        context_text = "\n".join([
            f"<document_{j+1}>{doc.get('content', '')}</document_{j+1}>"
            for j, doc in enumerate(context_docs)
        ])

        return f"""<context_documents>
<topic_id>{question.get('topic_id', '')}</topic_id>
{context_text}
</context_documents>

<target_event>{question.get("target_event", "")}</target_event>

<options>
<option_a>{question.get("option_A", "")}</option_a>
<option_b>{question.get("option_B", "")}</option_b>
<option_c>{question.get("option_C", "")}</option_c>
<option_d>{question.get("option_D", "")}</option_d>
</options>

Analyze which option(s) are plausible causes. Think step-by-step, then provide analysis and answer."""

    def call_model(self, prompt: str) -> Dict:
        """Call client with force_reasoning=True (returns {'thinking', 'answer', 'flag'})."""
        # Like your deepseek_v31_abductive_reasoning, but without retry (handle in run())
        return self.chat.generate_isolated(prompt,  reasoning_effort=self.reasoning_effort)

    def parse_response(self, response: Dict) -> Dict[str, Any]:
        """
        Parse response: thinking from 'thinking', analysis/answer from 'answer' text.
        
        Handles error flag, extracts <analysis> and <answer> tags (or fallbacks).
        """
        if not isinstance(response, dict):
            response = {"thinking": "", "answer": str(response), "flag": True}

        thinking = response.get("thinking", "")
        answer_text = response.get("answer", "")
        client_flag = response.get("flag", False)

        raw_response = f"{thinking}\n{answer_text}" if thinking else answer_text

        if "ERROR::" in answer_text:
            return {
                "answer": ["C"],
                "analysis": answer_text,
                "thinking": thinking,
                "raw_response": raw_response,
                "flag": True
            }

        # Extract analysis from answer_text
        analysis_match = re.search(r"<analysis>(.*?)</analysis>", answer_text, re.DOTALL | re.IGNORECASE)
        analysis = analysis_match.group(1).strip() if analysis_match else ""

        # Fallback for analysis (like your enforce_schema)
        if not analysis:
            analysis_fallback = re.search(r"Analysis:\s*(.*?)(Answer:|$)", answer_text, re.DOTALL | re.IGNORECASE)
            analysis = analysis_fallback.group(1).strip() if analysis_fallback else answer_text[:400]  # First 400 chars

        # Extract answer letters from answer_text
        answer_letters = []
        answer_match = re.search(r"<answer>(.*?)</answer>", answer_text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer_letters = _parse_answer_letters(answer_match.group(1).strip())
        else:
            # Fallback: look for letters in "Answer:" or standalone
            answer_pattern = re.search(r"Answer:\s*([A-D](?:\s*,\s*[A-D])*)", answer_text, re.IGNORECASE)
            if answer_pattern:
                answer_letters = _parse_answer_letters(answer_pattern.group(1))
            else:
                found = re.findall(r"\b([A-D])\b", answer_text)
                answer_letters = list(dict.fromkeys([x.upper() for x in found]))

        if not answer_letters:
            answer_letters = ["C"]

        # Flag if client error or missing key parts
        inference_flag = client_flag or not answer_letters or not analysis

        result = {
            "answer": answer_letters,
            "analysis": analysis,
            "thinking": thinking,
            "raw_response": raw_response,
            "flag": inference_flag
        }

        print(f"[DeepSeekV31Inference] ✅ Parsed: thinking_len={len(thinking)}, analysis_len={len(analysis)}, answer={answer_letters}")
        return result

    def supports_thinking(self) -> bool:
        return True

    def supports_caching(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return "DeepSeekV31"

    def get_progress_color(self) -> str:
        return "magenta"

def run_deepseek_inference(
    chat_client,
    questions: List[Dict],
    docs: List[Dict],
    version: str = "v3.1",
    sleep_seconds: int = 0,
    reasoning_effort: str = "off",
    experiment_path: str = None
) -> Dict[str, Any]:
    """
    Run DeepSeek inference with automatic retry and caching.
    Uses BaseInference template method pattern.
    """
    norm = (version or "").strip().lower()
    print(f"🔍 DeepSeek version: '{version}' → '{norm}'")

    # Select inference class
    if norm in {"r1", "deepseek-r1"}:
        print("   → DeepSeekR1Inference")
        inference = DeepSeekR1Inference(chat_client, experiment_path=experiment_path)
    elif norm in {"v3.1", "v31", "deepseek-v3.1", "deepseek-v31"}:
        print("   → DeepSeekV31Inference")
        inference = DeepSeekV31Inference(chat_client, experiment_path=experiment_path)
        inference.reasoning_effort = reasoning_effort
    else:
        raise ValueError(f"Unknown version '{version}'")

    # Run with BaseInference framework (handles retry, caching, progress)

    return inference.run(questions, docs, sleep_seconds)
