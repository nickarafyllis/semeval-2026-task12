"""
Llama inference implementation

Single Llama variant using the base class - only ~50 lines!
"""

import re
from typing import Dict, List
from .base import BaseInference


# ============================================================================
# LLAMA IMPLEMENTATION
# ============================================================================

class LlamaInference(BaseInference):
    """
    Llama inference with instruction formatting.

    Use for: Running Llama models (3.2, 3.3, etc.)
    Corresponds to: Cell 52 (llama_abductive_reasoning)
    """

    def build_enforce_schema_retry_suffix(self, retry_idx: int) -> str:
        """Add stricter schema enforcement on retries."""
        if retry_idx == 0:
            return ""
        return (
            "\n\nCRITICAL: You MUST output your final answer in EXACTLY this format:\n"
            "Analysis: [your reasoning]\n"
            "Answer: [One or more letters: A, B, C, or D separated by commas]\n"
            "Do NOT include 'Option' before letters. Just the letter(s)."
        )

    def format_prompt(self, question: Dict, context_docs: List[Dict]) -> str:
        """Format prompt in Llama instruction format."""
        # Format context (plain text, no XML for Llama)
        context_text = "\n\n".join([doc['content'] for doc in context_docs])

        # Format options
        options_text = "\n".join([
            f"{opt}. {question[f'option_{opt}']}"
            for opt in ["A", "B", "C", "D"]
        ])

        # Llama instruction format
        instruction = f"""Given the following context documents and a target event, determine which option(s) are plausible causes.

Context:
{context_text}

Target Event: {question["target_event"]}

Options:
{options_text}

Analyze the causal relationships and provide:
1. Your analysis of each option
2. Your final answer as a letter (A, B, C, or D) or multiple letters separated by commas

Format your response as:
Analysis: [your reasoning]
Answer: [letter(s)]
"""

        # Wrap in Llama format if system prompt exists
        if self.system_prompt:
            return f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{instruction} [/INST]"
        else:
            return f"<s>[INST] {instruction} [/INST]"

    def call_model(self, prompt: str) -> str:
        """Call Llama model."""
        return self.chat.generate_isolated(prompt)

    def parse_response(self, response: str) -> Dict:
        """Parse Llama plain text response."""
        # Extract analysis
        analysis_match = re.search(
            r'Analysis:(.*)Answer:', 
            response,
            re.DOTALL | re.IGNORECASE
        )
        analysis = analysis_match.group(1).strip() if analysis_match else response

        # Extract answer
        answer_match = re.search(
            r'Answer:\s*([A-D,\s]+)', 
            response,
            re.IGNORECASE
        )

        if answer_match:
            answer_text = answer_match.group(1).strip()
            # Parse letters
            answer_letters = [
                letter.strip().upper()
                for letter in re.split(r'[,\s]+', answer_text)
                if letter.strip() and letter.strip().upper() in ['A', 'B', 'C', 'D']
            ]
        else:
            # Fallback: look for any A, B, C, D in the response
            found_letters = re.findall(r'\b([A-D])\b', response)
            answer_letters = list(dict.fromkeys(found_letters))  # Remove duplicates, keep order

        if not answer_letters:
            answer_letters = ['C']  # Final fallback

        return {
            'answer': answer_letters,
            'analysis': analysis,
            'raw_response': response,
            'flag': len(answer_letters) == 0
        }

    def get_progress_color(self) -> str:
        return "green"


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_llama_inference(chat_client, questions: List[Dict], docs: List[Dict],
                       sleep_seconds: int = 0, experiment_path: str = None) -> Dict:
    """
    Convenience function to run Llama inference.

    Args:
        chat_client: ChatLlama instance
        questions: List of questions
        docs: List of documents
        sleep_seconds: Rate limiting delay
        experiment_path: Optional path for incremental saving

    Returns:
        Dictionary with predictions and analyses

    Example:
        >>> from src.models.llm_clients import ChatLlama
        >>> from configs.aws_config import get_bedrock_client, get_model_id
        >>>
        >>> bedrock_client = get_bedrock_client()
        >>> model_id = get_model_id("llama-3.3-70b")
        >>> chat = ChatLlama(model_id, bedrock_client, system_prompt)
        >>>
        >>> results = run_llama_inference(chat, questions, docs)
        >>> predictions = results['predictions']
    """
    inference = LlamaInference(chat_client, experiment_path=experiment_path)
    return inference.run(questions, docs, sleep_seconds)
