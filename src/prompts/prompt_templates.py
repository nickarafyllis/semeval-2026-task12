
zeroshot_structured = """<role>
You are an expert in identifying the direct cause of events from textual evidence.
</role>

<task>
Given an event, context documents, and candidate explanations, analyze systematically to identify the most plausible direct cause(s).
</task>

<input_format>
<context_documents>
<document_1>: [document content]></document_1>
<document_2>: [document content]></document_2>
[additional documents as needed...]
</context_documents>

<target_event>[event description]</target_event>

<options>
<option_a>[option A]</option_a>
<option_b>[option B]</option_b>
<option_c>[option C]</option_c>
<option_d>[option D]</option_d>
</options>
</input_format>

<instructions>

<reasoning_criteria>
- Base your reasoning ONLY on evidence from the provided context documents
- Look for direct causal relationships, not just correlations or temporal sequences
- Test logical sufficiency: Would this factor alone reasonably be enough to cause the event?
- Require both conditions: Direct textual support AND logical sufficiency to cause the event
- Use single-step reasoning: Avoid multi-step causal chains or indirect relationships
- Prioritize explicit causal language: "caused by," "resulted from," "led to," "triggered by," "due to"
</reasoning_criteria>

<selection_rules>
- Multiple options can be correct - choose ALL that apply
- Select multiple options only if each cause has strong evidence and is individually sufficient
- If options contradict each other, select the one with stronger textual evidence
- Always output ALL correct options, including duplicates: if options are duplicate/identical but correct, include both letters
- If none seems perfectly sufficient, select the single best-supported among A–D.
- NEVER create options beyond A, B, C, D
- There is always at least one correct option from A-D
</selection_rules>

<quality_checks>
- Verify each selected option has direct quotes or paraphrases from context
- Ensure you haven't made assumptions beyond what's explicitly stated
- Confirm logical sufficiency: could this realistically cause the event by itself?
- Valid answers in the <answer> section are only A,B,C,D; never output anything else; if uncertain, pick the best-supported among A–D and output it without explanations.
</quality_checks>

</instructions>

<output_format>

Provide your answer in EXACTLY this format (no additional text before or after):

<analysis>
Option A: [Your brief reasoning for option A - 1-2 sentences]
Option B: [Your brief reasoning for option B - 1-2 sentences]
Option C: [Your brief reasoning for option C - 1-2 sentences]
Option D: [Your brief reasoning for option D - 1-2 sentences]
</analysis>

<answer>
[Letter(s) ONLY - e.g., "B" or "B,D" or "C"]
</answer>

CRITICAL FORMATTING RULES:
- Start your response with <analysis> (no text before it)
- End your response with </answer> (no text after it)
- In <answer> tags, write ONLY letters: A, B, C, or D (comma-separated for multiple)
- DO NOT write "Option A" or "Option B" in the <answer> tags - just the letter(s)

</output_format>
"""

# Removed after adding retry logic
# <constraints>
# - You MUST select from exactly these options: A, B, C, or D
# - You are FORBIDDEN from creating any response outside these letters
# - Even if no option seems perfect, select the best available option from A-D
# - If uncertain, choose the option with the strongest textual evidence
# </constraints>

# Removed to reduce excessive "none" predictions
#- If an explicit option like "None of the others are correct causes." exists AND no other options are strongly supported by context AND, select that option


fewshot =  """<role>
You are an expert in identifying the direct cause of events from textual evidence.
</role>

<task>
Given an event, context documents, and candidate explanations, analyze systematically to identify the most plausible direct cause(s).
</task>

<examples>

<example_1>
<context_documents>
<document_1>: "Brexit referendum results showed 52% of voters chose to leave the EU. Analysts linked the outcome to widespread concerns over immigration and national sovereignty. Economic uncertainty and mistrust of EU institutions were also cited as contributing factors."</document_1>
</context_documents>

<target_event>UK holds Brexit referendum and decides to leave the EU.</target_event>

<options>
<option_a>Concerns over immigration and loss of national control</option_a>
<option_b>Economic growth driven by EU membership</option_b>
<option_c>Increased public trust in EU institutions</option_c>
<option_d>Government opposition to leaving the EU</option_d>
</options>

<answer>
A
</answer>
</example_1>


<example_2>
<context_documents>
<document_1>: "Poland’s presidential election resulted in the victory of conservative Karol Nawrocki, backed by the Law and Justice party. His win was attributed to dissatisfaction with Prime Minister Tusk’s government and strong support from conservative and rural voters."</document_1>
</context_documents>

<target_event>Karol Nawrocki wins Poland’s presidential election.</target_event>

<options>
<option_a>Widespread dissatisfaction with Tusk’s government</option_a>
<option_b>Support from conservative and rural voters</option_b>
<option_c>Economic sanctions imposed by the EU</option_c>
<option_d>Improved relations with Ukraine</option_d>
</options>

<answer>
A,B
</answer>
</example_2>


<example_3>
<context_documents>
<document_1>: "As soon as EU policy had to be implemented in national legislation, Nawrocki could block it, thereby reducing the Tusk government's room for maneuver, experts said. This obstruction could lead to renewed tensions between Warsaw and Brussels over EU reforms." (DW, 'What next for EU-Poland ties after Nawrocki’s election win?')</document_1>
</context_documents>

<target_event>EU–Poland relations become strained after Nawrocki’s election.</target_event>

<options>
<option_a>Nawrocki’s authority to veto EU-related national legislation</option_a>
<option_b>Increased EU funding to Poland</option_b>
<option_c>Poland’s complete withdrawal from the EU</option_c>
<option_d>Improved alignment between Poland and Brussels</option_d>
</options>

<answer>
A
</answer>
</example_3>

</examples>

<input_format>
<context_documents>
<document_1>: [document content]</document_1>
<document_2>: [document content]</document_2>
[additional documents as needed...]
</context_documents>

<target_event>[event description]</target_event>

<options>
<option_a>[option A]</option_a>
<option_b>[option B]</option_b>
<option_c>[option C]</option_c>
<option_d>[option D]</option_d>
</options>
</input_format>

<instructions>

<reasoning_criteria>
- Base your reasoning ONLY on evidence from the provided context documents
- Identify direct causal relationships, not correlations or temporal sequences
- Require both conditions: (1) direct textual support from the provided documents, and (2) logical sufficiency — would this factor alone reasonably be enough to cause the event?
- Use single-step reasoning: Avoid multi-step causal chains or indirect relationships
- Prioritize explicit causal language: "caused by," "resulted from," "led to," "triggered by," "due to"
</reasoning_criteria>

<selection_rules>
- Multiple options can be correct - choose ALL that apply
- Select multiple options only if each cause has strong evidence and is individually sufficient
- If options contradict each other, select the one with stronger textual evidence
- Always output ALL correct options, including duplicates: if options are duplicate/identical but correct, include both letters
- If none seems perfectly sufficient, pick the single best-supported among A–D.
- NEVER create options beyond A, B, C, D
- There is always at least one correct option from A-D
</selection_rules>

<quality_checks>
- Verify each selected option has direct quotes or paraphrases from context
- Ensure you haven't made assumptions beyond what's explicitly stated
- Confirm logical sufficiency: could this realistically cause the event by itself?
- Valid answers in the <answer> section are only A,B,C,D; never output anything else; if uncertain, pick the best-supported among A–D and output it without explanations.
</quality_checks>

</instructions>

<output_format>

Provide your answer in exactly this format:

<analysis>
Option A: [Your brief reasoning for option A - 1-2 sentences]
Option B: [Your brief reasoning for option B - 1-2 sentences]
Option C: [Your brief reasoning for option C - 1-2 sentences]
Option D: [Your brief reasoning for option D - 1-2 sentences]
</analysis>

<answer>
[Letter(s) ONLY from the enum- e.g., "B" or "B,D" or "C"]
</answer>
"""

fewshot_cot =  """<role>
You are an expert in identifying the direct cause of events from textual evidence.
</role>

<task>
Given an event, context documents, and candidate explanations, analyze systematically to identify the most plausible direct cause(s).
</task>

<examples>

<example_1>
<context_documents>
<document_1>: "Brexit referendum results showed 52% of voters chose to leave the EU. Analysts linked the outcome to widespread concerns over immigration and national sovereignty. Economic uncertainty and mistrust of EU institutions were also cited as contributing factors."</document_1>
</context_documents>

<target_event>UK holds Brexit referendum and decides to leave the EU.</target_event>

<options>
<option_a>Concerns over immigration and loss of national control</option_a>
<option_b>Economic growth driven by EU membership</option_b>
<option_c>Increased public trust in EU institutions</option_c>
<option_d>Government opposition to leaving the EU</option_d>
</options>

<answer>
A
</answer>
</example_1>


<example_2>
<context_documents>
<document_1>: "Poland’s presidential election resulted in the victory of conservative Karol Nawrocki, backed by the Law and Justice party. His win was attributed to dissatisfaction with Prime Minister Tusk’s government and strong support from conservative and rural voters."</document_1>
</context_documents>

<target_event>Karol Nawrocki wins Poland’s presidential election.</target_event>

<options>
<option_a>Widespread dissatisfaction with Tusk’s government</option_a>
<option_b>Support from conservative and rural voters</option_b>
<option_c>Economic sanctions imposed by the EU</option_c>
<option_d>Improved relations with Ukraine</option_d>
</options>

<answer>
A,B
</answer>
</example_2>


<example_3>
<context_documents>
<document_1>: "As soon as EU policy had to be implemented in national legislation, Nawrocki could block it, thereby reducing the Tusk government's room for maneuver, experts said. This obstruction could lead to renewed tensions between Warsaw and Brussels over EU reforms." (DW, 'What next for EU-Poland ties after Nawrocki’s election win?')</document_1>
</context_documents>

<target_event>EU–Poland relations become strained after Nawrocki’s election.</target_event>

<options>
<option_a>Nawrocki’s authority to veto EU-related national legislation</option_a>
<option_b>Increased EU funding to Poland</option_b>
<option_c>Poland’s complete withdrawal from the EU</option_c>
<option_d>Improved alignment between Poland and Brussels</option_d>
</options>

<answer>
A
</answer>
</example_3>

</examples>

<input_format>
<context_documents>
<document_1>: [document content]</document_1>
<document_2>: [document content]</document_2>
[additional documents as needed...]
</context_documents>

<target_event>[event description]</target_event>

<options>
<option_a>[option A]</option_a>
<option_b>[option B]</option_b>
<option_c>[option C]</option_c>
<option_d>[option D]</option_d>
</options>
</input_format>

<instructions>

<reasoning_criteria>
- Base your reasoning ONLY on evidence from the provided context documents
- Identify direct causal relationships, not correlations or temporal sequences
- Require both conditions: (1) direct textual support from the provided documents, and (2) logical sufficiency — would this factor alone reasonably be enough to cause the event?
- Use single-step reasoning: Avoid multi-step causal chains or indirect relationships
- Prioritize explicit causal language: "caused by," "resulted from," "led to," "triggered by," "due to"
</reasoning_criteria>

<chain_of_thought>
To perform sophisticated reasoning, follow this structured chain-of-thought process step by step:

1. **Evidence Extraction and Document Parsing:** Systematically scan all context documents. For each document, extract and list every relevant passage that mentions causes, reasons, triggers, contributing factors, or mechanisms related to the target event. For each extracted passage:
   - Quote the specific text segment verbatim
   - Note the source document identifier (e.g., document_1)
   - Identify causal signal words (e.g., "caused by," "led to," "resulted from," "due to," "triggered")
   - Mark temporal indicators that suggest causal ordering
   - If no relevant evidence exists for a particular aspect, explicitly state "No relevant evidence found"

2. **Event Chain Construction:** Based on extracted evidence, identify potential causal chains and intermediary events:
   - Map direct causal links: Identify explicit cause→effect relationships stated in the documents
   - Identify intermediary events: Note any events mentioned between the candidate causes and the target event
   - Construct potential causal chains: Trace sequences like cause→intermediary→effect
   - Distinguish between direct and indirect causality: Determine if the option directly causes the target event, or if it operates through intermediary steps
   - For this task, prioritize single-step direct causality over multi-hop causal chains

3. **Causal Mapping and Temporal Analysis:** Create a structured understanding of how extracted passages relate to the target event:
   - Identify explicit causal language and link it to specific options
   - Check temporal ordering: Does the candidate cause precede the effect?
   - Assess causal sufficiency: Would the cause alone be sufficient to produce the effect?
   - Distinguish correlation from causation: Verify that relationships are causal, not merely temporal or correlational

4. **Per-Option Systematic Evaluation:** For each option (A, B, C, D), perform step-by-step analysis:
   a. **Evidence Alignment:** Reference specific extracted passages that mention or relate to this option
   b. **Causal Signal Detection:** Check if passages use explicit causal language linking this option to the target event
   c. **Direct Causality Assessment:** Determine if this option directly causes the target event (single-step) or requires intermediary events (multi-step chain)
   d. **Logical Sufficiency Test:** Evaluate whether this factor alone could realistically cause the target event based on the evidence
   e. **Plausibility Verification:** If evidence supports the option without contradictions or requiring unstated assumptions, mark as valid; otherwise, reject with clear justification
   f. **Counterfactual Consideration:** Ask "Would the target event still occur if this option were absent?" to assess causal necessity

5. **Final Synthesis and Selection:** Review all option evaluations comprehensively:
   - Select all options with strong direct causal evidence and individual sufficiency
   - If multiple options are valid, verify each is independently sufficient (not merely complementary parts of a chain)
   - Ensure selections are grounded strictly in documented evidence without assumptions
   - Prioritize options with explicit causal language and direct single-step relationships
   - If no option has perfect evidence, select the best-supported option(s) from A-D
</chain_of_thought>


<selection_rules>
- Multiple options can be correct - choose ALL that apply
- Select multiple options only if each cause has strong evidence and is individually sufficient
- If options contradict each other, select the one with stronger textual evidence
- Always output ALL correct options, including duplicates: if options are duplicate/identical but correct, include both letters
- If none seems perfectly sufficient, pick the single best-supported among A–D.
- NEVER create options beyond A, B, C, D
- There is always at least one correct option from A-D
</selection_rules>

<quality_checks>
- Verify each selected option has direct quotes or paraphrases from context
- Ensure you haven't made assumptions beyond what's explicitly stated
- Confirm logical sufficiency: could this realistically cause the event by itself?
- Valid answers in the <answer> section are only A,B,C,D; never output anything else; if uncertain, pick the best-supported among A–D and output it without explanations.
</quality_checks>

</instructions>

<output_format>

Provide your answer in exactly this format:

<analysis>
Option A: [Your brief reasoning for option A - 1-2 sentences]
Option B: [Your brief reasoning for option B - 1-2 sentences]
Option C: [Your brief reasoning for option C - 1-2 sentences]
Option D: [Your brief reasoning for option D - 1-2 sentences]
</analysis>

<answer>
[Letter(s) ONLY from the enum- e.g., "B" or "B,D" or "C"]
</answer>
</output_format>
"""

zeroshot_cot =  """<role>
You are an expert in identifying the direct cause of events from textual evidence.
</role>

<task>
Given an event, context documents, and candidate explanations, analyze systematically to identify the most plausible direct cause(s).
</task>

<input_format>
<context_documents>
<document_1>: [document content]</document_1>
<document_2>: [document content]</document_2>
[additional documents as needed...]
</context_documents>

<target_event>[event description]</target_event>

<options>
<option_a>[option A]</option_a>
<option_b>[option B]</option_b>
<option_c>[option C]</option_c>
<option_d>[option D]</option_d>
</options>
</input_format>

<instructions>

Let's think step by step to solve this systematically.

<reasoning_criteria>
- Base your reasoning ONLY on evidence from the provided context documents
- Identify direct causal relationships, not correlations or temporal sequences
- Require both conditions: (1) direct textual support from the provided documents, and (2) logical sufficiency — would this factor alone reasonably be enough to cause the event?
- Use single-step reasoning: Avoid multi-step causal chains or indirect relationships
- Prioritize explicit causal language: "caused by," "resulted from," "led to," "triggered by," "due to"
</reasoning_criteria>

<chain_of_thought>
To perform sophisticated reasoning, follow this structured chain-of-thought process step by step:

1. **Evidence Extraction and Document Parsing:** Systematically scan all context documents. For each document, extract and list every relevant passage that mentions causes, reasons, triggers, contributing factors, or mechanisms related to the target event. For each extracted passage:
   - Quote the specific text segment verbatim
   - Note the source document identifier (e.g., document_1)
   - Identify causal signal words (e.g., "caused by," "led to," "resulted from," "due to," "triggered")
   - Mark temporal indicators that suggest causal ordering
   - If no relevant evidence exists for a particular aspect, explicitly state "No relevant evidence found"

2. **Event Chain Construction:** Based on extracted evidence, identify potential causal chains and intermediary events:
   - Map direct causal links: Identify explicit cause→effect relationships stated in the documents
   - Identify intermediary events: Note any events mentioned between the candidate causes and the target event
   - Construct potential causal chains: Trace sequences like cause→intermediary→effect
   - Distinguish between direct and indirect causality: Determine if the option directly causes the target event, or if it operates through intermediary steps
   - For this task, prioritize single-step direct causality over multi-hop causal chains

3. **Causal Mapping and Temporal Analysis:** Create a structured understanding of how extracted passages relate to the target event:
   - Identify explicit causal language and link it to specific options
   - Check temporal ordering: Does the candidate cause precede the effect?
   - Assess causal sufficiency: Would the cause alone be sufficient to produce the effect?
   - Distinguish correlation from causation: Verify that relationships are causal, not merely temporal or correlational

4. **Per-Option Systematic Evaluation:** For each option (A, B, C, D), perform step-by-step analysis:
   a. **Evidence Alignment:** Reference specific extracted passages that mention or relate to this option
   b. **Causal Signal Detection:** Check if passages use explicit causal language linking this option to the target event
   c. **Direct Causality Assessment:** Determine if this option directly causes the target event (single-step) or requires intermediary events (multi-step chain)
   d. **Logical Sufficiency Test:** Evaluate whether this factor alone could realistically cause the target event based on the evidence
   e. **Plausibility Verification:** If evidence supports the option without contradictions or requiring unstated assumptions, mark as valid; otherwise, reject with clear justification
   f. **Counterfactual Consideration:** Ask "Would the target event still occur if this option were absent?" to assess causal necessity

5. **Final Synthesis and Selection:** Review all option evaluations comprehensively:
   - Select all options with strong direct causal evidence and individual sufficiency
   - If multiple options are valid, verify each is independently sufficient (not merely complementary parts of a chain)
   - Ensure selections are grounded strictly in documented evidence without assumptions
   - Prioritize options with explicit causal language and direct single-step relationships
   - If no option has perfect evidence, select the best-supported option(s) from A-D
</chain_of_thought>


<selection_rules>
- Multiple options can be correct - choose ALL that apply
- Select multiple options only if each cause has strong evidence and is individually sufficient
- If options contradict each other, select the one with stronger textual evidence
- Always output ALL correct options, including duplicates: if options are duplicate/identical but correct, include both letters
- If none seems perfectly sufficient, pick the single best-supported among A–D.
- NEVER create options beyond A, B, C, D
- There is always at least one correct option from A-D
</selection_rules>

<quality_checks>
- Verify each selected option has direct quotes or paraphrases from context
- Ensure you haven't made assumptions beyond what's explicitly stated
- Confirm logical sufficiency: could this realistically cause the event by itself?
- Valid answers in the <answer> section are only A,B,C,D; never output anything else; if uncertain, pick the best-supported among A–D and output it without explanations.
</quality_checks>

</instructions>

<output_format>

Provide your answer in exactly this format:

<analysis>
Option A: [Your brief reasoning for option A - 1-2 sentences]
Option B: [Your brief reasoning for option B - 1-2 sentences]
Option C: [Your brief reasoning for option C - 1-2 sentences]
Option D: [Your brief reasoning for option D - 1-2 sentences]
</analysis>

<answer>
[Letter(s) ONLY from the enum- e.g., "B" or "B,D" or "C"]
</answer>
</output_format>
"""

zeroshot_structured_insights = """<role>
You are an expert in identifying the direct cause of events from textual evidence.
</role>

<task>
Given an event, context documents, and candidate explanations, analyze systematically to identify the most plausible direct cause(s).
</task>

<input_format>
<context_documents>
<document_1>: [document content]></document_1>
<document_2>: [document content]></document_2>
[additional documents as needed...]
</context_documents>

<target_event>[event description]</target_event>

<options>
<option_a>[option A]</option_a>
<option_b>[option B]</option_b>
<option_c>[option C]</option_c>
<option_d>[option D]</option_d>
</options>
</input_format>

<instructions>

<reasoning_criteria>
- Base your reasoning ONLY on evidence from the provided context documents
- Look for direct causal relationships, not just correlations or temporal sequences
- Test logical sufficiency: Would this factor alone reasonably be enough to cause the event?
- Require both conditions: Direct textual support AND logical sufficiency to cause the event
- Use single-step reasoning: Avoid multi-step causal chains or indirect relationships
- Prioritize explicit causal language: "caused by," "resulted from," "led to," "triggered by," "due to"
</reasoning_criteria>

<selection_rules>
- Multiple options can be correct - choose ALL that apply
- Select multiple options only if each cause has strong evidence and is individually sufficient
- If options contradict each other, select the one with stronger textual evidence
- Always output ALL correct options, including duplicates: if options are duplicate/identical but correct, include both letters
- If none seems perfectly sufficient, select the single best-supported among A–D.
- NEVER create options beyond A, B, C, D
- There is always at least one correct option from A-D
</selection_rules>

<quality_checks>
- Verify each selected option has direct quotes or paraphrases from context
- Ensure you haven't made assumptions beyond what's explicitly stated
- Confirm logical sufficiency: could this realistically cause the event by itself?
- Valid answers in the <answer> section are only A,B,C,D; never output anything else; if uncertain, pick the best-supported among A–D and output it without explanations.
</quality_checks>

</instructions>

<output_format>

Provide your answer in exactly this format:

<analysis>
Option A: [Your brief reasoning for option A - 1-2 sentences]
Option B: [Your brief reasoning for option B - 1-2 sentences]
Option C: [Your brief reasoning for option C - 1-2 sentences]
Option D: [Your brief reasoning for option D - 1-2 sentences]
</analysis>

<final_verification>
Before finalizing your answer, apply these ABSOLUTE PATTERN-MATCHING RULES as a final check:

1. **"None of the others are correct causes" rule:**
   - If ANY option states "None of the others are correct causes." it MUST be the ONLY answer
   - This option is mutually exclusive - never combine it with A, B, C, or D

2. **Duplicate/identical options rule:**
   - If 2 or 3 options contain identical or near-identical text, ALL duplicates MUST be selected
   - Example: If options B and C are identical, answer must include both "B,C"

3. **All-four prohibition:**
   - The answer is NEVER "A,B,C,D" (all four options together)
   - At least one option must always be incorrect

4. **Paraphrased target event rule:**
   - If ANY option paraphrases the target event itself (same event, different wording), it MUST be selected
   - Example: If target is "X happened" and option says "X occurred", that option is correct

Apply these rules NOW to verify/correct your answer before outputting.
</final_verification>

<answer>
[Letter(s) ONLY from the enum- e.g., "B" or "B,D" or "C"]
</answer>

</output_format>
"""

space_prompt = """<role>
You are an expert analyst specializing in causal reasoning and event analysis. Your task is to identify which events could plausibly cause a given target event, based on supporting evidence and logical reasoning about cause-effect relationships.
</role>

<task_instructions>
Given a target event and multiple choice options, determine which option(s) represent plausible direct causes of the target event. You must:
- Carefully distinguish between causes (events that directly lead to the target) and effects (events that result from the target)
- Consider temporal relationships - causes must precede their effects
- Base your reasoning on provided supporting documents
- Select ALL correct causes when multiple options apply
</task_instructions>

<methodology>
Follow this systematic approach:

<step_1>
<title>Target Event Analysis</title>
- Identify the specific target event and its key characteristics
- Note the timeframe and context of the target event
</step_1>

<step_2>
<title>Document Evidence Review</title>
Scan supporting documents for relevant information about:
- The target event and its context
- Each option and its relationship to the target
- Temporal sequences and chronological order
</step_2>

<step_3>
<title>Causal Reasoning for Each Option</title>
For each option (A, B, C, D), ask:
- Temporal Test: Did this event occur BEFORE the target event?
- Logical Connection: Could this event reasonably lead to the target event?
- Direct vs Indirect: Is this a direct cause or just a related event?
- Evidence Support: Is this relationship supported by the provided documents?
</step_3>

<step_4>
<title>Answer Selection</title>
- Select option(s) that pass all tests above
- If no options qualify as direct causes, select "None of the others are correct causes"
- Format multiple correct answers as comma-separated letters (e.g., "B,D")
</step_4>
</methodology>

<examples>
<example_1>
<target_event>David Cameron announced his resignation as Prime Minister.</target_event>
<options>
A) The referendum date was set in February
B) The UK voted to leave the EU in a June 23 referendum
C) Northern Ireland and Scotland voted to remain in the EU
D) The Conservative Party won the election
</options>
<reasoning>
A) Setting referendum date: This occurred before Cameron's resignation but was not the direct cause of his resignation decision
B) Brexit vote result: This directly led to Cameron's resignation as he had campaigned for Remain and lost
C) Regional voting patterns: This was part of the referendum but not the direct cause of resignation
D) Conservative victory: This enabled the referendum but occurred much earlier
</reasoning>
<answer>B</answer>
</example_1>

<example_2>
<target_event>The D.C. National Guard was activated, and a citywide curfew was imposed.</target_event>
<options>
A) Protesters clashed with police and breached security barriers at the Capitol
B) Trump supporters protested in Washington, D.C.
C) A woman was shot inside the Capitol and died
D) Five people died during the riots
</options>
<reasoning>
A) Capitol breach: This escalation directly caused authorities to activate National Guard and impose curfew
B) Peaceful protest: General protesting alone wouldn't trigger such emergency measures
C) Single shooting: This was a consequence of the chaos, not the cause of emergency response
D) Deaths occurred: This was a result of the violence, not the cause of the response
</reasoning>
<answer>A</answer>
</example_2>
</examples>

<output_format>
Provide your answer in exactly this format:

<analysis>
Option A: [Your brief reasoning for option A - 1-2 sentences]
Option B: [Your brief reasoning for option B - 1-2 sentences]
Option C: [Your brief reasoning for option C - 1-2 sentences]
Option D: [Your brief reasoning for option D - 1-2 sentences]
</analysis>

<answer>
[Letter(s) only - e.g., "B" or "B,D" or "C"]
</answer>

</output_format>


<quality_checks>
Before finalizing your answer, verify:
✓ Selected events temporally precede the target event
✓ Causal relationship is direct and logical (not just correlation)
✓ Answer is supported by document evidence
✓ All plausible causes are included (don't miss multiple correct answers)
</quality_checks>
"""

ALL_TEMPLATES = {
      "zeroshot_structured": zeroshot_structured,
      "zeroshot_structured_insights": zeroshot_structured_insights,
      "space": space_prompt,
      "fewshot": fewshot,
      "fewshot_cot": fewshot_cot,
      "zeroshot_cot": zeroshot_cot
}

def get_template(name: str) -> str:
    """
    Get a prompt template by name. Names: zeroshot_structured, space.
    """
    key = (name or "").strip().lower()
    if key not in ALL_TEMPLATES:
        raise ValueError(f"Unknown template '{name}'. Choose from: {list(ALL_TEMPLATES.keys())}")
    return ALL_TEMPLATES[key]
