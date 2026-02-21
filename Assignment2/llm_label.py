# Load dependencies
import pandas as pd
from vllm import LLM, SamplingParams
from pathlib import Path

# For parsing we'll import JSON
import json

# Define the model
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Initialize the LLM with the specified model and parameters
llm = LLM(
    model=MODEL,
    tensor_parallel_size=1, # Use 1 GPU for inference
    dtype="float16",
)

# Function for the prompt
def make_prompt(claim_text):
    return f"""[INST] You are a patent classifier. Your task is to determine if a patent claim describes a green/sustainable technology.

Read the following patent claim and respond in this exact JSON format:
{{
    "llm_green_suggested": 0 or 1,
    "llm_confidence": "low", "medium", or "high",
    "llm_rationale": "1-3 sentences citing specific phrases from the claim"
}}

Only use the claim text to make your decision. Do not use any other knowledge.

Patent claim:
{claim_text}
[/INST]"""

# Define input and output paths
INPUT_PATH  = Path("csv/hitl_green_100.csv")
OUTPUT_PATH = Path("csv/hitl_llm_labeled.csv")

# Load the input data
df = pd.read_csv(INPUT_PATH)

# Generate LLM outputs for each claim in the dataframe
# We set temperature=0 for deterministic outputs, and max_tokens=512 to allow for a detailed rationale.
sampling_params = SamplingParams(temperature=0, max_tokens=512)
prompts = [make_prompt(text) for text in df["text"]]

# This will generate LLM outputs for each prompt. The outputs will be in the form of JSON strings as specified in the prompt.
outputs = llm.generate(prompts, sampling_params)

# Parse the LLM outputs and extract the following:
# llm_green_suggested, llm_confidence, llm_rationale
results = []
for output in outputs:
    text = output.outputs[0].text.strip()
    try:
        text = output.outputs[0].text.strip()
        # Mistral sometimes omits the closing brace — fix it if missing
        if not text.endswith("}"):
            text = text + "}"
        parsed = json.loads(text)
        results.append({
            "llm_green_suggested": parsed["llm_green_suggested"],
            "llm_confidence":      parsed["llm_confidence"],
            "llm_rationale":       parsed["llm_rationale"],
        })
    except json.JSONDecodeError:
        results.append({
            "llm_green_suggested": "",
            "llm_confidence":      "",
            "llm_rationale":       text,  # save raw output so you can inspect it
        })

results_df = pd.DataFrame(results)
df[["llm_green_suggested", "llm_confidence", "llm_rationale"]] = results_df
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df)} labeled examples to {OUTPUT_PATH}")
print(df[["llm_green_suggested", "llm_confidence"]].value_counts())