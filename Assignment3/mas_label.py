# Load dependencies
import pandas as pd
import json
from vllm import LLM, SamplingParams
from pathlib import Path
import os
from dotenv import load_dotenv
import re

# Load Hugging Face token from .env file
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Define input and output paths
INPUT_PATH  = Path("csv/hitl_green_100.csv")
OUTPUT_PATH = Path("csv/mas_labeled.csv")

# Deterministic output, 512 tokens is enough for arguments and JSON
SAMPLING_PARAMS = SamplingParams(temperature=0, max_tokens=512)

# Models
ADVOCATE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
SKEPTIC_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_MODEL    = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load the uncertain claims from the CSV file
df = pd.read_csv(INPUT_PATH)
claims = df["text"].tolist()

# Define the prompts for each agent
def advocate_prompt(claim):
    # Argues FOR green classification — looks for environmental benefits
    return f"""[INST] You are a green technology expert reviewing a patent claim.
Your task is to argue FOR why this patent claim could be classified as green/sustainable technology.
Focus on any environmental benefits, energy efficiency, or sustainability aspects.
Patent claim:
{claim}
Respond in 2-3 sentences arguing for the green classification. [/INST]"""

def skeptic_prompt(claim):
    # Argues AGAINST green classification — looks for greenwashing or weak signals
    return f"""[INST] You are a critical analyst reviewing a patent claim for greenwashing.
Your task is to argue AGAINST classifying this patent claim as green/sustainable technology.
Look for weak green signals, missing environmental benefits, or misleading sustainability claims.
Patent claim:
{claim}
Respond in 2-3 sentences arguing against the green classification. [/INST]"""

def judge_prompt(claim, advocate_arg, skeptic_arg):
    # Weighs both arguments and produces the final JSON label
    return f"""[INST] You are an impartial judge evaluating whether a patent claim describes green/sustainable technology.
You have received arguments from two experts. Weigh both arguments and make a final decision.
Patent claim:
{claim}
Advocate argues FOR green classification:
{advocate_arg}
Skeptic argues AGAINST green classification:
{skeptic_arg}
Respond in this exact JSON format:
{{
    "mas_green_suggested": 0 or 1,
    "mas_confidence": "low", "medium", or "high",
    "mas_rationale": "1-3 sentences explaining your final decision"
}}
[/INST]"""

# Load each agent independently
# Step 1 — Advocate
print("Loading Advocate (Mistral)...")
advocate_llm = LLM(model=ADVOCATE_MODEL, tensor_parallel_size=1, dtype="float16")
advocate_outputs = advocate_llm.generate([advocate_prompt(c) for c in claims], SAMPLING_PARAMS)
advocate_args = [o.outputs[0].text.strip() for o in advocate_outputs]
del advocate_llm
print("Advocate done.")

# Step 2 — Skeptic
print("Loading Skeptic (Qwen)...")
skeptic_llm = LLM(model=SKEPTIC_MODEL, tensor_parallel_size=1, dtype="float16")
skeptic_outputs = skeptic_llm.generate([skeptic_prompt(c) for c in claims], SAMPLING_PARAMS)
skeptic_args = [o.outputs[0].text.strip() for o in skeptic_outputs]
del skeptic_llm
print("Skeptic done.")

# Step 3 — Judge
print("Loading Judge (Llama-3)...")
judge_llm = LLM(model=JUDGE_MODEL, tensor_parallel_size=1, dtype="float16")
judge_outputs = judge_llm.generate(
    [judge_prompt(c, a, s) for c, a, s in zip(claims, advocate_args, skeptic_args)],
    SAMPLING_PARAMS
)
del judge_llm
print("Judge done.")

# Parse the judge's JSON output and combine it with the original DataFrame
parsed_results = []
for i, output in enumerate(judge_outputs):
    text = output.outputs[0].text.strip()
    
    # Extract just the JSON block — ignores trailing [/INST] tokens from Llama-3
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    
    try:
        parsed = json.loads(match.group())
        parsed_results.append({
            "mas_green_suggested": parsed["mas_green_suggested"],
            "mas_confidence":      parsed["mas_confidence"],
            "mas_rationale":       parsed["mas_rationale"],
            "advocate_arg":        advocate_args[i],
            "skeptic_arg":         skeptic_args[i],
        })
    except (json.JSONDecodeError, AttributeError):
        # Save raw output so we can inspect failed parses manually
        parsed_results.append({
            "mas_green_suggested": "",
            "mas_confidence":      "",
            "mas_rationale":       text,
            "advocate_arg":        advocate_args[i],
            "skeptic_arg":         skeptic_args[i],
        })

results_df = pd.DataFrame(parsed_results)
df = pd.concat([df, results_df], axis=1)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df)} labeled examples to {OUTPUT_PATH}")
print(df[["mas_green_suggested", "mas_confidence"]].value_counts())