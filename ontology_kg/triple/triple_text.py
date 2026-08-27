import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

load_dotenv()

client = OpenAI()

PROMPT_TEMPLATE = """You are an expert in constructing knowledge graphs. 
Your task is to extract knowledge triplets (subject–relation–object) from the given document section. 
The input content may include:
- Plain text (technical specifications, requirements, descriptions)
- Mathematical expressions (formulas, equations, definitions)
- Enumerated content (lists, measurement conversions)
- Tabular content (structured tables with conditions and permitted variations)

## Output Format
- Use the following structure for each triplet:
  <triplet> {{subject}} <subj> {{object}} <obj> relation_name
- `<subj>` and `<obj>` are not natural text; they are boundary markers that indicate where the subject ends and the object begins. 
  Think of them as explicit delimiters: "here ends the subject", "here ends the object".

## Extraction Guidelines
1. Extract only semantically meaningful and domain-relevant relations.
2. For plain text:
   - Coverage or scope → (covers, applies_to)
   - Process or production → (produced_by, used_for)
   - Definitions → (stands_for, defined_as)
   - Specifications → (has_condition_AND, has_condition_OR, has_condition, has_consequence, has_consequence_OR, has_consequence_AND)
3. For mathematical formulas:
   - Variable definitions → (X <subj> Y <obj> stands_for)
   - Formula representation → (X <subj> formula <obj> calculated_by)
   - Component terms → (X <subj> Y <obj> includes_component)
   - Operations → (X <subj> Y <obj> divided_by / multiplied_by / added_to / subtracted_by)
4. For enumerations or units:
   - Capture requirement constraints (e.g., thickness ≥ 3 mm)
   - Always normalize numerical values into SI units (millimeters, Celsius, etc.) in the output triplets, even if the input uses mixed units.

5. When text starts with a numbering pattern (e.g., "1.", "2.1", "3.2.4"), do not extract that numeric label as a triplet. Only process the semantic content that follows.
6. Keep relation naming consistent (snake_case).
7. Use technical terms exactly as written in the input text.
8. Do not generate vague or trivial relations.
9. Do NOT output placeholder tokens such as "<relation>" or generic words like "relation". 
    Always replace with an actual relation name (e.g., covers, produced_by, equivalent_to).

---
## Examples

### Example 1 – Specification
**Input:** This specification covers four grades of steel plates produced by TMCP for welded construction.  
**Output:**  
<triplet> this specification <subj> four grades <obj> covers  
<triplet> steel plates <subj> TMCP <obj> produced_by  
<triplet> TMCP <subj> thermo-mechanically controlled processing <obj> stands_for  
<triplet> steel plates <subj> welded construction <obj> used_for  

---
### Example 2 – Formula
**Input:** CE = C + Mn/6 + (Cr + Mo + V) /5 + (Ni + Cu) / 51  
**Output:**  
<triplet> CE <subj> carbon equivalent <obj> stands_for  
<triplet> CE <subj> C + Mn/6 + (Cr + Mo + V) /5 + (Ni + Cu) / 51 <obj> calculated_by  
<triplet> carbon equivalent <subj> C <obj> includes_component  
<triplet> carbon equivalent <subj> Mn <obj> includes_component  

---
### Example 3 – Unit Conversion 
**Input:** The plates shall have a minimum thickness of 3/8 in. [10 mm].  
**Output (using mm):**  
<triplet> plates <subj> 10 mm <obj> has_minimum_thickness  

---
### Example 4 – Unit Conversion2
**Input:** The plates shall have a minimum thickness of 3/8 in.  
**Output (normalized to mm):**  
<triplet> plates <subj> 10 mm <obj> has_minimum_thickness  

---
### Example 5 – Conditional Requirement
**Input:** If the plate thickness exceeds 2 in. [50 mm], heat treatment shall be required.  
**Output (using mm):**  
<triplet> case1 <subj> plate thickness ≥ 50 mm <obj> has_condition  
<triplet> case1 <subj> heat treatment <obj> has_consequence  

---
Now extract triplets from the following section:
 
**Input:** {text}  
**Output:**
"""

def extract_triplets(text):
    prompt = PROMPT_TEMPLATE.format(text=text)

    response = client.chat.completions.create(
        model = "gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()

def text_main(INPUT_FILE, OUTPUT_FILE):
    data = pd.read_csv(INPUT_FILE)

    df = data[~data.content_type.str.startswith("table")]
    df.reset_index(drop=True, inplace= True) 
    

    df["triplets"] = None

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    for i, row in df.iterrows():
        title, text = row["title"], row["text"]
        print(f"[{i+1}/{len(df)}] Processing section: {title}")

        triplets = extract_triplets(text)

        df.at[i, "triplets"] = triplets


        df.iloc[[i]].to_csv(
            OUTPUT_FILE,
            mode="a",
            index=False,
            encoding="utf-8-sig",
            header=not os.path.exists(OUTPUT_FILE) or i == 0
        )

if __name__ == "__main__":
    INPUT_FILE = ""
    OUTPUT_FILE = ""
    text_main(INPUT_FILE, OUTPUT_FILE)