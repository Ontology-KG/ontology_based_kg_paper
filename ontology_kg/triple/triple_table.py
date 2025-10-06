import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

load_dotenv()

client = OpenAI()


PROMPT_TEMPLATE = """You are an expert in constructing knowledge graphs. 
Your task is to extract knowledge triplets (subject–relation–object) from the given tabular content. 
The input content may include:
- Tabular content (structured tables with conditions and permitted variations)
- Table title
- Table captions outside the table

## Output Format
- Use the following structure for each triplet:
  <triplet> {{subject}} <subj> {{object}} <obj> relation_name
- `<subj>` and `<obj>` are not natural text; they are boundary markers that indicate where the subject ends and the object begins. 
  Think of them as explicit delimiters: "here ends the subject", "here ends the object".

## Extraction Guidelines
1. Extract only semantically meaningful and domain-relevant relations.
2. For enumerations or units:
  - Capture requirement constraints (e.g., thickness ≥ 3 mm)
  - Always normalize numerical values into SI units (millimeters, Celsius, etc.) in the output triplets, even if the input uses mixed units.
3. For tabular data with conditions:
  - If the input starts with a "|" character, treat it strictly as TABLE data: do not apply any plain text rules, and only use table-specific rules.
  -  When processing tables, the ONLY allowed relations are:
    - If there is only one condition → `has_condition`
    - If there are multiple conditions → `has_condition_AND` or `has_condition_OR`
    - If there is only one consequence → `has_consequence`
    - If there are multiple consequences → `has_consequence_AND` or `has_consequence_OR`
    - Do not use any other relation names.
  - Each cell in a table must be represented as a rule node (caseX).
    - For every non-empty cell, create a new case node (case1, case2, …).
    - Each case node must include:
      - all row labels as conditions,
      - all column labels as conditions,
      - and the cell value as the consequence.
  - **When logical expressions contain both AND and OR, you must introduce a `logic_groupX` node.**  
    - Inside a `logic_groupX`, you must use **only one type of edge**:  
      either `has_condition_AND`, `has_condition_OR`, `has_consequence_AND`, or `has_consequence_OR`.
    - The `logic_groupX` itself must then connect to the case node (`caseX`)  
      using the appropriate relation among the same four types.  
    - Always preserve operator precedence by grouping first.  
      Example: `(A OR B) AND C` → A and B are both connected to a `logic_groupX` with `has_condition_OR`;  
      then this `logic_groupX` and C are connected to `caseX` with `has_condition_AND`.   
  - Always include the column/row name together with the cell value when forming a condition or consequence.  
    Example: `(8 mm <= specified_width_column <= 20 mm)`, `(thickness_column <= 15.3 mm)`, `(permitted_variation_column = 1/8 mm)`
  - Do not connect conditions directly to consequences; always go through a case or logic_group.

4. Keep relation naming consistent (snake_case).
5. Use technical terms exactly as written in the input text.
6. Do not generate vague or trivial relations.
7. Do NOT output placeholder tokens such as "<relation>" or generic words like "relation". 
  Always replace with an actual relation name.

---
## Examples

### Example 1 – Unit Conversion
**Input:** The plates shall have a minimum thickness of 3/8 in. [10 mm].  
**Output (using mm):**  
<triplet> plates <subj> 10 mm <obj> has_minimum_thickness  

---

### Example 2 – Unit Conversion2
**Input:** The plates shall have a minimum thickness of 3/8 in.  
**Output (normalized to mm):**  
<triplet> plates <subj> 10 mm <obj> has_minimum_thickness  

---

### Example 3 – Table (Simple Condition)
**Input:** | Thickness A Range Rolled for the Heat   | Thickness A Difference Between Pieces or Plates-as-rolled in the Thickness A Range   | Minimum Number of Tension Tests Required                                                                                                                  | |-||-| | Under 3 ⁄ 8 in. [10 mm]                 | 1 ⁄ 16 in. [2 mm] or less                                                            | Two B tests per heat, taken from different pieces or plates-as-rolled having any thickness A in the thickness A range                                     | |                                         | More than 1 ⁄ 16 in. [2 mm]                                                          | Two B tests per heat, one taken from the minimum thickness A in the thickness A range and one taken from the maximum thickness A in the thickness A range | | 3 ⁄ 8 to 2 in. [10 to 50 mm], incl      | Less than 3 ⁄ 8 in. [10 mm]                                                          | Two B tests per heat, taken from different pieces or plates-as-rolled having any thickness A in the thickness A range                                     | |                                         | 3 ⁄ 8 in. [10 mm] or more                                                            | Two B tests per heat, one taken from the minimum thickness A in the thickness A range and one taken from the maximum thickness A in the thickness A range | 
**Output:**  
<triplet> case1 <subj> thickness < 10 mm <obj> has_condition_AND
<triplet> case1 <subj> thickness_difference ≤ 2 mm <obj> has_condition_AND
<triplet> case1 <subj> minimum_tension_tests = two_tests_per_heat_taken_from_different_pieces_or_plates_as_rolled_having_any_thickness_in_the_thickness_range <obj> has_consequence

<triplet> case2 <subj> thickness < 10 mm <obj> has_condition_AND
<triplet> case2 <subj> thickness_difference > 2 mm <obj> has_condition_AND
<triplet> case2 <subj> minimum_tension_tests = two_tests_per_heat_one_taken_from_the_minimum_thickness_and_one_taken_from_the_maximum_thickness <obj> has_consequence

<triplet> case3 <subj> 10 mm ≤ thickness ≤ 50 mm <obj> has_condition_AND
<triplet> case3 <subj> thickness_difference < 10 mm <obj> has_condition_AND
<triplet> case3 <subj> minimum_tension_tests = two_tests_per_heat_taken_from_different_pieces_or_plates_as_rolled_having_any_thickness_in_the_range <obj> has_consequence

<triplet> case4 <subj> 10 mm ≤ thickness ≤ 50 mm <obj> has_condition_AND
<triplet> case4 <subj> thickness_difference ≥ 10 mm <obj> has_condition_AND
<triplet> case4 <subj> minimum_tension_tests = two_tests_per_heat_one_taken_from_the_minimum_thickness_and_one_taken_from_the_maximum_thickness <obj> has_consequence

---

### Example 4 – Table (Mixed Condition with logic_group)
**Input:** |                  |                                | Specified Dimensions, in. Permitted Variations Over Specified Width and Length A for Thicknesses Pounds per Square   | Specified Dimensions, in. Permitted Variations Over Specified Width and Length A for Thicknesses Pounds per Square   | Specified Dimensions, in. Permitted Variations Over Specified Width and Length A for Thicknesses Pounds per Square   | Specified Dimensions, in. Permitted Variations Over Specified Width and Length A for Thicknesses Pounds per Square   | Specified Dimensions, in. Permitted Variations Over Specified Width and Length A for Thicknesses Pounds per Square   | Specified Dimensions, in. Permitted Variations Over Specified Width and Length A for Thicknesses Pounds per Square   | Specified Dimensions, in. Permitted Variations Over Specified Width and Length A for Thicknesses Pounds per Square   | Specified Dimensions, in. Permitted Variations Over Specified Width and Length A for Thicknesses Pounds per Square   | ||||||||||| | Length           |                                | To 3 ⁄ 8 , excl                                                                                                      | To 3 ⁄ 8 , excl                                                                                                      | 3 ⁄ 8 to 5 ⁄ 8                                                                                                       | 3 ⁄ 8 to 5 ⁄ 8                                                                                                       | 5 ⁄ 8 to 1, excl                                                                                                     | 5 ⁄ 8 to 1, excl                                                                                                     | 1 to 2, incl                                                                                                         | 1 to 2, incl                                                                                                         | |                  | Width                          | To 15.3, excl                                                                                                        | To 15.3, excl                                                                                                        | 15.3 to 25.5,                                                                                                        | 15.3 to 25.5,                                                                                                        | 25.5 to 40.8, excl                                                                                                   | 25.5 to 40.8, excl                                                                                                   | 40.8 to 81.7,                                                                                                        | 40.8 to 81.7,                                                                                                        | |                  |                                | Width                                                                                                                | Length                                                                                                               | Width                                                                                                                | Length                                                                                                               | Width                                                                                                                | Length                                                                                                               | Width                                                                                                                | Length                                                                                                               | | To 120, excl     | To 60, excl                    | 3 ⁄ 8                                                                                                                | 1 ⁄ 2                                                                                                                | 7 ⁄ 16                                                                                                               | 5 ⁄ 8                                                                                                                | 1 ⁄ 2                                                                                                                | 3 ⁄ 4                                                                                                                | 5 ⁄ 8                                                                                                                | 1                                                                                                                    | |                  | 60 to 84, excl                 | 7 ⁄ 16                                                                                                               | 5 ⁄ 8                                                                                                                | 1 ⁄ 2                                                                                                                | 11 ⁄ 16                                                                                                              | 5 ⁄ 8                                                                                                                | 7 ⁄ 8                                                                                                                | 3 ⁄ 4                                                                                                                | 1                                                                                                                    | |
**Output:**  
<triplet> case1 <subj> length ≤ 3048 mm, excl <obj> has_condition_AND  
<triplet> case1 <subj> width ≤ 1524 mm, excl <obj> has_condition_AND  
<triplet> case1 <subj> logic_group1 <obj> has_condition_AND  
<triplet> case1 <subj> width_permitted_variation = 9.525 mm <obj> has_consequence_OR  
<triplet> case1 <subj> length_permitted_variation = 12.7 mm <obj> has_consequence_OR
<triplet> logic_group1 <subj> thickness ≤ 9.525 mm, excl <obj> has_condition_OR  
<triplet> logic_group1 <subj> equivalent_weight ≤ 74.74 kg/m², excl <obj> has_condition_OR 

<triplet> case2 <subj> length ≤ 3048 mm, excl <obj> has_condition_AND
<triplet> case2 <subj> width ≤ 1524 mm, excl <obj> has_condition_AND
<triplet> case2 <subj> logic_group2 <obj> has_condition_AND
<triplet> case2 <subj> width_permitted_variation = 11.1125 mm <obj> has_consequence_OR
<triplet> case2 <subj> length_permitted_variation = 15.875 mm <obj> has_consequence_OR
<triplet> logic_group2 <subj> 9.525 mm < thickness ≤ 15.875 mm <obj> has_condition_OR
<triplet> logic_group2 <subj> 74.74 < equivalent_weight ≤ 124.5 kg/m² <obj> has_condition_OR

<triplet> case3 <subj> length ≤ 3048 mm, excl <obj> has_condition_AND
<triplet> case3 <subj> width ≤ 1524 mm, excl <obj> has_condition_AND
<triplet> case3 <subj> logic_group3 <obj> has_condition_AND
<triplet> case3 <subj> width_permitted_variation = 12.7 mm <obj> has_consequence_OR
<triplet> case3 <subj> length_permitted_variation = 19.05 mm <obj> has_consequence_OR
<triplet> logic_group3 <subj> 15.875 mm < thickness ≤ 25.4 mm <obj> has_condition_OR
<triplet> logic_group3 <subj> 124.5 < equivalent_weight ≤ 199.0 kg/m² <obj> has_condition_OR

<triplet> case4 <subj> length ≤ 3048 mm, excl <obj> has_condition_AND
<triplet> case4 <subj> width ≤ 1524 mm, excl <obj> has_condition_AND
<triplet> case4 <subj> logic_group4 <obj> has_condition_AND
<triplet> case4 <subj> width_permitted_variation = 15.875 mm <obj> has_consequence_OR
<triplet> case4 <subj> length_permitted_variation = 25.4 mm <obj> has_consequence_OR
<triplet> logic_group4 <subj> 25.4 mm < thickness ≤ 50.8 mm <obj> has_condition_OR
<triplet> logic_group4 <subj> 199.0 < equivalent_weight ≤ 399.0 kg/m² <obj> has_condition_OR

<triplet> case5 <subj> length ≤ 3048 mm, excl <obj> has_condition_AND
<triplet> case5 <subj> 1524 mm < width ≤ 2133.6 mm, excl <obj> has_condition_AND
<triplet> case5 <subj> logic_group5 <obj> has_condition_AND
<triplet> case5 <subj> width_permitted_variation = 11.1125 mm <obj> has_consequence_OR
<triplet> case5 <subj> length_permitted_variation = 15.875 mm <obj> has_consequence_OR
<triplet> logic_group5 <subj> thickness ≤ 9.525 mm, excl <obj> has_condition_OR
<triplet> logic_group5 <subj> equivalent_weight ≤ 74.74 kg/m², excl <obj> has_condition_OR

<triplet> case6 <subj> length ≤ 3048 mm, excl <obj> has_condition_AND
<triplet> case6 <subj> 1524 mm < width ≤ 2133.6 mm, excl <obj> has_condition_AND
<triplet> case6 <subj> logic_group6 <obj> has_condition_AND
<triplet> case6 <subj> width_permitted_variation = 12.7 mm <obj> has_consequence_OR
<triplet> case6 <subj> length_permitted_variation = 17.4625 mm <obj> has_consequence_OR
<triplet> logic_group6 <subj> 9.525 mm < thickness ≤ 15.875 mm <obj> has_condition_OR
<triplet> logic_group6 <subj> 74.74 < equivalent_weight ≤ 124.5 kg/m² <obj> has_condition_OR

<triplet> case7 <subj> length ≤ 3048 mm, excl <obj> has_condition_AND
<triplet> case7 <subj> 1524 mm < width ≤ 2133.6 mm, excl <obj> has_condition_AND
<triplet> case7 <subj> logic_group7 <obj> has_condition_AND
<triplet> case7 <subj> width_permitted_variation = 15.875 mm <obj> has_consequence_OR
<triplet> case7 <subj> length_permitted_variation = 22.225 mm <obj> has_consequence_OR
<triplet> logic_group7 <subj> 15.875 mm < thickness ≤ 25.4 mm <obj> has_condition_OR
<triplet> logic_group7 <subj> 124.5 < equivalent_weight ≤ 199.0 kg/m² <obj> has_condition_OR

<triplet> case8 <subj> length ≤ 3048 mm, excl <obj> has_condition_AND
<triplet> case8 <subj> 1524 mm < width ≤ 2133.6 mm, excl <obj> has_condition_AND
<triplet> case8 <subj> logic_group8 <obj> has_condition_AND
<triplet> case8 <subj> width_permitted_variation = 19.05 mm <obj> has_consequence_OR
<triplet> case8 <subj> length_permitted_variation = 25.4 mm <obj> has_consequence_OR
<triplet> logic_group8 <subj> 25.4 mm < thickness ≤ 50.8 mm <obj> has_condition_OR
<triplet> logic_group8 <subj> 199.0 < equivalent_weight ≤ 399.0 kg/m² <obj> has_condition_OR


---

### Example 5 - Table (Mixed Condition with logic_group)
**Input:** |                      | Permitted Variations Over Specified Width A for Thicknesses Given in Inches or Equivalent Weights Given in Pounds per Square Foot, in.   | Permitted Variations Over Specified Width A for Thicknesses Given in Inches or Equivalent Weights Given in Pounds per Square Foot, in.   | Permitted Variations Over Specified Width A for Thicknesses Given in Inches or Equivalent Weights Given in Pounds per Square Foot, in.   | Permitted Variations Over Specified Width A for Thicknesses Given in Inches or Equivalent Weights Given in Pounds per Square Foot, in.   | Permitted Variations Over Specified Width A for Thicknesses Given in Inches or Equivalent Weights Given in Pounds per Square Foot, in.   | Permitted Variations Over Specified Width A for Thicknesses Given in Inches or Equivalent Weights Given in Pounds per Square Foot, in.   | |||||||| | Specified Width, in. | To 3 ⁄ 8 , excl                                                                                                                          | 3 ⁄ 8 to 5 ⁄ 8 , excl                                                                                                                    | 5 ⁄ 8 to 1, excl                                                                                                                         | 1 to 2, incl                                                                                                                             | Over 2 to 10, incl                                                                                                                       | Over 10 to 15, incl                                                                                                                      | |                      | To 15.3, excl                                                                                                                            | 15.3 to 25.5, excl                                                                                                                       | 25.5 to 40.8, excl                                                                                                                       | 40.8 to 81.7, incl                                                                                                                       | 81.7 to 409.0, incl                                                                                                                      | 409.0 to 613.0, incl                                                                                                                     | | Over 8 to 20, excl   | 1 ⁄ 8                                                                                                                                    | 1 ⁄ 8                                                                                                                                    | 3 ⁄ 16                                                                                                                                   | 1 ⁄ 4                                                                                                                                    | 3 ⁄ 8                                                                                                                                    | 1 ⁄ 2                                                                                                                                    | | 20 to 36, excl       | 3 ⁄ 16                                                                                                                                   | 1 ⁄ 4                                                                                                                                    | 5 ⁄ 16                                                                                                                                   | 3 ⁄ 8                                                                                                                                    | 7 ⁄ 16                                                                                                                                   | 9 ⁄ 16                                                                                                                                   | 
**Output:**
<triplet> case1 <subj> 203.2 mm < width ≤ 508 mm, excl <obj> has_condition_AND
<triplet> case1 <subj> logic_group1 <obj> has_condition_AND
<triplet> case1 <subj> width_permitted_variation = 3.175 mm <obj> has_consequence_OR
<triplet> logic_group1 <subj> thickness ≤ 9.525 mm, excl <obj> has_condition_OR
<triplet> logic_group1 <subj> equivalent_weight ≤ 74.74 kg/m², excl <obj> has_condition_OR

<triplet> case2 <subj> 203.2 mm < width ≤ 508 mm, excl <obj> has_condition_AND
<triplet> case2 <subj> logic_group2 <obj> has_condition_AND
<triplet> case2 <subj> width_permitted_variation = 3.175 mm <obj> has_consequence_OR
<triplet> logic_group2 <subj> 9.525 mm < thickness ≤ 15.875 mm <obj> has_condition_OR
<triplet> logic_group2 <subj> 74.74 < equivalent_weight ≤ 124.5 kg/m² <obj> has_condition_OR

<triplet> case3 <subj> 203.2 mm < width ≤ 508 mm, excl <obj> has_condition_AND
<triplet> case3 <subj> logic_group3 <obj> has_condition_AND
<triplet> case3 <subj> width_permitted_variation = 4.7625 mm <obj> has_consequence_OR
<triplet> logic_group3 <subj> 15.875 mm < thickness ≤ 25.4 mm <obj> has_condition_OR
<triplet> logic_group3 <subj> 124.5 < equivalent_weight ≤ 199.0 kg/m² <obj> has_condition_OR

<triplet> case4 <subj> 203.2 mm < width ≤ 508 mm, excl <obj> has_condition_AND
<triplet> case4 <subj> logic_group4 <obj> has_condition_AND
<triplet> case4 <subj> width_permitted_variation = 6.35 mm <obj> has_consequence_OR
<triplet> logic_group4 <subj> 25.4 mm < thickness ≤ 50.8 mm <obj> has_condition_OR
<triplet> logic_group4 <subj> 199.0 < equivalent_weight ≤ 399.0 kg/m² <obj> has_condition_OR

<triplet> case5 <subj> 203.2 mm < width ≤ 508 mm, excl <obj> has_condition_AND
<triplet> case5 <subj> logic_group5 <obj> has_condition_AND
<triplet> case5 <subj> width_permitted_variation = 9.525 mm <obj> has_consequence_OR
<triplet> logic_group5 <subj> 50.8 mm < thickness ≤ 254 mm <obj> has_condition_OR
<triplet> logic_group5 <subj> 399.0 < equivalent_weight ≤ 1995.0 kg/m² <obj> has_condition_OR

<triplet> case6 <subj> 203.2 mm < width ≤ 508 mm, excl <obj> has_condition_AND
<triplet> case6 <subj> logic_group6 <obj> has_condition_AND
<triplet> case6 <subj> width_permitted_variation = 12.7 mm <obj> has_consequence_OR
<triplet> logic_group6 <subj> 254 mm < thickness ≤ 381 mm <obj> has_condition_OR
<triplet> logic_group6 <subj> 1995.0 < equivalent_weight ≤ 2989.0 kg/m² <obj> has_condition_OR

<triplet> case7 <subj> 508 mm < width ≤ 914.4 mm, excl <obj> has_condition_AND
<triplet> case7 <subj> logic_group7 <obj> has_condition_AND
<triplet> case7 <subj> width_permitted_variation = 4.7625 mm <obj> has_consequence_OR
<triplet> logic_group7 <subj> thickness ≤ 9.525 mm, excl <obj> has_condition_OR
<triplet> logic_group7 <subj> equivalent_weight ≤ 74.74 kg/m², excl <obj> has_condition_OR

<triplet> case8 <subj> 508 mm < width ≤ 914.4 mm, excl <obj> has_condition_AND
<triplet> case8 <subj> logic_group8 <obj> has_condition_AND
<triplet> case8 <subj> width_permitted_variation = 6.35 mm <obj> has_consequence_OR
<triplet> logic_group8 <subj> 9.525 mm < thickness ≤ 15.875 mm <obj> has_condition_OR
<triplet> logic_group8 <subj> 74.74 < equivalent_weight ≤ 124.5 kg/m² <obj> has_condition_OR

<triplet> case9 <subj> 508 mm < width ≤ 914.4 mm, excl <obj> has_condition_AND
<triplet> case9 <subj> logic_group9 <obj> has_condition_AND
<triplet> case9 <subj> width_permitted_variation = 7.9375 mm <obj> has_consequence_OR
<triplet> logic_group9 <subj> 15.875 mm < thickness ≤ 25.4 mm <obj> has_condition_OR
<triplet> logic_group9 <subj> 124.5 < equivalent_weight ≤ 199.0 kg/m² <obj> has_condition_OR

<triplet> case10 <subj> 508 mm < width ≤ 914.4 mm, excl <obj> has_condition_AND
<triplet> case10 <subj> logic_group10 <obj> has_condition_AND
<triplet> case10 <subj> width_permitted_variation = 9.525 mm <obj> has_consequence_OR
<triplet> logic_group10 <subj> 25.4 mm < thickness ≤ 50.8 mm <obj> has_condition_OR
<triplet> logic_group10 <subj> 199.0 < equivalent_weight ≤ 399.0 kg/m² <obj> has_condition_OR

<triplet> case11 <subj> 508 mm < width ≤ 914.4 mm, excl <obj> has_condition_AND
<triplet> case11 <subj> logic_group11 <obj> has_condition_AND
<triplet> case11 <subj> width_permitted_variation = 11.1125 mm <obj> has_consequence_OR
<triplet> logic_group11 <subj> 50.8 mm < thickness ≤ 254 mm <obj> has_condition_OR
<triplet> logic_group11 <subj> 399.0 < equivalent_weight ≤ 1995.0 kg/m² <obj> has_condition_OR

<triplet> case12 <subj> 508 mm < width ≤ 914.4 mm, excl <obj> has_condition_AND
<triplet> case12 <subj> logic_group12 <obj> has_condition_AND
<triplet> case12 <subj> width_permitted_variation = 12.7 mm <obj> has_consequence_OR
<triplet> logic_group12 <subj> 254 mm < thickness ≤ 381 mm <obj> has_condition_OR
<triplet> logic_group12 <subj> 1995.0 < equivalent_weight ≤ 2989.0 kg/m² <obj> has_condition_OR

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

    df = data[data.content_type.str.startswith("table")]
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