import csv
import re
from pathlib import Path

import pandas as pd


def remove_section_numbers(value):
    if pd.isna(value) or value == "":
        return value
    if not isinstance(value, str):
        return str(value)

    cleaned = value.strip()
    pattern = r"^[A-Za-z]*[0-9]+(\.[0-9]+)*\s+"

    return re.sub(pattern, "", cleaned).strip()


def create_section_array(row):
    section_array = []

    for column in [
        "section",
        "subsection",
        "subsubsection",
        "subsubsubsection",
    ]:
        value = row.get(column, "")

        if pd.notna(value) and str(value).strip():
            section_array.append(str(value).strip())

    return section_array


def clean_quotes_and_whitespace(value):
    previous = None

    while previous != value:
        previous = value
        value = value.strip(" \"'")

    return value


def detect_only_section(row):
    title = str(row["title"]).strip()
    text = str(row["text"]).strip()

    if not title or not text:
        return False

    last_title_part = title.split(" -> ")[-1].strip()

    text_without_number = re.sub(
        r"^\d+(\.\d+)*\.?\s*",
        "",
        text,
    ).strip()

    text_without_semicolon = text_without_number.rstrip(";").strip()

    normalized_title = last_title_part.lower()
    normalized_text = text_without_semicolon.lower()

    if normalized_title == normalized_text:
        return True

    if (
        normalized_title in normalized_text
        and len(last_title_part) > 5
    ):
        return True

    if (
        normalized_text in normalized_title
        and len(text_without_semicolon) > 5
    ):
        return True

    title_words = set(normalized_title.split())
    text_words = set(normalized_text.split())

    if title_words and text_words:
        common_words = title_words.intersection(text_words)
        similarity = len(common_words) / max(
            len(title_words),
            len(text_words),
        )

        if similarity > 0.7:
            return True

    return False


def assign_sections_to_items(df):
    df_sorted = df.sort_values("sort_id").copy()

    enhanced_arrays = []

    current_section = ""
    current_subsection = ""
    current_subsubsection = ""
    current_subsubsubsection = ""

    current_table_header = None
    current_table_subheader = None

    for _, row in df_sorted.iterrows():
        content_type = row.get("content_type", "")
        label = row.get("label", "")

        row_section_array = create_section_array(row)

        if row_section_array:
            if len(row_section_array) >= 1:
                current_section = row_section_array[0].strip()
                current_subsection = ""
                current_subsubsection = ""
                current_subsubsubsection = ""

            if len(row_section_array) >= 2:
                current_subsection = row_section_array[1].strip()
                current_subsubsection = ""
                current_subsubsubsection = ""

            if len(row_section_array) >= 3:
                current_subsubsection = row_section_array[2].strip()
                current_subsubsubsection = ""

            if len(row_section_array) >= 4:
                current_subsubsubsection = row_section_array[3].strip()

        elif content_type == "section":
            content = str(row.get("content", "")).strip()

            if content:
                current_section = content
                current_subsection = ""
                current_subsubsection = ""
                current_subsubsubsection = ""

        current_section_array = []

        if current_section:
            current_section_array.append(current_section)

        if current_subsection:
            current_section_array.append(current_subsection)

        if current_subsubsection:
            current_section_array.append(current_subsubsection)

        if current_subsubsubsection:
            current_section_array.append(current_subsubsubsection)

        if label == "table_header":
            current_table_header = row["content"]
            current_table_subheader = None

            enhanced_arrays.append(
                current_section_array.copy()
            )

        elif label == "table_subheader":
            current_table_subheader = row["content"]

            enhanced_arrays.append(
                current_section_array.copy()
            )

        elif label in [
            "table",
            "table_footnote",
            "table_note",
        ]:
            section_array = current_section_array.copy()

            if label == "table":
                if current_table_header:
                    section_array.append(
                        current_table_header
                    )

                if current_table_subheader:
                    section_array.append(
                        current_table_subheader
                    )

            elif label in [
                "table_footnote",
                "table_note",
            ]:
                if current_table_header:
                    section_array.append(
                        current_table_header
                    )

            enhanced_arrays.append(section_array)

        else:
            enhanced_arrays.append(
                current_section_array.copy()
            )

    df_sorted["enhanced_section_array"] = enhanced_arrays

    return df_sorted


def safe_sentence_split(text):
    unit_pattern = (
        r"(\d+(?:\s*[-/]\s*\d+)?\s*-?"
        r"(?:µin|µm|in|mm|ft|cm|m|lb|kg|ksi|MPa|°F|°C))\."
    )

    standalone_unit_pattern = (
        r"\b(µin|µm|in|mm|ft|cm|m|lb|kg|ksi|MPa|°F|°C)\."
    )

    units = []

    def replace_unit(match):
        units.append(match.group(0))

        return (
            f"{match.group(1)}"
            f"§UNIT{len(units) - 1}§"
        )

    protected_text = re.sub(
        unit_pattern,
        replace_unit,
        text,
    )

    protected_text = re.sub(
        standalone_unit_pattern,
        replace_unit,
        protected_text,
    )

    abbreviation_pattern = (
        r"(\b(?:Fed|Std|No|etc|vs|Mr|Mrs|Dr|Prof|Fig|"
        r"Table|Spec|Vol|Ch|Sec|Art|Para|Inc|Corp|Ltd|"
        r"Co|Ave|St|Rd|Blvd|min|max|temp|approx)\s*)\."
    )

    dotted_abbreviation_pattern = (
        r"(\b[A-Z](?:\.[A-Z])+)\."
    )

    abbreviations = []

    def replace_abbreviation(match):
        abbreviations.append(match.group(0))

        return (
            f"{match.group(1)}"
            f"§ABB{len(abbreviations) - 1}§"
        )

    protected_text = re.sub(
        abbreviation_pattern,
        replace_abbreviation,
        protected_text,
        flags=re.IGNORECASE,
    )

    protected_text = re.sub(
        dotted_abbreviation_pattern,
        replace_abbreviation,
        protected_text,
    )

    section_number_pattern = (
        r"(\b[A-Z]\d+(?:\.\d+)*)\."
    )

    section_numbers = []

    def replace_section_number(match):
        section_numbers.append(match.group(0))

        return (
            f"{match.group(1)}"
            f"§SEC{len(section_numbers) - 1}§"
        )

    protected_text = re.sub(
        section_number_pattern,
        replace_section_number,
        protected_text,
    )

    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=\.)\s+(?!\d)|(?<=[!?])\s+",
            protected_text,
        )
        if sentence.strip()
    ]

    restored_sentences = []

    for sentence in sentences:
        restored = re.sub(
            r"§UNIT(\d+)§",
            lambda match: units[
                int(match.group(1))
            ][-1:],
            sentence,
        )

        restored = re.sub(
            r"§ABB(\d+)§",
            lambda match: abbreviations[
                int(match.group(1))
            ][-1:],
            restored,
        )

        restored = re.sub(
            r"§SEC(\d+)§",
            lambda match: section_numbers[
                int(match.group(1))
            ][-1:],
            restored,
        )

        if restored.strip():
            restored_sentences.append(
                restored.strip()
            )

    return restored_sentences


def process_text_by_type(row):
    """Process text according to its content type and label."""

    text = row["text"]
    content_type = row.get("content_type", "")
    label = row.get("label", "")

    if (
        content_type == "table"
        or label
        in [
            "table",
            "table_header",
            "table_subheader",
        ]
    ):
        return [text.strip()] if text.strip() else []

    if content_type == "section":
        return [text.strip()] if text.strip() else []

    return safe_sentence_split(text)


def process_csv_pipeline(input_file, output_file):
    df = pd.read_csv(input_file)

    section_columns = [
        "section",
        "subsection",
        "subsubsection",
        "subsubsubsection",
    ]

    existing_columns = [
        column
        for column in section_columns
        if column in df.columns
    ]

    for column in existing_columns:
        df[column] = df[column].apply(
            remove_section_numbers
        )

    df = assign_sections_to_items(df)

    def convert_to_title_format(value):
        if isinstance(value, list) and value:
            return " -> ".join(value)

        return ""

    df["section_array"] = df[
        "enhanced_section_array"
    ].apply(convert_to_title_format)

    df["section_array"] = df[
        "section_array"
    ].apply(clean_quotes_and_whitespace)

    df = df[
        df["section_array"].notna()
        & (df["section_array"] != "")
    ]

    section_types = [
        "section",
        "subsection",
        "subsubsection",
        "subsubsubsection",
    ]

    if "content_type" in df.columns:
        df["is_section_header"] = df[
            "content_type"
        ].isin(section_types)
    else:
        df["is_section_header"] = False

    columns_to_keep = [
        "section_array",
        "content",
        "sort_id",
        "is_section_header",
    ]

    if "enumerated" in df.columns:
        columns_to_keep.append("enumerated")

    if "content_type" in df.columns:
        columns_to_keep.append("content_type")

    if "label" in df.columns:
        columns_to_keep.append("label")

    df = df[columns_to_keep]

    df = df.rename(
        columns={
            "section_array": "title",
            "content": "text",
        }
    )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
    )

    df["only_section"] = df.apply(
        detect_only_section,
        axis=1,
    )

    df["text"] = df.apply(
        process_text_by_type,
        axis=1,
    )

    if "label" in df.columns:
        df = df.drop(columns=["label"])

    df = (
        df.explode("text")
        .reset_index(drop=True)
    )

    df = df[
        df["text"].notna()
        & (df["text"] != "")
    ]

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
    )

    return df


if __name__ == "__main__":
    data_dir = Path("data")

    output_dir = (
        data_dir
        / "triplet_ready_data"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files_to_process = [
        data_dir
        / "ontology_label"
        / "table_A6A6M_14_labeled_v6.csv",
    ]

    for input_file in files_to_process:
        output_filename = (
            f"{input_file.stem}"
            "_enhanced_complete_triplet_ready.csv"
        )

        output_path = (
            output_dir
            / output_filename
        )

        process_csv_pipeline(
            input_file,
            output_path,
        )