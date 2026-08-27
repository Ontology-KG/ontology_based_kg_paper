import argparse
import re
from pathlib import Path

import pandas as pd


SECTION_COLUMNS = [
    "section",
    "subsection",
    "subsubsection",
    "subsubsubsection",
]

TABLE_LABELS = {
    "table",
    "table_row",
    "table_cell",
    "table_data",
    "caption",
}

DEFAULT_REMOVE_KEYWORDS = [
    "Copyr",
    "Copyright",
    "©",
    "All rights reserved",
    "A6/A6M",
    "A578",
    "- 14",
    "www.",
    "API SPECIFICATION",
    "SUMMARY OF CHANGES",
    "2355 to 2983,",
]


def preprocess_content(df, remove_keywords, column_name="content"):
    df_processed = df.copy().sort_values("sort_id")

    if "label" in df_processed.columns:
        df_processed = df_processed[
            ~df_processed["label"].isin(["page_footer", "page_header"])
        ]

    for keyword in remove_keywords:
        if keyword:
            mask = df_processed[column_name].str.contains(
                keyword,
                case=False,
                na=False,
            )
            df_processed = df_processed[~mask]

    mask_only_numbers = (
        df_processed[column_name]
        .str.strip()
        .str.match(r"^\d+$", na=False)
    )
    df_processed = df_processed[~mask_only_numbers]

    df_processed[column_name] = df_processed[column_name].str.replace(
        "\n",
        " ",
        regex=False,
    )

    mask_single_alphabet = (
        df_processed[column_name]
        .str.strip()
        .str.match(r"^[a-zA-Z]$", na=False)
    )
    df_processed = df_processed[~mask_single_alphabet]

    df_processed[column_name] = df_processed[column_name].str.replace(
        "--",
        "",
        regex=False,
    )

    df_processed = df_processed[
        df_processed[column_name].astype(str).str.len() > 3
    ]

    mask_empty = (
        df_processed[column_name].str.strip().str.len() == 0
    )
    df_processed = df_processed[~mask_empty]

    return df_processed


def parse_section_title(content):
    if not content or not isinstance(content, str):
        return None

    content = content.strip()

    if content == "API":
        return None

    annex_match = re.match(
        r"^Annex\s+([A-Z])(?:\s*\([^)]*\))?\s*(.*)$",
        content,
    )

    if annex_match:
        letter = annex_match.group(1)
        rest = annex_match.group(2).strip()
        title = content if not rest else f"Annex {letter} {rest}"

        return _section_info(
            "annex",
            1,
            letter,
            title,
        )

    long_text_match = re.match(
        r"^(\d+(?:\.\d+)*)\s+([^—–:;]+?)(?:[—–:;]\s*.*)?$",
        content,
    )

    if long_text_match:
        number, title = long_text_match.groups()
        title = title.strip()

        if len(title) > 2:
            return _numbered_info(
                "extracted_from_long",
                number,
                title,
            )

    md_numbered_match = re.match(
        r"^#\s*((?:\d+(?:\.\d+)*)|(?:[A-Z](?:\.\d+)+))\s+(.+)$",
        content,
    )

    if md_numbered_match:
        number, title = md_numbered_match.groups()

        return _numbered_info(
            "markdown_numbered",
            number,
            title,
        )

    numbered_match = re.match(
        r"^((?:\d+(?:\.\d+)*)|(?:[A-Z](?:\.\d+)+))\.?\s+(.+)$",
        content,
    )

    if numbered_match:
        number, title = numbered_match.groups()
        title = title.rstrip(".;:")

        return _numbered_info(
            "numbered_text",
            number,
            title,
        )

    section_patterns = [
        (
            r"^SUPPLEMENTARY\s+REQUIREMENTS?\s*$",
            "supplementary_requirements",
        ),
        (
            r"^APPENDIXES?\s*$",
            "appendix",
        ),
        (
            r"^ANNEXES?\s*$",
            "appendix",
        ),
    ]

    for pattern, section_type in section_patterns:
        if re.match(pattern, content, re.IGNORECASE):
            return _section_info(
                section_type,
                1,
                None,
                content,
            )

    supplementary_match = re.match(
        r"^([A-Z]\d+)[.;:]?\s+(.+)$",
        content,
    )

    if supplementary_match:
        number, title = supplementary_match.groups()
        title = title.rstrip(".;:")

        return _section_info(
            "supplementary",
            2,
            number,
            f"{number} {title}",
        )

    supplementary_subsection_match = re.match(
        r"^([A-Z]\d+)\.(\d+(?:\.\d+)*)[.;:]?\s+(.+)$",
        content,
    )

    if supplementary_subsection_match:
        main_number, sub_number, title = (
            supplementary_subsection_match.groups()
        )

        number = f"{main_number}.{sub_number}"

        return _numbered_info(
            "supplementary_numbered",
            number,
            title.rstrip(".;:"),
        )

    number_only_match = re.match(
        r"^([A-Z](?:\.\d+)+)\s*",
        content,
    )

    if number_only_match:
        return _number_only_info(
            number_only_match.group(1)
        )

    return None


def _section_info(section_type, level, number, title):
    return {
        "type": section_type,
        "level": level,
        "number": number,
        "title": title,
        "full_title": title,
        "main_section_num": number,
        "sub_section_num": None,
        "subsub_section_num": None,
        "subsubsub_section_num": None,
    }


def _numbered_info(section_type, number, title):
    parts = number.split(".")

    return {
        "type": section_type,
        "level": len(parts),
        "number": number,
        "title": title,
        "full_title": f"{number} {title}",
        "main_section_num": parts[0],
        "sub_section_num": (
            parts[1] if len(parts) > 1 else None
        ),
        "subsub_section_num": (
            parts[2] if len(parts) > 2 else None
        ),
        "subsubsub_section_num": (
            parts[3] if len(parts) > 3 else None
        ),
    }


def _number_only_info(number):
    parts = number.split(".")

    return {
        "type": "number_only",
        "level": len(parts),
        "number": number,
        "title": None,
        "full_title": None,
        "main_section_num": parts[0],
        "sub_section_num": (
            parts[1] if len(parts) > 1 else None
        ),
        "subsub_section_num": (
            parts[2] if len(parts) > 2 else None
        ),
        "subsubsub_section_num": (
            parts[3] if len(parts) > 3 else None
        ),
    }


def label_sections_safe(df):
    df = (
        df[~df["content"].isna()]
        .sort_values(["file_id", "sort_id"])
        .reset_index(drop=True)
    )

    current_main = None
    current_sub = None
    current_subsub = None
    current_subsubsub = None

    pending = None
    rows = []

    for _, row in df.iterrows():
        content = str(
            row.get("content", "")
        ).strip()

        parsed = parse_section_title(content)

        section_title_mark = (
            "label" in df.columns
            and row["label"]
            in [
                "title",
                "section_title",
                "header",
            ]
        )

        if (
            pending
            and not parsed
            and not content.startswith("Annex")
        ):
            parsed = {
                **pending,
                "title": content,
                "full_title": (
                    f"{pending['number']} {content}"
                ),
            }

            section_title_mark = True
            pending = None

        elif pending:
            pending = None

        if parsed:
            level = parsed["level"]

            if parsed["type"] in [
                "annex",
                "supplementary_requirements",
                "appendix",
            ]:
                current_main = parsed["full_title"]
                current_sub = None
                current_subsub = None
                current_subsubsub = None

            elif parsed["type"] == "number_only":
                pending = parsed

            elif parsed["type"] in [
                "numbered",
                "markdown_numbered",
                "numbered_text",
                "supplementary",
                "extracted_from_long",
                "supplementary_numbered",
            ]:
                if level == 1:
                    current_main = parsed["full_title"]
                    current_sub = None
                    current_subsub = None
                    current_subsubsub = None

                elif level == 2:
                    current_sub = parsed["full_title"]
                    current_subsub = None
                    current_subsubsub = None

                elif level == 3:
                    current_subsub = parsed["full_title"]
                    current_subsubsub = None

                elif level == 4:
                    current_subsubsub = parsed["full_title"]

        new_row = row.copy()

        new_row["section"] = current_main
        new_row["subsection"] = current_sub
        new_row["subsubsection"] = current_subsub
        new_row["subsubsubsection"] = current_subsubsub
        new_row["is_section_title"] = section_title_mark

        rows.append(new_row)

    return pd.DataFrame(rows)


def add_table_footnote_mapping(df):
    df = df.copy()

    df["footnote_id"] = df.get(
        "footnote_id",
        None,
    )

    df["table_group_id"] = df.get(
        "table_group_id",
        None,
    )

    current_table_id = None
    last_table_row_idx = None
    table_counter = 1

    def extract_table_number(text):
        match = re.match(
            r"^\s*TABLE\s+(\d+)\b",
            str(text),
            re.IGNORECASE,
        )

        return match.group(1) if match else None

    def match_letter_footnote(text):
        match = re.match(
            r"^\s*([A-Z])\s*[\.\)\-–—:]?\s+",
            str(text),
        )

        return match.group(1) if match else None

    def find_nearest_table_group(idx):
        for previous_idx in range(
            idx - 1,
            -1,
            -1,
        ):
            previous_group_id = df.iloc[
                previous_idx
            ].get("table_group_id")

            if (
                isinstance(previous_group_id, str)
                and previous_group_id
            ):
                return previous_group_id

        return None

    for idx, row in df.iterrows():
        content = str(
            row.get("content", "")
        ).strip()

        label = str(
            row.get("label", "") or ""
        )

        table_number = extract_table_number(
            content
        )

        if table_number:
            current_table_id = (
                f"TABLE_{table_number}"
            )

            df.at[
                idx,
                "table_group_id",
            ] = current_table_id

            last_table_row_idx = idx
            continue

        if label in TABLE_LABELS:
            if current_table_id is None:
                current_table_id = (
                    f"TABLE_{table_counter}"
                )
                table_counter += 1

            df.at[
                idx,
                "table_group_id",
            ] = current_table_id

            last_table_row_idx = idx
            continue

        footnote_id = match_letter_footnote(
            content
        )

        if footnote_id:
            df.at[
                idx,
                "footnote_id",
            ] = footnote_id

            if (
                current_table_id is not None
                and last_table_row_idx is not None
            ):
                df.at[
                    idx,
                    "table_group_id",
                ] = current_table_id

            else:
                nearest_group = (
                    find_nearest_table_group(idx)
                )

                if nearest_group:
                    df.at[
                        idx,
                        "table_group_id",
                    ] = nearest_group

                    current_table_id = (
                        nearest_group
                    )

            continue

        if (
            label
            in {
                "section_header",
                "heading",
            }
            and not extract_table_number(content)
        ):
            current_table_id = None
            last_table_row_idx = None

    return df


def add_content_type(df):
    df = df.copy()

    previous_sections = {
        column: None
        for column in SECTION_COLUMNS
    }

    content_types = []

    for _, row in df.iterrows():
        current_sections = {
            column: row.get(column)
            for column in SECTION_COLUMNS
        }

        content_type = "text"

        changed_sections = [
            column
            for column in SECTION_COLUMNS
            if (
                current_sections[column]
                != previous_sections[column]
                and pd.notna(
                    current_sections[column]
                )
            )
        ]

        if changed_sections:
            content_type = changed_sections[-1]

        if pd.notna(
            row.get("table_group_id")
        ):
            content_type = "table"

            if pd.notna(
                row.get("footnote_id")
            ):
                content_type = "table_footnote"

        content_types.append(content_type)
        previous_sections = current_sections

    df["content_type"] = content_types

    return df


def normalize_section_rows(row):
    for index, column in enumerate(
        SECTION_COLUMNS
    ):
        if pd.notna(row[column]):
            text = str(row[column])

            if (
                len(text) >= 150
                or text.endswith(".")
            ):
                for section_column in (
                    SECTION_COLUMNS[index:]
                ):
                    row[section_column] = None

                if row.get(
                    "content_type"
                ) not in [
                    "table",
                    "table_footnote",
                ]:
                    row["content_type"] = "text"

                break

    return row


def process_file(
    input_csv,
    remove_keywords=None,
):
    df = pd.read_csv(input_csv)

    if remove_keywords is None:
        remove_keywords = DEFAULT_REMOVE_KEYWORDS

    df_clean = preprocess_content(
        df,
        remove_keywords,
    )

    labeled_df = label_sections_safe(
        df_clean
    )

    labeled_df = add_table_footnote_mapping(
        labeled_df
    )

    labeled_df = add_content_type(
        labeled_df
    )

    labeled_df = labeled_df.apply(
        normalize_section_rows,
        axis=1,
    )

    return labeled_df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Label section and table structures "
            "in parsed industrial standard documents."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the parsed input CSV file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the labeled output CSV file.",
    )

    args = parser.parse_args()

    output_path = args.output

    if output_path is None:
        output_path = args.input.with_name(
            f"{args.input.stem}_labeled.csv"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labeled_df = process_file(args.input)

    labeled_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()