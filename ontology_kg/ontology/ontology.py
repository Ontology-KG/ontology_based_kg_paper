import pandas as pd
import re

def preprocess_content(df, remove_keyword, column_name='content'):
    df_processed = df.copy().sort_values("sort_id")

    if "label" in df_processed.columns:
        df_processed = df_processed[~df_processed.label.isin(["page_footer", "page_header"])]

    for keyword in remove_keyword:
        if keyword:
            mask = df_processed[column_name].str.contains(keyword, case=False, na=False)
            df_processed = df_processed[~mask]

    mask_only_numbers = df_processed[column_name].str.strip().str.match(r'^\d+$', na=False)
    df_processed = df_processed[~mask_only_numbers]

    df_processed[column_name] = df_processed[column_name].str.replace('\n', ' ', regex=False)
    mask_single_alphabet = df_processed[column_name].str.strip().str.match(r'^[a-zA-Z]$', na=False)
    df_processed = df_processed[~mask_single_alphabet]

    df_processed[column_name] = df_processed[column_name].str.replace('--', '', regex=False)
    df_processed = df_processed[df_processed[column_name].astype(str).str.len() > 3]

    mask_empty = df_processed[column_name].str.strip().str.len() == 0
    df_processed = df_processed[~mask_empty]
    return df_processed


def parse_section_title(content):
    if not content or not isinstance(content, str):
        return None

    content = content.strip()
    if content in "API":
        return None

    annex_match = re.match(r'^Annex\s+([A-Z])(?:\s*\([^)]*\))?\s*(.*)$', content)
    if annex_match:
        letter = annex_match.group(1)
        rest = annex_match.group(2).strip()
        title = content if not rest else f"Annex {letter} {rest}"
        return _section_info('annex', 1, letter, title)

    long_text_match = re.match(r'^(\d+(?:\.\d+)*)\s+([^—–:;]+?)(?:[—–:;]\s*.*)?$', content)
    if long_text_match:
        number, title = long_text_match.groups()
        title = title.strip()
        if len(title) > 2:
            return _numbered_info('extracted_from_long', number, title)

    md_numbered_match = re.match(r'^#\s*((?:\d+(?:\.\d+)*)|(?:[A-Z](?:\.\d+)+))\s+(.+)$', content)
    if md_numbered_match:
        number, title = md_numbered_match.groups()
        return _numbered_info('markdown_numbered', number, title)

    numbered_match = re.match(r'^((?:\d+(?:\.\d+)*)|(?:[A-Z](?:\.\d+)+))\.?\s+(.+)$', content)
    if numbered_match:
        number, title = numbered_match.groups()
        title = title.rstrip('.;:')
        return _numbered_info('numbered_text', number, title)

    for pattern, sec_type in [
        (r'^SUPPLEMENTARY\s+REQUIREMENTS?\s*$', 'supplementary_requirements'),
        (r'^APPENDIXES?\s*$', 'appendix'),
        (r'^ANNEXES?\s*$', 'appendix')
    ]:
        if re.match(pattern, content, re.IGNORECASE):
            return _section_info(sec_type, 1, None, content)

    supp_match = re.match(r'^([A-Z]\d+)[.;:]?\s+(.+)$', content)
    if supp_match:
        number, title = supp_match.groups()
        title = title.rstrip('.;:')
        return _section_info('supplementary', 2, number, f"{number} {title}")

    supp_sub_match = re.match(r'^([A-Z]\d+)\.(\d+(?:\.\d+)*)[.;:]?\s+(.+)$', content)
    if supp_sub_match:
        main_num, sub_num, title = supp_sub_match.groups()
        number = f"{main_num}.{sub_num}"
        return _numbered_info('supplementary_numbered', number, title.rstrip('.;:'))

    number_only_match = re.match(r'^([A-Z](?:\.\d+)+)\s*', content)
    if number_only_match:
        number = number_only_match.group(1)
        return _number_only_info(number)
    return None


def _section_info(sec_type, level, number, title):
    return {
        'type': sec_type,
        'level': level,
        'number': number,
        'title': title,
        'full_title': title,
        'main_section_num': number,
        'sub_section_num': None,
        'subsub_section_num': None,
        'subsubsub_section_num': None
    }


def _numbered_info(sec_type, number, title):
    parts = number.split('.')
    return {
        'type': sec_type,
        'level': len(parts),
        'number': number,
        'title': title,
        'full_title': f"{number} {title}",
        'main_section_num': parts[0],
        'sub_section_num': parts[1] if len(parts) > 1 else None,
        'subsub_section_num': parts[2] if len(parts) > 2 else None,
        'subsubsub_section_num': parts[3] if len(parts) > 3 else None
    }


def _number_only_info(number):
    parts = number.split('.')
    return {
        'type': 'number_only',
        'level': len(parts),
        'number': number,
        'title': None,
        'full_title': None,
        'main_section_num': parts[0],
        'sub_section_num': parts[1] if len(parts) > 1 else None,
        'subsub_section_num': parts[2] if len(parts) > 2 else None,
        'subsubsub_section_num': parts[3] if len(parts) > 3 else None
    }


def label_sections_safe(df):
    df = df[~df['content'].isna()].sort_values(["file_id", "sort_id"]).reset_index(drop=True)
    current_main = current_sub = current_subsub = current_subsubsub = None
    pending = None
    rows = []

    for _, row in df.iterrows():
        content = str(row.get('content', '')).strip()
        parsed = parse_section_title(content)
        section_title_mark = 'label' in df.columns and row['label'] in ['title', 'section_title', 'header']

        if pending and not parsed and not content.startswith("Annex"):
            parsed = {**pending, 'title': content, 'full_title': f"{pending['number']} {content}"}
            section_title_mark = True
            pending = None
        elif pending:
            pending = None

        if parsed:
            lvl = parsed['level']
            if parsed['type'] in ['annex', 'supplementary_requirements', 'appendix']:
                current_main, current_sub, current_subsub, current_subsubsub = parsed['full_title'], None, None, None
            elif parsed['type'] == 'number_only':
                pending = parsed
            elif parsed['type'] in [
                'numbered', 'markdown_numbered', 'numbered_text',
                'supplementary', 'extracted_from_long', 'supplementary_numbered'
            ]:
                if lvl == 1:
                    current_main = parsed['full_title']
                    current_sub = current_subsub = current_subsubsub = None
                elif lvl == 2:
                    current_sub = parsed['full_title']
                    current_subsub = current_subsubsub = None
                elif lvl == 3:
                    current_subsub = parsed['full_title']
                    current_subsubsub = None
                elif lvl == 4:
                    current_subsubsub = parsed['full_title']

        new_row = row.copy()
        new_row['section'] = current_main
        new_row['subsection'] = current_sub
        new_row['subsubsection'] = current_subsub
        new_row['subsubsubsection'] = current_subsubsub
        new_row['is_section_title'] = section_title_mark
        rows.append(new_row)

    return pd.DataFrame(rows)


def add_table_footnote_mapping(df):
    df = df.copy()
    df['footnote_id'] = df.get('footnote_id', None)
    df['table_group_id'] = df.get('table_group_id', None)

    current_table_id = last_table_row_idx = None
    table_counter = 1
    TABLE_LABELS = {'table', 'table_row', 'table_cell', 'table_data', 'caption'}

    def _extract_table_number(text):
        m = re.match(r'^\s*TABLE\s+(\d+)\b', str(text), re.IGNORECASE)
        return m.group(1) if m else None

    def _match_letter_footnote(text):
        m = re.match(r'^\s*([A-Z])\s*[\.\)\-–—:]?\s+', str(text))
        return m.group(1) if m else None

    def _find_nearest_table_group(idx):
        for j in range(idx - 1, -1, -1):
            prev_gid = df.iloc[j].get('table_group_id')
            if isinstance(prev_gid, str) and prev_gid:
                return prev_gid
        return None

    for idx, row in df.iterrows():
        content = str(row.get('content', '')).strip()
        label = str(row.get('label', '') or '')

        num = _extract_table_number(content)
        if num:
            current_table_id = f"TABLE_{num}"
            df.at[idx, 'table_group_id'] = current_table_id
            last_table_row_idx = idx
            continue
        if label in TABLE_LABELS:
            if current_table_id is None:
                current_table_id = f"TABLE_{table_counter}"
                table_counter += 1
            df.at[idx, 'table_group_id'] = current_table_id
            last_table_row_idx = idx
            continue

        fn = _match_letter_footnote(content)
        if fn:
            df.at[idx, 'footnote_id'] = fn
            if current_table_id is not None and last_table_row_idx is not None:
                df.at[idx, 'table_group_id'] = current_table_id
            else:
                found = _find_nearest_table_group(idx)
                if found:
                    df.at[idx, 'table_group_id'] = found
                    current_table_id = found
            continue

        if label in {'section_header', 'heading'} and not _extract_table_number(content):
            current_table_id = None
            last_table_row_idx = None
            continue

    return df


def add_content_type(df):
    df = df.copy()
    prevs = {'section': None, 'subsection': None, 'subsubsection': None, 'subsubsubsection': None}
    content_types = []

    for _, row in df.iterrows():
        currs = {k: row.get(k) for k in prevs}
        label = str(row.get('label', '') or '')
        content_type = 'text'

        changed = [k for k in prevs if currs[k] != prevs[k] and pd.notna(currs[k])]
        if changed:
            content_type = changed[-1]

        if pd.notna(row.get('table_group_id')):
            content_type = 'table'
            if pd.notna(row.get('footnote_id')):
                content_type = 'table_footnote'

        content_types.append(content_type)
        prevs = currs

    df['content_type'] = content_types
    return df


def trans_row(row):
    for i, col in enumerate(['section', 'subsection', 'subsubsection', 'subsubsubsection']):
        if pd.notna(row[col]):
            text = str(row[col])
            if len(text) >= 150 or text.endswith('.'):
                for c in ['section', 'subsection', 'subsubsection', 'subsubsubsection'][i:]:
                    row[c] = None
                if row.get('content_type') not in ['table', 'table_footnote']:
                    row['content_type'] = 'text'
                break
    return row


def main(input_csv):
    df = pd.read_csv(input_csv)
    remove_keywords = [
        "Copyr", "Copyright", "©", "All rights reserved",
        "A6/A6M", "A578", "- 14", "www.",
        "API SPECIFICATION", "SUMMARY OF CHANGES", "2355 to 2983,"
    ]
    df_clean = preprocess_content(df, remove_keywords)
    labeled_df = label_sections_safe(df_clean)
    labeled_df = add_table_footnote_mapping(labeled_df)
    labeled_df = add_content_type(labeled_df)
    labeled_df = labeled_df.apply(trans_row, axis=1)
    return labeled_df

if __name__ == "__main__":
    input_csv = "YOUR_INPUT_FILE.csv"  # <-- Replace with your input CSV path

    output_csv = input_csv.replace('.csv', '_labeled.csv')
    labeled_df = main(input_csv)
    labeled_df.to_csv(output_csv, index=False, encoding='utf-8')
