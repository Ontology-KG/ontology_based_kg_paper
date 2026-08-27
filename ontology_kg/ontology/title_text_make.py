import pandas as pd
import re
import csv
import os

def remove_section_numbers(value):
    """섹션 제목에서 앞의 번호 패턴을 제거하는 함수 (제목에서만, 내용에서는 보존)"""
    if pd.isna(value) or value == '':
        return value
    if not isinstance(value, str):
        return str(value)
    
    cleaned = value.strip()
    pattern = r'^[A-Za-z]*[0-9]+(\.[0-9]+)*\s+'
    result = re.sub(pattern, '', cleaned)
    return result.strip()

def create_section_array(row):
    """section 계층을 배열 형태로 변환하는 함수"""
    section_array = []
    
    if pd.notna(row['section']) and str(row['section']).strip():
        section_array.append(str(row['section']).strip())
    
    if pd.notna(row['subsection']) and str(row['subsection']).strip():
        section_array.append(str(row['subsection']).strip())
    
    if pd.notna(row['subsubsection']) and str(row['subsubsection']).strip():
        section_array.append(str(row['subsubsection']).strip())
    
    if pd.notna(row['subsubsubsection']) and str(row['subsubsubsection']).strip():
        section_array.append(str(row['subsubsubsection']).strip())
    
    return section_array

def clean_quotes_and_whitespace(val):
    """따옴표와 공백을 반복적으로 제거"""
    prev = None
    while prev != val:
        prev = val
        val = val.strip(' "\'')
    return val

def detect_only_section(row):
    """title의 마지막 부분과 text가 동일한지 확인하는 함수"""
    title = str(row['title']).strip()
    text = str(row['text']).strip()
    
    if not title or not text:
        return False
    
    if ' -> ' in title:
        last_title_part = title.split(' -> ')[-1].strip()
    else:
        last_title_part = title.strip()
    
    text_without_number = re.sub(r'^\d+(\.\d+)*\.?\s*', '', text).strip()
    text_without_semicolon = text_without_number.rstrip(';').strip()
    
    if 'Scope' in title or '4.1.' in text:
        print(f"   title: '{title}' → last_part: '{last_title_part}'")
        print(f"   text: '{text}' → cleaned: '{text_without_semicolon}'")
    

    if last_title_part.lower() == text_without_semicolon.lower():
        print(f"[DEBUG] 완전 일치 발견: '{last_title_part}' == '{text_without_semicolon}'")
        return True
    
    if (last_title_part.lower() in text_without_semicolon.lower() and 
        len(last_title_part) > 5):  # 너무 짧은 문자열은 제외
        print(f"[DEBUG] title이 text에 포함됨: '{last_title_part}' in '{text_without_semicolon}'")
        return True
    
    if (text_without_semicolon.lower() in last_title_part.lower() and 
        len(text_without_semicolon) > 5):
        print(f"[DEBUG] text가 title에 포함됨: '{text_without_semicolon}' in '{last_title_part}'")
        return True
    
    title_words = set(last_title_part.lower().split())
    text_words = set(text_without_semicolon.lower().split())
    
    if len(title_words) > 0 and len(text_words) > 0:
        common_words = title_words.intersection(text_words)
        similarity = len(common_words) / max(len(title_words), len(text_words))
        if similarity > 0.7:  # 70% 이상 유사
            print(f"[DEBUG] 높은 유사도 발견: {similarity:.2f} - '{last_title_part}' vs '{text_without_semicolon}'")
            return True
    
    return False

def assign_sections_to_items(df):
    """sort_id 순서대로 처리하면서 계층 구조를 추적하고 각 행에 할당"""
    
    df_sorted = df.sort_values('sort_id').copy()
    
    enhanced_arrays = []
    
    current_section = ""
    current_subsection = ""
    current_subsubsection = ""
    current_subsubsubsection = ""
    current_table_header = None
    current_table_subheader = None
    
    for idx, row in df_sorted.iterrows():
        content_type = row.get('content_type', '')
        label = row.get('label', '')
        
        row_section_array = create_section_array(row)
        
        if len(row_section_array) > 0:
            if len(row_section_array) >= 1 and row_section_array[0].strip():
                current_section = row_section_array[0].strip()
                current_subsection = ""
                current_subsubsection = ""
                current_subsubsubsection = ""
            
            if len(row_section_array) >= 2 and row_section_array[1].strip():
                current_subsection = row_section_array[1].strip()
                current_subsubsection = ""
                current_subsubsubsection = ""
            
            if len(row_section_array) >= 3 and row_section_array[2].strip():
                current_subsubsection = row_section_array[2].strip()
                current_subsubsubsection = ""
            
            if len(row_section_array) >= 4 and row_section_array[3].strip():
                current_subsubsubsection = row_section_array[3].strip()
        
        elif content_type == 'section':
            content = str(row.get('content', '')).strip()
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
        
        # 테이블 헤더 처리
        if label == 'table_header':
            current_table_header = row['content']
            current_table_subheader = None
            enhanced_arrays.append(current_section_array.copy())
            
        elif label == 'table_subheader':
            current_table_subheader = row['content']
            enhanced_arrays.append(current_section_array.copy())
            
        elif label in ['table', 'table_footnote', 'table_note']:
            section_array = current_section_array.copy()
            
            if label == 'table':
                if current_table_header:
                    section_array.append(current_table_header)
                if current_table_subheader:
                    section_array.append(current_table_subheader)
                    
            elif label in ['table_footnote', 'table_note']:
                if current_table_header:
                    section_array.append(current_table_header)
            
            enhanced_arrays.append(section_array)
        
        else:
            enhanced_arrays.append(current_section_array.copy())
    
    df_sorted['enhanced_section_array'] = enhanced_arrays

    return df_sorted

def safe_sentence_split(text):
    """단위와 약어를 보호하면서 안전하게 문장을 분리하는 함수"""
    
    unit_pattern = r'(\d+(?:\s*[-/]\s*\d+)?\s*-?(?:µin|µm|in|mm|ft|cm|m|lb|kg|ksi|MPa|°F|°C))\.'
    standalone_unit_pattern = r'\b(µin|µm|in|mm|ft|cm|m|lb|kg|ksi|MPa|°F|°C)\.'
    units = []
    
    def replace_unit(match):
        units.append(match.group(0))
        return f"{match.group(1)}§UNIT{len(units) - 1}§"
    

    protected_text = re.sub(unit_pattern, replace_unit, text)
    protected_text = re.sub(standalone_unit_pattern, replace_unit, protected_text)
    

    abbreviation_pattern = r'(\b(?:Fed|Std|No|etc|vs|Mr|Mrs|Dr|Prof|Fig|Table|Spec|Vol|Ch|Sec|Art|Para|Inc|Corp|Ltd|Co|Ave|St|Rd|Blvd|min|max|temp|approx)\s*)\.'
    

    dotted_abbreviation_pattern = r'(\b[A-Z](?:\.[A-Z])+)\.'
    
    abbreviations = []
    
    def replace_abbreviation(match):
        abbreviations.append(match.group(0))
        return f"{match.group(1)}§ABB{len(abbreviations) - 1}§"
    
    def replace_dotted_abbreviation(match):
        abbreviations.append(match.group(0))
        return f"{match.group(1)}§ABB{len(abbreviations) - 1}§"
    
    protected_text = re.sub(abbreviation_pattern, replace_abbreviation, protected_text, flags=re.IGNORECASE)
    protected_text = re.sub(dotted_abbreviation_pattern, replace_dotted_abbreviation, protected_text)
    

    section_number_pattern = r'(\b[A-Z]\d+(?:\.\d+)*)\.'
    section_numbers = []
    
    def replace_section_number(match):
        section_numbers.append(match.group(0))
        return f"{match.group(1)}§SEC{len(section_numbers) - 1}§"
    
    protected_text = re.sub(section_number_pattern, replace_section_number, protected_text)
    

    sentences = [s.strip() for s in re.split(r'(?<=\.)\s+(?!\d)|(?<=[!?])\s+', protected_text) if s.strip()]
    

    restored_sentences = []
    for sentence in sentences:
        restored = re.sub(r'§UNIT(\d+)§', lambda m: units[int(m.group(1))][-1:], sentence)
        restored = re.sub(r'§ABB(\d+)§', lambda m: abbreviations[int(m.group(1))][-1:], restored)
        restored = re.sub(r'§SEC(\d+)§', lambda m: section_numbers[int(m.group(1))][-1:], restored)
        if restored.strip():
            restored_sentences.append(restored.strip())
    
    return restored_sentences

def process_text_by_type(row):
    """content_type과 label에 따라 텍스트를 다르게 처리"""
    text = row['text']
    content_type = row.get('content_type', '')
    label = row.get('label', '')
    
    if content_type == 'table' or label in ['table', 'table_header', 'table_subheader']:
        # 테이블 관련 컨텐츠는 분리하지 않음
        return [text.strip()] if text.strip() else []
    elif content_type == 'section':
        # Section 타입도 분리하지 않음 (번호나 제목이므로)
        return [text.strip()] if text.strip() else []
    else:
        # 일반 텍스트와 list_item은 안전한 문장 분리 사용
        return safe_sentence_split(text)

def process_csv_pipeline(input_file, output_file):
    df = pd.read_csv(input_file)

    section_columns = ['section', 'subsection', 'subsubsection', 'subsubsubsection']
    existing_columns = [col for col in section_columns if col in df.columns]
    
    for col in existing_columns:
        df[col] = df[col].apply(remove_section_numbers)    
    
    df = assign_sections_to_items(df)
    
    def convert_to_title_format(x):
        if isinstance(x, list) and len(x) > 0:
            return ' -> '.join(x)
        else:
            return ""
    
    df['section_array'] = df['enhanced_section_array'].apply(convert_to_title_format)
    df['section_array'] = df['section_array'].apply(clean_quotes_and_whitespace)
    
    before_filter = len(df)
    df = df[df['section_array'].notna() & (df['section_array'] != '')]
    after_filter = len(df)
    
    section_types = ['section', 'subsection', 'subsubsection', 'subsubsubsection']
    df['is_section_header'] = df.get('content_type', '').isin(section_types)
    
    columns_to_keep = ['section_array', 'content', 'sort_id', 'is_section_header']
    
    if 'enumerated' in df.columns:
        columns_to_keep.append('enumerated')
    if 'content_type' in df.columns:
        columns_to_keep.append('content_type')
    if 'label' in df.columns:
        columns_to_keep.append('label')
    
    df = df[columns_to_keep]
    
    df = df.rename(columns={
        'section_array': 'title',
        'content': 'text'
    })
    
    df['text'] = df['text'].fillna("").astype(str)
    
    df['only_section'] = df.apply(detect_only_section, axis=1)
    only_section_count = df['only_section'].sum()
    
    if only_section_count > 0:
        only_section_rows = df[df['only_section'] == True]
        for idx, row in only_section_rows.head(3).iterrows():
            print(f"   - title='{row['title']}'")
            print(f"     text='{row['text']}'")
            print(f"     only_section={row['only_section']}")
    before_explode = len(df)
    df['text'] = df.apply(process_text_by_type, axis=1)
    
    if 'label' in df.columns:
        df = df.drop(['label'], axis=1)
    
    df = df.explode('text').reset_index(drop=True)
    after_explode = len(df)
    
    before_empty_filter = len(df)
    df = df[df['text'].notna() & (df['text'] != '')]
    after_empty_filter = len(df)
    
    section_header_count = df['is_section_header'].sum()
    only_section_count = df['only_section'].sum()
    total_count = len(df)
    regular_text_count = total_count - section_header_count
    

    test_patterns = ['9-in.', 'U.S.', 'Fed.', 'Std.']
    for pattern in test_patterns:
        matching_rows = df[df['text'].astype(str).str.contains(pattern, na=False)]
        if len(matching_rows) > 0:
            sample_text = matching_rows.iloc[0]['text']
    
    
    only_section_data = df[df['only_section'] == True]
    if len(only_section_data) > 0:
        
        for idx, row in only_section_data.head(2).iterrows():
            last_title_part = row['title'].split(' -> ')[-1] if ' -> ' in row['title'] else row['title']
            print(f"   - 마지막 title: '{last_title_part}'")
            print(f"     text: '{row['text']}'")
            print(f"     is_section_header: {row['is_section_header']}")
    
    # 섹션 헤더 샘플 확인
    section_headers = df[df['is_section_header'] == True]
    if len(section_headers) > 0:
        print(f"\n[DEBUG] 섹션 헤더 샘플 ({len(section_headers)}개):")
        for idx, row in section_headers.head(2).iterrows():
            print(f"   - title='{row['title']}', text='{row['text'][:40]}...', only_section={row['only_section']}")
    
    # 일반 텍스트 샘플 확인  
    regular_text = df[(df['is_section_header'] == False) & (df['only_section'] == False)]
    if len(regular_text) > 0:
        print(f"\n[DEBUG] 순수 일반 텍스트 샘플 ({len(regular_text)}개):")
        for idx, row in regular_text.head(2).iterrows():
            print(f"   - title='{row['title']}', text='{row['text'][:40]}...'")
    
    # 1.3, 1.4, 1.5, 1.6 확인
    target_texts = ['1.3', '1.4', '1.5', '1.6']
    print(f"\n[DEBUG] 특정 텍스트 확인:")
    for target in target_texts:
        matching_rows = df[df['text'].astype(str).str.contains(target, na=False)]
        if len(matching_rows) > 0:
            for idx, row in matching_rows.iterrows():
                flags = []
                if row['is_section_header']: flags.append("섹션헤더")
                if row['only_section']: flags.append("only_section")
                if not flags: flags.append("일반텍스트")
                print(f"   - '{target}' 발견: title='{row['title']}' ({', '.join(flags)})")
    
    df.to_csv(output_file, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)

    return df

if __name__ == "__main__":
    # 파일을 저장할 기본 디렉토리
    output_base_dir = "/home/jiin/SPEC/data/triplet_ready_data/"
    
    # input file
    files_to_process = [
        "/home/jiin/SPEC/data/ontology_label/table_A6A6M_14_labeled_v6.csv",
        #"/home/jiin/SPEC/data/ontology_label/A578A578M_07_parsing_labeled_v4.csv",
        #"/home/jiin/SPEC/data/ontology_label/API 2W 6TH(2019)_parsing_labeled_v4.csv"
    ]

    for input_file in files_to_process:
        base_filename = os.path.basename(input_file)

        filename_without_ext, _ = os.path.splitext(base_filename)
        output_filename = f"{filename_without_ext}_enhanced_complete_triplet_ready.csv"
        output_path = os.path.join(output_base_dir, output_filename)
        
        process_csv_pipeline(input_file, output_path)