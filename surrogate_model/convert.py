import os
import json
import numpy as np

CELL_SEP = ' | '
LINE_SEP = '<NEWLINE>'
SPAN_SEP = ' & '
chr1 = CELL_SEP.strip()
chr2 = SPAN_SEP.strip()
chr3 = LINE_SEP.strip()

ENTITY_MAP = {
    'Incidents': ['Accident', 'Victim', 'Suspect'],
    'Weather': ['Weather']
}

def load_json_file(filepath):
    with open(filepath, 'r') as fp:
        return json.load(fp)
    
def save_json_file(filepath, filename,  data):
    os.makedirs(filepath, exist_ok=True)
    with open(os.path.join(filepath, filename), 'w') as fp:
        json.dump(data, fp)
            

def save_text_file(filepath, filename,  content_list):
    os.makedirs(filepath, exist_ok=True)
    with open(os.path.join(filepath, filename), 'w') as fp:
        fp.write('\n'.join(content_list))

def convert_entity_to_table(entity_name, entity_attributes, relevant_headers):
    if not relevant_headers:
        return f"{entity_name}:"
    
    header_row = CELL_SEP + CELL_SEP.join(sorted(relevant_headers)) + CELL_SEP
    
    data_rows = []
    for record_id in entity_attributes:
        row_cells = []
        for header in sorted(relevant_headers):
            if header in entity_attributes[record_id]:
                values = entity_attributes[record_id][header]
                cell_content = SPAN_SEP.join(values) if values else ''
            else:
                cell_content = ''
            row_cells.append(cell_content)
        
        if any(row_cells):
            row_str = CELL_SEP + CELL_SEP.join(row_cells) + CELL_SEP
            data_rows.append(row_str)
    
    if data_rows:
        table_content = header_row + LINE_SEP + LINE_SEP.join(data_rows)
        return f"{entity_name}: {LINE_SEP}{table_content}"
    else:
        return f"{entity_name}:"


def convert_data_to_text_and_table(data, indices, headers):
    text_list = []
    table_list = []
    max_header_count = 0
    
    header_set = set(headers)
    
    for data_id in indices:
        text_list.append(data[data_id]['text'])
        
        entity_tables = []
        for entity_name, entity_attributes in data[data_id]['attributes'].items():
            entity_headers = set()
            for record_id in entity_attributes:
                for header in entity_attributes[record_id]:
                    if header in header_set:
                        entity_headers.add(header)
            
            if len(entity_headers) > max_header_count:
                max_header_count = len(entity_headers)
            
            entity_table = convert_entity_to_table(
                entity_name, entity_attributes, entity_headers
            )
            entity_tables.append(entity_table)
        
        table_list.append(f' {LINE_SEP} '.join(entity_tables))
    
    return text_list, table_list, max_header_count


def process_dataset_version(data, split_info, schema_info, stage, version, output_base_path):
    index_key = f'{stage}{version}_index'
    indices = split_info[index_key]
    
    if version == 0:
        headers = (schema_info['train1'] + schema_info['validation'] + schema_info['test1'])
    else:
        headers = (schema_info[f'train{version}'] + schema_info['validation'])
        if stage == 'test':
            headers += schema_info[f'test{version}']
    
    text_list, table_list, max_headers = convert_data_to_text_and_table(
        data, indices, headers
    )
    
    print(f" Stage: {stage}, Level: {version}, Max headers: {max_headers}")
    
    output_dir = f'{output_base_path}_{version}'
  
    save_text_file(output_dir, f'{stage}.data', table_list)
    save_text_file(output_dir, f'{stage}.text', text_list)


def parse_text_to_table(text):

    text = text.split(chr3)
    header_region = text[0]
    non_header_region = text[1:]

    headers = [h.strip() for h in header_region.split(chr1)]
    headers = [_ for _ in headers if len(_)>0]
    spans = [] 
    for row in non_header_region:
        tmp =  row.replace(chr2, chr1).split(chr1)
        tmp = [_.strip() for _ in tmp]
        spans += [_ for _ in tmp if len(_)>0]

    return headers, spans


def extract_table_by_name(text, name):
    lines = [line.strip() for line in text.replace(LINE_SEP, "\n").strip().splitlines()]
    if name + ":" not in lines:
        return ""
    
    table = []
    for line in lines[lines.index(name + ":") + 1:]:
        if line.endswith(":"):
            break
        table.append(line.strip())
    
    return f" {LINE_SEP} ".join(table)


def convert_to_att_form(file, entities):
    data = {}
    with open(file) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            headers = []
            spans = []
            for entity in entities:
                table_text = extract_table_by_name(line, entity)
                cur_headers, cur_spans = parse_text_to_table(table_text)
                headers += cur_headers
                spans += cur_spans 
              
            data[idx] = (headers, spans)
    return data


def extract_attributes(data_dict, entities):
    attributes = []
    for entity in entities:
        if entity in data_dict:
            for row_id in data_dict[entity]:
                row = data_dict[entity][row_id]
                attributes.extend(row.keys())
    return attributes


def extract_values(data_dict, entities):
    values = []
    for entity in entities:
        if entity in data_dict:
            for row_id in data_dict[entity]:
                row = data_dict[entity][row_id]
                for header in row:
                    values.extend(row[header])
    return values


def get_header_preview(hyp_headers, all_headers):
    return {h:1 for h in all_headers if h in hyp_headers}


if __name__ == "__main__":
    DOMAIN = 'Incidents'
    LEVEL = 0
    
    BASE_DATA_PATH = f'tear/data'
    BASE_OUTPUT_PATH = f'tear/surrogate_model/surrogate_data/{DOMAIN}'

    SEED = 2025
    TABLE_PATH = f"tear/surrogate_model/{DOMAIN}_{LEVEL}_{SEED}"
    
    clean_file = f'{BASE_DATA_PATH}/{DOMAIN}_multi_clean.json'
    schema_file = f'{BASE_DATA_PATH}/{DOMAIN}_multi_sd.json'
    split_file = f'{BASE_DATA_PATH}/{DOMAIN}_multi_split.json'
    print("Loading data files...")
    data = load_json_file(clean_file)
    schema = load_json_file(schema_file)
    split = load_json_file(split_file)
  

    #task = 'serialization'
    task = 'generate'
    
    if task=='serialization': 
        print(f"Loaded {len(data)} data points")
        for stage in ['train', 'valid', 'test']:
            process_dataset_version(data, split, schema, stage, LEVEL, BASE_OUTPUT_PATH)
        print("\nSerialization completed successfully!")
        
    else:
        # generate_previews given inference results.
        print(f"\nGenerating previews for {TABLE_PATH}")
        for stage in ['valid','test']:
            indices = split[f'{stage}{LEVEL}_index']
            entities = ENTITY_MAP[DOMAIN]
            
            if LEVEL == 0: 
                headers = set(schema['train1'] + schema['validation'] + schema['test1'])
            elif stage == 'valid':
                headers = set(schema[f'train{LEVEL}'])
            elif stage == 'test':
                headers = set(schema[f'train{LEVEL}'] + schema['validation'])
            
            # Convert raw sequences to previews
            hyp_file = f"{TABLE_PATH}/checkpoint_average_best-3.pt.{stage}_constrained.out.text"
            hyp_data = convert_to_att_form(hyp_file, entities)
            
            
            header_results = {}
            value_results = {}
            for idx, hyp_idx in zip(indices, hyp_data.keys()):
                header_results[idx] = get_header_preview(hyp_data[hyp_idx][0], headers)
                value_results[idx] = list(set(hyp_data[hyp_idx][1]))
    
            save_json_file(TABLE_PATH, f'header_previews_{stage}.json', header_results)
            save_json_file(TABLE_PATH, f'value_previews_{stage}.json', value_results)
            
        print("Previews generated successfully!")