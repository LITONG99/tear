import json
import random 
import torch
import numpy as np
import os 
from collections import defaultdict
from transformers import set_seed


def fix_seed(seed):
    random.seed(seed)
    if torch.cuda.is_available():
        #print("GPU CUDA is available!")
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
    else:
        #print("CUDA is not available! CPU is available!")
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    set_seed(seed)

def filer_data_by_headers(data, h):
    '''
    Hide the unknown headers from data
    data = {table: row: k2v}
    h = list
    '''
    new_data = {}
    for e in data:
        new_e = {}
        for rid in data[e]:
            new_row = {}
            att = data[e][rid]
            for k in att:
                if k in h: new_row[k] = att[k]
            if len(new_row)>0:
                new_e[len(new_e)]=new_row
        new_data[e] = new_e

    return new_data 


def get_headers_in_table(table, multiple_entity=False):
    h = []
    if multiple_entity:
        for entity in table:
            for rid in table[entity]: h += list(table[entity][rid].keys())
    else:
        for rid in table:
            h += list(table[rid].keys())
    return list(set(h))


def get_values_in_table(table, multiple_entity=False):
    h = []
    if multiple_entity:
        for entity in table:
            for rid in table[entity]: 
                for k in table[entity][rid]:
                    h += table[entity][rid][k]
    else:
        for rid in table:
            for k in table[rid]:
                h += table[entity][rid][k]
        
    return list(set(h))

def get_value_for_header_in_table(table, header, multiple_entity=False):
    res = []
    if multiple_entity:
        for entity in table:
            for rid in table[entity]: 
                if header in table[entity][rid]:
                    res += table[entity][rid][header]
    else:
        for rid in table:
            if header in table[rid]:
                res += table[rid][header]
    return res
        


def filter_empty_data(data):
    new_data = {}
    for index in data:
        new_data[index] = {}
        if not data[index]:
            continue
        for entity in data[index]:
            new_entity = {}
            for rid in data[index][entity]:
                if not rid.isdigit():
                    continue
                new_row = {}
                att = data[index][entity][rid]
                for k in att:
                    # core 
                    new_value = [str(v) for v in att[k] if len(str(v))>0]
                    if len(new_value)>0:        
                        new_row[k] = new_value
                if len(new_row)>0:
                    new_entity[len(new_entity)]=new_row
            new_data[index][entity] = new_entity

    return new_data 

def load_data(domain, data_path, level, update=[]):
    with open(f"{data_path}/{domain}_multi_clean.json", 'r') as fp:
        data = json.load(fp)

    with open(f"{data_path}/{domain}_multi_sd.json", 'r') as fp:
        schema = json.load(fp)

    with open(f"{data_path}/{domain}_multi_split.json", 'r') as fp:
        split = json.load(fp)

    train_index, valid_index, test_index = split[f'train{level}_index'], split[f'valid{level}_index'], split[f'test{level}_index']

    if level == '0':
        headers = set(schema[f'train1']+ schema['validation']+ schema[f'test1'])
        train_headers =  valid_headers = test_headers = list(headers)
    else:
        train_headers, valid_headers, test_headers = schema[f'train{level}'], schema['validation'], schema[f'test{level}']

    valid_headers += train_headers # during valiation, the train headers already known
    test_headers += valid_headers + train_headers 

    if len(update)>0:
        train_headers += update
        valid_headers += update    

    train_text = {}
    valid_text = {}
    test_text = {}

    train_data = {}
    valid_data = {}
    test_data = {}

    for index in train_index:
        train_data[index] = filer_data_by_headers(data[index]['attributes'], train_headers)
        train_text[index] = data[index]['text']

    for index in valid_index:
        valid_data[index] = filer_data_by_headers(data[index]['attributes'], valid_headers)
        valid_text[index] = data[index]['text']

    for index in test_index:
        test_data[index] = filer_data_by_headers(data[index]['attributes'], test_headers)
        test_text[index] = data[index]['text']

    return train_text, valid_text, test_text, train_data, valid_data, test_data, train_headers, valid_headers, test_headers


def load_header(data, entities, labeled_headers, keep_new=True):
    # Prepare for define
    entity_headers = defaultdict(list)
    for idx in data:
        if idx == 'extracted_version':
            continue
        for entity in data[idx]:
            if not entity in entities: continue
            tmp = get_headers_in_table(data[idx], multiple_entity=True)
            for h in tmp:
                if (keep_new and not h in labeled_headers) or (not keep_new and h in labeled_headers):
                    entity_headers[h].append(idx)
    return entity_headers


def save_at(data, path, file):
    os.makedirs(path, exist_ok=True)
    if file.endswith('json'):
        with open(f"{path}/{file}", 'w') as fp:
            json.dump(data, fp)
    else:
        torch.save(data, f'{path}/{file}')
    

def load_from(path, file):
    if file.endswith('json'):
        with open(f"{path}/{file}", 'r') as fp:
            return json.load(fp)
    else:
        tmp =  torch.load(f'{path}/{file}')
        return tmp
    

def load_infer_headers(path, stage,  extracted = None):
    infer_headers = load_from(path, f'header_previews_{stage}.json') #index --> header preview
    if extracted: # update with extracted table
        for index in infer_headers:
            if index in extracted and len(extracted[index])>0:
                tmp = get_headers_in_table(extracted[index], multiple_entity=True)
                if len(tmp)>0: infer_headers[index] = tmp
    return infer_headers


def load_tagged_values(path, stage,  text, extracted=None):
    tagged_values = load_from(path, f'value_previews_{stage}.json')
    res = {}
    for index in text:
        if extracted and index in extracted: # update with extracted table
            tmp = get_values_in_table(extracted[index], multiple_entity=True)
            if len(tmp)>0:
                res[index] = tmp
                continue
        if len(tagged_values[index])==0: # regard the text as a single value
            res[index] = [text[index]]
        else:
            res[index] = tagged_values[index]
    return res
    
def parse_response_json_to_table(text, entities):
    valid_json = True
    # Part1: try to load all tables
    entity_index = {}
    first_entity = entities[0]
    entity_index[first_entity] = text.find(f'"{first_entity}"')

    # Slice the string up to the accident_index to search backwards
    substring = text[:entity_index[first_entity]]
    # Find the last occurrence of '{' in this substring
    begin_index = substring.rfind('{')
    end_index = text.rfind('}')
    try:
        assert begin_index != -1 and end_index != -1
        response = json.loads(text[begin_index:end_index+1])
        return response, valid_json
    except: pass


    # Part 2: try to load table by table
    n_entity = len(entities)
    for entity in entities:
        entity_index[entity] = text.find(f'"{entity}"')
    
    for entity in entities:
        if entity_index[entity]<0:
            entity_index[entity] = 1000000

    entity_order = sorted([i for i in entities], key = lambda x: entity_index[x])
    
    res = {}
    for i, entity in enumerate(entity_order):
        if i == n_entity-1:
            substring = text[entity_index[entity]:]
        else:
            next_entity = entity_order[i+1]
            substring = text[entity_index[entity]:entity_index[next_entity]]
        beg_index = substring.find('{')
        end_index = substring.rfind('}')
        try:    
            entity_response = json.loads(substring[beg_index: end_index+1])
        except:
            entity_response = {}
            valid_json = False
        res[entity] = entity_response
    return res, valid_json

def parse_response_json_to_definition(text):
    valid_json = True
    begin_index = text.find('{')
    end_index = text.rfind('}')
    try:
        assert begin_index != -1 and end_index != -1
        response = json.loads(text[begin_index:end_index+1])
        return response, valid_json
    except: pass
    return {}, False

def parse_response_to_term(task, text):
    if task =='judge': terms = ['Two', 'One']
    for t in terms:
        if t in text:
            return t, True
    return terms[0], False 
