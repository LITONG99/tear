from collections import defaultdict
import torch
import random
from utils import get_headers_in_table, get_values_in_table, load_from, load_infer_headers, load_tagged_values
from tqdm import tqdm
from eval_multi import calc_similarity_matrix

def set_sim_qk(a: tuple, b:tuple):
    seta = set(a)
    setb = set(b)
    return len(seta ^ setb)+0.01*len(setb-seta)+1e-6

def retrieve(args, query_text, query_headers, query_values, labeled_text, labeled_data, device='cuda'): 
    retrieved = defaultdict(list)

    labeled_index = list(labeled_text.keys())
    test_index = list(query_text.keys()) 

    for mode, fs in zip(args.retrieve, args.few_shot):
        if fs<=0: continue

        if mode=='embedding':
            with torch.no_grad():
                embedding =load_from(args.embedding_path, f"{args.domain}_text_embedding.pt")
                test_embedding = torch.stack([embedding[index] for index in test_index], dim=0).to(device)
                labeled_embedding = torch.stack([embedding[index] for index in labeled_index ], dim=1).to(device)
            
                sim = torch.matmul(test_embedding, labeled_embedding)
                _, numerical_indices = torch.topk(sim, fs, dim=1, largest=True)

                numerical_indices = numerical_indices.cpu()

                for i, index in enumerate(test_index): retrieved[index] += [labeled_index [j] for j in numerical_indices[i]]
        
        elif 'header' in mode: #
            sorted_query_headers = {}
            test_groups = defaultdict(list)
            
            for index in test_index:    
                d = tuple(sorted(query_headers[index]))
                sorted_query_headers[index] = d
                test_groups[d].append(index)
            
            res = retrieve_by_field(labeled_text, labeled_data, test_index, sorted_query_headers, test_groups, fs)
            for index in res: retrieved[index] += res[index]

        elif 'value' in mode:
            labeled_values = {}
            for index in labeled_index:    
                d = get_values_in_table(labeled_data[index], multiple_entity=True)
                labeled_values[index] = d

            queries = [query_values[_] for _ in test_index]
            keys = []
            for idx in labeled_index:
                if len(labeled_values[idx])>0: keys.append(labeled_values[idx])
                else: keys.append(labeled_text[index])
            sim = calc_similarity_matrix(queries, keys, metric='c', display=True) 
            _, numerical_indices = torch.topk(torch.Tensor(sim), fs, dim=1, largest=True)
            for i, index in enumerate(test_index):
                retrieved[index] += [labeled_index[j] for j in numerical_indices[i]]    
        else:
            raise NotImplementedError


    return retrieved

def retrieve_by_field(labeled_text, labeled_data, test_index, test_infer_table, test_groups, fs):
    res = defaultdict(list)

    labeled_index = list(labeled_text.keys())

    labeled_groups = defaultdict(list)
    for index in labeled_index:
        d = get_headers_in_table(labeled_data[index], multiple_entity=True)
        labeled_groups[tuple(sorted(d))].append(index)
    key_combinations = list(labeled_groups.keys()) 
    test_key_combinations = {_:i for i, _ in enumerate(test_groups.keys())}

    sim = [[set_sim_qk(test_comb, comb) for comb in key_combinations] for test_comb in test_key_combinations]
    _, numerical_comb_indices = torch.topk(torch.Tensor(sim), fs, dim=1)
    
    for index in tqdm(test_index, desc='Retrieving-header-utility'):
        if index in test_infer_table:
            test_comb = test_infer_table[index]
            best_matched_combs = numerical_comb_indices[test_key_combinations[test_comb]]
   
            current_shot = 0
            for comb in best_matched_combs:
                comb = key_combinations[comb]
                random.shuffle(labeled_groups[comb])
                res[index] += labeled_groups[comb][:fs-current_shot] 
                current_shot = len(res[index])
                if current_shot>=fs: break
        else:
            res[index] += []

    return res

def sample_context(k, texts, values, mode= 'text_length'):
    '''
    - texts: List
    - values: List[List]
    '''
    assert len(texts) == len(values)
    
    if mode=='text_length': # longer text, larger weight
        indexes = [i for i in range(len(texts))]
        weights = [len(tmp) for tmp in texts]
        samples = random.choices(indexes, weights=weights, k=k)

        res = []
        for s in samples:
            res.append({'text': texts[s], 'values':values[s]})
        return res    
    else:
        raise NotImplementedError


def get_demonstration(args, test_text, labeled_text, labeled_data):
    header_preview = None
    value_preview = None
    for r in args.retrieve:
        if r == 'header':
            if args.task == 'extract' or not args.use_extracted: extracted = None
            else: extracted = load_from(f'{args.work_dir}/extract_output', f'{args.use_extracted}-table.json')
            header_preview = load_infer_headers(args.surrogate_checkpoint, args.stage, extracted=extracted)
        elif r == 'value':
            if args.task == 'extract' or not args.use_extracted: extracted = None
            else: extracted = load_from(f'{args.work_dir}/extract_output', f'{args.use_extracted}-table.json')
            value_preview = load_tagged_values(args.surrogate_checkpoint, args.stage, test_text, extracted=extracted)
        else:
            pass
    retrieved_examples = retrieve(args, test_text, header_preview, value_preview,  labeled_text, labeled_data)
    return retrieved_examples