import numpy as np
from tqdm import tqdm
from vendi_score import vendi
import time
import torch
import random

from eval_multi import calc_similarity_matrix
from schema import attribute_schema,  relevance_prefix, process_entity_schema_text
from interact_llm import get_probability_from_llm, get_response_from_llm
from kneed import KneeLocator
from utils import load_from

from embedding import get_embedding_from_llm, initialize_embedder


from collections import defaultdict

def vendi_score(sim):
    # https://github.com/vertaix/Vendi-Score
    sim = torch.Tensor(sim)
    return vendi.score_K(sim, normalize=True) 


def get_delta(sim):
    tmp = torch.max(sim-torch.eye(len(sim)), dim=0).values
    maxi_sim = [[1, tmp.max()], [tmp.max(), 1]]
    eps_2 = vendi_score(maxi_sim)-1
    return eps_2


def get_suspicious_sets(sim, k, delta, eps=0):
    m = len(sim)

    H = set(range(0, k))
    U = set(range(k, m))
     
    eps = 1 + delta*(1-eps) 

    start_time = time.perf_counter()
    overall_start_time = start_time
    
    res={}
    edges = {} # idx -> set of int

    # compute edges
    for i in H | U:
        edges[i] = {}
        res[tuple([i])] = 1
        for j in range(max(i+1, k), m):
            cur_p = [i,j]
            cur_sim = sim[np.ix_(cur_p, cur_p)]
            tmp = vendi_score(cur_sim) 
            if tmp<eps:
                edges[i][j] = tmp
                res[tuple(sorted([i,j]))] = float(tmp)

    # intialzied t with |p|=2
    t = {}
    count = 0
    for i in edges:
        for j in edges[i]:
            # set -> (diverse, search_set)
            search_set = set(edges[i].keys()) & set(edges[j].keys())
            count += len(search_set)
            t[str([i,j])[1:-1]] = (edges[i][j], search_set)

    print(f'Initialize search with {len(t)} edges, {count} active branch to be searched.')

    # search set of a \in [x_1, ..., x_n]: choose n-1 from [x_1, ...x_n] + {a}, pass the test
    #
    start_len = 3
    num_div_test=0
    last_div_test = 0

    while len(t)>0: # Outer loop
        start_time = time.perf_counter()
        next_t = {}
        count = 0
        for p in tqdm(t, desc=f'Search |p|={start_len}'): # Inner loop
            div, search_set = t[p]
            p = [int(_) for _ in p.split(', ')]
            
            # expand from p
            for i in search_set:
                cur_search_set = search_set 
                cur_div = div

                # Quick pruning
                flag = True
                for j in range(0, start_len-1):
                    key = str(p[:j] + p[j+1:] + [i])[1:-1]
                    if key in t:
                        div_a, search_set_a = t[key]
                        cur_div = max(cur_div, div_a)
                        cur_search_set = cur_search_set & search_set_a
                    else:
                        flag = False
                        break
                if not flag: continue
                
                cur_p = p + [i]
                cur_sim = sim[np.ix_(cur_p, cur_p)]
                ## Core Operation
                tmp = vendi_score(cur_sim)
                
                num_div_test += 1 
                ##
                if tmp<eps:
                    # update 
                    count += len(cur_search_set)
                    next_t[str(cur_p)[1:-1]] = (max(cur_div, tmp), cur_search_set)
                    res[tuple(sorted(cur_p))] = float(tmp)

        tmp = time.perf_counter() - start_time
        print(f"Execution {num_div_test-last_div_test} test for {start_len}: {tmp} seconds (~{tmp/60:.1f} min); {len(next_t)} valid set, {count} active branch.")
        
        last_div_test = num_div_test
        start_len += 1
        count = 0
        t = next_t

    res = [[k, res[k]] for k in res]
    meta_info = {
        'pruning time':  round(time.perf_counter() - overall_start_time, 4),
        'max_suspecious_size':  start_len -1,
        'num_suspecious_set': len(res)
    }
    print(meta_info)
    return res, meta_info


def solve_scp(U, subsets):
    remaining = U.copy()
    cover = []
    while remaining:
        subset = max(subsets, key=lambda s: len(s & remaining)+0.00001*len(s)) # this simple trick helps to find local maximal
        cover.append(subset)
        remaining -= subset 

    print(f'Get a solution with {len(cover)} subsets.')
    return cover
    
def hybrid_integration_strategy(args, known_headers, definition, new_headers):
    # update the definition for known attributes
    schema_definition = process_entity_schema_text(args.domain, known_headers, return_json=False)
    for h in known_headers: definition[h] = schema_definition[h]

    aux_data = {'attribute_schema': attribute_schema, 'schema_text': process_entity_schema_text(args.domain, known_headers)}
    
    header_names = []
    for h in known_headers: header_names.append(h)
    for h in new_headers: header_names.append(h)

    # compute delta
    sim = calc_similarity_matrix(header_names, header_names, metric='c')
    sim = torch.tensor((sim+sim.T)/2)
    n = len(sim)
    k = len(known_headers)
    delta = get_delta(sim[:k, :k])
    print('estimated delta:', delta)

    suspicious_set_2, meta_info = get_suspicious_sets(sim, k, delta, args.delta_correction)  

    # initialize W=P*
    W = [set(_[0]) for _ in suspicious_set_2] + [{i} for i in range(n)]
    B = set(range(n))
    candidates = {}
  
    llm_call = 0
    judge_one = 0
    judge_two = 0
    while len(B)>0 and len(W)>0:
        Q = solve_scp(B, W) # find a greedy cover of elements.
        print(f'Solving for |B|={len(B)}, |W|={len(W)}. Get {len(Q)} sets.')
        for q in Q:
            q = q & B
            if len(q)==1:
                B = B - q
                q = list(q)
                if q[0]<k: continue # this is a known attribute
                name = header_names[q[0]]
                candidates[name] = definition[name]
            else:
                # LLM analysis
                args.task = 'judge'
                merge_set = [header_names[_] for _ in q]
                
                judge_term = get_response_from_llm(args, {1: merge_set}, definition, labeled_text=None, labeled_table=None, 
                                                   save_response=False, aux_data=aux_data, response_type='term', show_progress=False)
                llm_call += 1
                if judge_term[merge_set[0]]!='Two': # rsp=yes
                    judge_one += 1
                    W = [w for w in W if len(q & w)==0]
                    B = B-q
                    if any(_<k for _ in q): # known attribute
                        continue 
                    else:
                        tmp = [_ for _ in range(len(known_headers))]
                        # Descriptive Prefix
                        prefix = relevance_prefix[args.domain] + ', '.join([known_headers[_] for _ in tmp])+', '
                        probability = get_probability_from_llm(args, prefix, merge_set, save_response=False, show_progress=False)
                        max_key = max(probability, key=probability.get)
                        candidates[max_key] = definition[max_key]
                else: 
                    judge_two += 1
                    W = [w for w in W if w != q]

    print(f'LLM call {llm_call} ,judge_one {judge_one}, judge_two {judge_two}')

    meta_info.update({'llm_call': llm_call, 'llm_judge_one':judge_one, 'llm_judge_two': judge_two,
                       'delta_correction':float(args.delta_correction)})
    
    return candidates, meta_info


def distilled_relevance_score(args, candidates, aux, save_name):
    # Prepare prefix for LLM query
    tmp = [_ for _ in range(len(aux['labeled_headers']))]
    for _ in range(args.dp_version): random.shuffle(tmp)

    prefix = relevance_prefix[args.domain] + ", ".join([aux['labeled_headers'][_] for _ in tmp]) + ', '
    
    # Get relevance probabilities from LLM
    probabilities = get_probability_from_llm(args, prefix, candidates, save_response=True)
    ranked = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    # Find knee point if applicable
    knee_x = 0
    if args.knee_s > 0:
        relevance_scores = [score for _, score in ranked]
        kneedle = KneeLocator(range(len(relevance_scores)), relevance_scores, S=args.knee_s, curve="convex", direction="decreasing")
        if kneedle.knee_y is not None:
            knee_x = relevance_scores.index(kneedle.knee_y)

    return ranked, knee_x

def aggregated_relevance(args, candidates, aux, save_name, method): 
    aggregated_scores = defaultdict(list)
    
    # Collect scores from all relevance runs
    for run_id in range(1, args.dp_version + 1):
        args.dp_version = run_id
        ranked_for_run, _ = distilled_relevance_score(args, candidates, aux, f"{save_name}_relevance_{run_id}")
        
        for header, score in ranked_for_run:
            aggregated_scores[header].append(score)
    
    # Apply aggregation function
    if method == 'average': final_scores = {h: sum(scores) / len(scores) for h, scores in aggregated_scores.items()}
    elif method == 'maximum': final_scores = {h: max(scores) for h, scores in aggregated_scores.items()}
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    # Sort by aggregated score
    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)


def rank_by_score(args, prediction, test_headers=None, save_name=None, method='relevance', aux={}):
  
    candidates = list(prediction.keys())

    if method=='frequency':
        frequency = aux['frequency']
        candidates.sort(key=frequency.get, reverse=True)
        ranked = [(_ ,int(frequency[_])) for _ in candidates]
    elif method =='similarity':
        labeled_header_embedding = load_from(args.embedding_path, f'{args.domain}_labeled_embedding.pt')
        header_names = []
        tmp = []
        for h in aux['labeled_headers']:
            tmp.append(labeled_header_embedding[h])
            header_names.append(h)
        k_embedding = torch.stack(tmp, dim=0)

        emb_tokenizer, emb_model = initialize_embedder(args)
        new_embeddings = {}
        get_embedding_from_llm(new_embeddings, prediction,  llm_tokenizer=emb_tokenizer, llm_model=emb_model)
        # save_at(new_embeddings, args.embedding_path, f'{save_name}.pt')
        new_embeddings = torch.stack([new_embeddings[h] for h in candidates], axis=0)

        sim = torch.mm(new_embeddings, k_embedding.T)
        similarity, _  = sim.max(dim=-1)
        simialrity  = list(similarity)
        res = {}
        for i, h in enumerate(candidates):
            res[h] = float(simialrity[i])
        candidates.sort(key=res.get, reverse=True)
        ranked = [(_ ,res[_]) for _ in candidates]
    elif method == 'maximum' or method =='average':
        ranked = aggregated_relevance(args, candidates, aux, save_name, method)
    elif method =='relevance':
        ranked, _ = distilled_relevance_score(args, candidates, aux, save_name)

    
    return ranked
