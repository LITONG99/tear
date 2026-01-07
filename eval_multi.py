import numpy as np
from sacrebleu import sentence_chrf
import bert_score
from tqdm import tqdm
from collections import defaultdict

bert_scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
metric_cache = {'c': dict(),'E':dict(), 'BS-scaled':dict()} # (string, string) cache; this could speed up a lot

def calc_data_similarity(tgt, pred, metric):
        if isinstance(tgt, dict):
            p, r, f = dictionary_matching_score(tgt, pred, metric)
            return f
        elif isinstance(tgt, list): # list of values, for multi-valued attributes
            p, r, f = set_matching_score(tgt, pred, metric)
            return f
        elif isinstance(pred, list):
            f = max([calc_data_similarity(tgt, _, metric) for _ in pred])
            return f
        else:
            if (tgt, pred) in metric_cache[metric]:
                return metric_cache[metric][(tgt, pred)]

            if metric == 'E':
                ret = int(tgt == pred)
            elif metric == 'c':
                ret = sentence_chrf(pred, [tgt, ]).score / 100
            elif metric == 'BS-scaled': 
                global bert_scorer
                ret = bert_scorer.score([pred, ], [tgt, ])[2].item()
                ret = min(max(ret, 0),1)
            else:
                raise ValueError(f"Metric cannot be {metric}")

            metric_cache[metric][(tgt, pred)] = ret
            return ret


def calc_similarity_matrix(tgt_data, pred_data, metric, display=False):
    if display: iter_ = tqdm(tgt_data, desc='Retrieving')
    else: iter_ = tgt_data
    tmp = [[calc_data_similarity(tgt, pred, metric) for pred in pred_data] for tgt in iter_]
    return np.array(tmp, dtype=float)
    

def get_f1(p, r):
    if isinstance(p, list):
        f1 = [get_f1(a,b) for a, b in zip(p, r)]
        return f1
    else:
        if p+r == 0.0: f1 = 0.0
        else: f1 = 2 * p * r / (p+r)
        return f1


def set_matching_score(X, Y, metric):
    '''
    X: groundtruth
    Y: prediction
    X, Y: list (or set) of strings, dictionaries, or lists.
    '''
    if len(X)==1 and len(Y)==1:
        s = calc_data_similarity(list(X)[0], list(Y)[0], metric)
        return s, s, s

    h_sim = calc_similarity_matrix(X, Y, metric)
    p = np.mean(np.max(h_sim, axis=0))
    r = np.mean(np.max(h_sim, axis=1))
    return p, r, get_f1(p,r)


def dictionary_matching_score(X, Y, metric):
    '''
    X: groundtruth
    Y: prediction
    X, Y: dictionaries, the values are list
    '''
    x_header, y_header = list(X.keys()), list(Y.keys())
    x_cell, y_cell = list(X.values()), list(Y.values())
    if len(X)==1 and len(Y)==1:
        s = calc_data_similarity(x_cell[0], y_cell[0], metric)
        return s, s, s

    h_sim = calc_similarity_matrix(x_header, y_header, metric) 
    n_tgt, n_pred = h_sim.shape
    
    header_match_p = np.argmax(h_sim, axis=0) # max in every column
    header_match_r = np.argmax(h_sim, axis=1) # max in every row
    
    c_sim = calc_similarity_matrix(x_cell, y_cell, metric)  
    p = np.mean(c_sim[header_match_p, np.arange(n_pred)])
    r = np.mean(c_sim[np.arange(n_tgt), header_match_r])
    return p ,r, get_f1(p,r)


def compute_metrics_multi(hyp_data, tgt_data):
    '''
    hyp_data: index -> entity-> rid -> header: value
    Note that there can be NO empty value (delete the header:value pair), NO empty row (delete the rid). 
    And there must be aligned entities (keep the entry even if there are no rows)
    '''
    res = {}
    all_metric = ['h-p','h-r','h-f','structured-p','structured-r','structured-f']
    for f in ['E', 'c', 'BS-scaled']:
        res[f] = defaultdict(list)
        print(f'Evaluating...similarity-f={f}')
        global metric_cache
        for index in tqdm(tgt_data, total=len(tgt_data)):
            tgt_table = tgt_data[index]
            hyp_table = hyp_data[index]
            
            entity_score = defaultdict(list)

            row_weight = []
            column_weight = []
            cell_weight = []
            for entity in tgt_table:
                tgt_entity = tgt_table[entity] #{rowid->dict}
                if not entity in hyp_table: hyp_entity = {}
                else: hyp_entity = hyp_table[entity]
                
                if len(tgt_entity)==0 and len(hyp_entity)==0: continue
                elif len(tgt_entity)==0:
                    for _ in all_metric:
                        entity_score[_].append(0)
                    
                    column_weight.append(0)
                    row_weight.append(0)
                    cell_weight.append(0)
                else:
                    tgt_headers = set([])
                    tgt_content = []
                    for rid in tgt_entity:
                        tgt_headers = tgt_headers | set(tgt_entity[rid].keys())
                        for _ in list(tgt_entity[rid].values()):
                            tgt_content += _

                    
                    column_weight.append(len(tgt_headers))

                    row_weight.append(len(tgt_entity))

                    if len(hyp_entity)==0:
                        for _ in all_metric:
                            entity_score[_].append(0)
                    else:
                        hyp_headers = set([])
                        cell_content = []
                        for rid in hyp_entity:
                            hyp_headers = hyp_headers | set(hyp_entity[rid].keys())
                            for _ in list(hyp_entity[rid].values()):
                                cell_content += _
                            
                        hp, hr, hf = set_matching_score(tgt_headers, hyp_headers, f)
                        cp, cr, cf = set_matching_score(list(tgt_entity.values()), list(hyp_entity.values()), f)
                        
                        entity_score['h-p'].append(hp)
                        entity_score['h-r'].append(hr)
                        entity_score['h-f'].append(hf)

                        entity_score['structured-p'].append(cp)
                        entity_score['structured-r'].append(cr)
                        entity_score['structured-f'].append(cf)

            if sum(row_weight)==0 or sum(column_weight)==0:
                continue
            row_weight = np.array(row_weight)/sum(row_weight)
            column_weight = np.array(column_weight)/sum(column_weight)

            for k in entity_score:
                tmp = np.array(entity_score[k])
                # weighted: only for multi-table cases, averaged by num of header / record
                if 'h' in k:
                    res[f][k].append( tmp @ column_weight)
                else:
                    res[f][k].append( tmp @ row_weight)

        for k in res[f]: res[f][k] = np.mean(res[f][k])*100
    
    res['print'] = []
    for content in ['h', 'structured']:
        for f in ['E', 'c', 'BS-scaled']:
            tmp = "%s:%s, Micro-precision = %.2f, recall = %.2f, f1 = %.2f;" % (content, f, 
                    res[f][f'{content}-p'], res[f][f'{content}-r'], res[f][f'{content}-f'])
            print(tmp)
            res['print'].append(tmp)
    return res


def compute_metrics_for_headers(hs, ts, max_n=None):
    pred_k = len(hs)
    beg_iter = 1
    end_iter = pred_k+1
    
    res = defaultdict(list)
    global metric_cache
    for metric in ['c', 'BS-scaled']:
        for k in range(beg_iter, end_iter):
            hp, hr, hf = set_matching_score(ts, hs[:k], metric)
            res[metric].append(float(hr))          

    res['print'] = []
    r_scores = {}

    for metric in ['c', 'BS-scaled']:
        tmp = "%s:%s" % ('r', metric)
        for k in range(beg_iter, end_iter):
            tmp += "@%d = %.2f, " % (k, res[metric][k-beg_iter])
            if max_n: # append the last value for consistent normalization
                r_scores[metric] =  res[metric] + [res[metric][-1]] * (max_n-len(res[metric]))
            else: r_scores[metric] =  res[metric]
            r_scores[metric] = sum(r_scores[metric])/len(r_scores[metric])
        res['print'].append(tmp)
    
    if max_n: 
        r_scores['mr'] = pred_k/max_n
        print("Recall-AUC-chrf: %.5f, \t Recall-AUC-bs: %.5f, \t Ratio: %.5f." % (r_scores['c'], r_scores['BS-scaled'], r_scores['mr']))
    else:
        print("Recall-AUC-chrf: %.5f, \t Recall-AUC-bs: %.5f." % (r_scores['c'], r_scores['BS-scaled']))
    res['average'] = r_scores
    return res

