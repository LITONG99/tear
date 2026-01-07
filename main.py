import argparse 
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

from utils import load_data, fix_seed, save_at, filter_empty_data, load_infer_headers, \
    get_headers_in_table, load_from, load_tagged_values, get_value_for_header_in_table, load_header
from retrieve import get_demonstration, sample_context 

from schema import process_schema_text, process_definition_examples, definition_schema, discover_schema,   domain_entities
from eval_multi import compute_metrics_multi, compute_metrics_for_headers
import random

from interact_llm import get_response_from_llm
from tear import hybrid_integration_strategy, rank_by_score

def extract(args):
    exp_name = f'{args.domain}-{args.llm_model}-lev{args.level}-{args.few_shot}-{args.random_seed}-{args.stage}'
    

    train_text, valid_text, test_text, \
    train_data, valid_data, test_data, \
    train_headers, valid_headers, _ = load_data(args.domain, args.data_path, args.level)

    labeled_text = train_text
    labeled_data = train_data
    labeled_headers = train_headers
    if args.stage =='test':
        labeled_text.update(valid_text)
        labeled_data.update(valid_data)
        labeled_headers += valid_headers
    else: 
        test_text = valid_text
        test_data = valid_data
        
    labeled_headers = set(labeled_headers) # note that set operation output random order.
    retrieved_examples =  get_demonstration(args, test_text, labeled_text, labeled_data)

    aux_data = {'known_headers':labeled_headers}   

    # get the the schema and previews for instruction
    aux_data['schema_text'] = process_schema_text(args.domain, aux_data['known_headers'])
    aux_data['header_preview'] = load_infer_headers(args.surrogate_checkpoint, args.stage)
    aux_data['value_preview'] = load_tagged_values(args.surrogate_checkpoint, args.stage, test_text)
    
    extracted= get_response_from_llm(args, test_text, retrieved_examples,  labeled_text, labeled_data,  save_response=True, aux_data=aux_data)

    print('Finishing', exp_name)
    
    extracted = filter_empty_data(extracted)
    save_at(extracted, f"{args.work_dir}/{args.task}_output", f'{exp_name}-table.json')
    
    metrics = compute_metrics_multi(extracted, test_data)
    save_at(metrics,f"{args.work_dir}/{args.task}_output",  f'{exp_name}-metrics.json')


def discover(args):
    exp_name = f'{args.domain}-{args.llm_model}-lev{args.level}-{args.few_shot}-{args.random_seed}-{args.stage}'
    args.exp_name = exp_name

    train_text, valid_text, test_text, \
    train_data, valid_data, test_data, \
    train_headers, valid_headers, _ = load_data(args.domain, args.data_path, args.level)

    labeled_text = train_text
    labeled_data = train_data
    labeled_headers = train_headers

    if args.stage =='test':
        labeled_text.update(valid_text)
        labeled_data.update(valid_data)
        labeled_headers += [h for h in valid_headers if not h in train_headers]
    else: 
        test_text = valid_text
        test_data = valid_data
   

    tmp = [_ for _ in range(len(labeled_headers))]  
    random.shuffle(tmp)
    labeled_headers = [labeled_headers[_] for _ in tmp]
    aux_data = {'known_headers':labeled_headers}    
    aux_data['discover_schema'] = discover_schema[args.domain]
    
    ## 1. Discover
    retrieved_examples =  get_demonstration(args, test_text, labeled_text, labeled_data)
    aux_data['extracted'] = load_from(f'{args.work_dir}/extract_output', f'{args.use_extracted}-table.json')
    
    aux_data['headers'] = {}
    for index in test_data:
        extracted = aux_data['extracted'][index]
        headers_in_table = [ _ for _ in get_headers_in_table(extracted, multiple_entity=True) if _ in labeled_headers]
        aux_data['headers'][index] = headers_in_table


    discovered= get_response_from_llm(args, test_text, retrieved_examples,  labeled_text, labeled_data,  save_response=True, aux_data=aux_data)
    discovered['extracted_version'] = args.use_extracted

    save_at(discovered, f"{args.work_dir}/{args.task}_output", f'{exp_name}_table.json')
   
    discovered = load_from(f"{args.work_dir}/{args.task}_output", f"{exp_name}_table.json")
    
    # header -> index --> value
    entity_new_headers = load_header(discovered, domain_entities[args.domain], labeled_headers, keep_new=True)
    entity_old_headers = load_header(labeled_data, domain_entities[args.domain], labeled_headers, keep_new=False)


    ## 2. Writing definition as condensed context
    context = {}
    # Sample context for known header+
    for h in labeled_headers:
        texts = [labeled_text[idx] for idx in entity_old_headers[h]]
        values = [get_value_for_header_in_table(labeled_data[idx], h, multiple_entity=True) for idx in entity_old_headers[h]]
        context[h] = sample_context(args.define_context_sample_n, texts=texts, values=values)

    # Sample context for new headers
    for h in entity_new_headers:
        texts = [test_text[idx] for idx in entity_new_headers[h]]
        values = [get_value_for_header_in_table(discovered[idx], h, multiple_entity=True) for idx in entity_new_headers[h]]
        context[h] = sample_context(args.define_context_sample_n, texts=texts, values=values)

    max_example_headers = 10
    tmp = [h for h in entity_old_headers]
    if len(tmp)>max_example_headers:
        random.shuffle(tmp)
    
    args.task = 'define'
    aux_data['definition_examples'] = process_definition_examples(args.domain, tmp[:max_example_headers], context)
    aux_data['definition_schema'] = definition_schema

    definition = get_response_from_llm(args, {h: 1 for h in entity_new_headers}, context,  labeled_text=None, labeled_table=None,
                                    save_response=True, aux_data=aux_data, response_type='def')
    

    frequency = {_: len(entity_new_headers[_]) for _ in entity_new_headers}
    delet_entries = []
    for h in definition:
        if len(definition[h])>0 and h in entity_new_headers:
                if isinstance(definition[h], dict):
                    definition[h] = list(definition[h].values())[0]
                else: pass
        else: 
            delet_entries.append(h)
    for _ in delet_entries: 
        del definition[_]
    save_at({"attributes":definition, "frequency": frequency}, f"{args.work_dir}/{args.task}_output", f'{exp_name}_proposals.json')


def integrate(args):
    exp_name = f'{args.domain}-{args.llm_model}-lev{args.level}-{args.few_shot}-{args.random_seed}-{args.stage}'
    args.exp_name = exp_name

    _, _, _, _, _, _, train_headers, valid_headers, _ = load_data(args.domain, args.data_path, args.level)

    labeled_headers = train_headers
    if args.stage =='test':
        labeled_headers += [h for h in valid_headers if not h in train_headers]

    tmp = [_ for _ in range(len(labeled_headers))]
    random.shuffle(tmp)
    labeled_headers = [labeled_headers[_] for _ in tmp]
   
    discovered = load_from(f"{args.work_dir}/discover_output", f"{exp_name}_table.json")
    entity_new_headers = load_header(discovered, domain_entities[args.domain], labeled_headers, keep_new=True)

    proposals = load_from(f"{args.work_dir}/define_output", f'{exp_name}_proposals.json')

   
    if args.strategy =='hybrid':
        new_attributes, meta_info = hybrid_integration_strategy(args, labeled_headers, proposals['attributes'], entity_new_headers)
        results = {"attributes":new_attributes, 'num_original':len(entity_new_headers)}
        results.update(meta_info)
        save_at(results, f"{args.work_dir}/{args.task}_output", f'{exp_name}_{args.strategy}_{args.delta_correction}_{args.diversity_kernel}.json')
    elif args.strategy == 'none':
        proposals['num_original']= len(entity_new_headers)
        save_at(proposals, f"{args.work_dir}/{args.task}_output", f'{exp_name}_{args.strategy}_{args.delta_correction}_{args.diversity_kernel}.json')
    else:
        raise NotImplementedError

    return


def ranking(args):
    exp_name = f'{args.domain}-{args.llm_model}-lev{args.level}-{args.few_shot}-{args.random_seed}-{args.stage}'
    _, _, _, _, _, _, train_headers, valid_headers, test_headers = load_data(args.domain, args.data_path, args.level)

    labeled_headers = train_headers
    if args.stage =='test':
        labeled_headers += [h for h in valid_headers if not h in train_headers]
    else:
        test_headers = valid_headers

    if args.strategy =='none':
        tmp = load_from(f'{args.work_dir}/define_output', f"{exp_name}_proposals.json")
    elif args.strategy == 'hybrid':
        save_name = f'{exp_name}_{args.delta_correction}_{args.diversity_kernel}'
        tmp = load_from(f'{args.work_dir}/judge_output', f"{exp_name}_{args.strategy}_{args.delta_correction}_{args.diversity_kernel}.json")
    else: 
        raise NotImplementedError
    
    aux_data = {'labeled_headers':labeled_headers} 
    if 'frequency' in tmp:
        aux_data['frequency'] = tmp['frequency']

    test_headers = [h for h in test_headers if not h in labeled_headers]
    ranked = rank_by_score(args, tmp['attributes'], test_headers, save_name, method=args.ranking_method, aux=aux_data)
    
    ranked_ql = [h for h , s in ranked]
    metric_path = f'{save_name}_{args.ranking_method}_{args.dp_version}_metrics.json'
    metrics = {}
    if test_headers: 
        # compute recall-AUC
        metrics.update(compute_metrics_for_headers(ranked_ql, test_headers, max_n=tmp['num_original']))
  
    res = {'metrics': metrics, 'ranked_ql':[(i, h, s) for i, (h, s) in enumerate(ranked)]}
    if save_name:
        print(metric_path) 
        save_at(res, f"{args.work_dir}/{args.task}_output", f"{metric_path}")
    
    return res



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='rank', choices=['extract', 'discover', 'integrate', 'rank'])
    parser.add_argument('--update', type=bool, default=False, help='For extraction with selected text-dirven attributes.')
    parser.add_argument("--stage", type=str, default='valid', choices=['valid', 'test'], 
                                                            help='If test, the valid and training data are concatenated as labeled sample pool.')

    parser.add_argument("--data_path", type=str, default='data')
    parser.add_argument("--work_dir", type=str, default='')
    parser.add_argument("--domain", type=str, default='Incidents', choices=['Incidents', 'Weather'])
    parser.add_argument("--level", type=str, default='3', help='0 for complete schema', choices=['0','1','2','3'])

    # backbone
    parser.add_argument("--llm_model", type=str, default='Llama', choices=['Llama', 'Qwen']) 
    parser.add_argument("--llm_max_token", type=int, default=1024) 
    parser.add_argument("--llm_temperature", type=int, default=0.5) 
    parser.add_argument("--llm_sample_p", type=int, default=0.5)
    parser.add_argument("--llm_batch_size", type=int, default=4) 
    parser.add_argument("--random_seed", type=int, default=2025)

    
    # previews and Referecing Module
    parser.add_argument("--retrieve", type=str, default='embedding-header-value', help='Referencing Module')
    parser.add_argument("--few_shot", type=int, default=5)
    parser.add_argument("--surrogate_checkpoint", type=str, default='surrogate_model/{domain}_{level}_{llm_model}{random_seed}')
    parser.add_argument("--embedding_model_dir", type=str, default='')
    parser.add_argument("--embedding_path", type=str, default='embedding')

    # discovery
    parser.add_argument("--use_extracted", type=str, default='{domain}-{llm_model}-lev{level}-[5, 5, 5]-{random_seed}-{stage}', help='update the demonstrations for discovery.') 
    
    # integration
    parser.add_argument("--strategy", type=str, default='hybrid', choices=['none', 'hybrid'])
    parser.add_argument("--diversity_kernel", type=str, default='chrf', choices=['chrf'])
    parser.add_argument("--define_context_sample_n", type=int, default=5, help='used to define the proposal')
    parser.add_argument("--delta_correction", type=float, default=0, help='delta = estimated_delta(1-correction)')
   
    # ranking
    parser.add_argument("--ranking_method", type=str, default='maximum', help='relevance, similarity, frequency, average, maximum')
    parser.add_argument("--knee_s", type=float, default=1, help='For numerical estimated truncted point of the relevance-based recmendation list. Greater value leads to later knee.')
    parser.add_argument("--dp_version", type=int, default=10, help='Seed to randomize the permutation order in Descriptive Prefix. Only used for relevance-based ranking.')


    args = parser.parse_args()
    fix_seed(args.random_seed)
    args.retrieve = args.retrieve.split('-')

    args.surrogate_checkpoint = args.surrogate_checkpoint.format(domain=args.domain, level = args.level, 
                                                                     llm_model='', 
                                                                     random_seed=args.random_seed)
                                                                     
    
    args.few_shot = [args.few_shot]*len(args.retrieve)

    if args.task =='extract':
        extract(args)
    elif args.task =='discover': 
        assert args.level!='0'
        args.use_extracted = args.use_extracted.format(domain=args.domain, level = args.level, 
                                                        llm_model=args.llm_model, random_seed=args.random_seed,
                                                        stage=args.stage)
        discover(args)
    elif args.task == 'integrate':  
        assert args.level!='0'
        integrate(args)
    else: # ranking
        assert args.level != '0' 
        assert args.strategy!= 'hybrid' or args.ranking_method in ['relevance','average','maximum']
        ranking(args)
