import transformers
from tqdm import tqdm
import torch
from collections import defaultdict
import time

from utils import parse_response_json_to_definition, parse_response_json_to_table, parse_response_to_term, fix_seed, save_at
from prompt import compose_prompt
from schema import domain_entities


LLM_DIR = {
    'Qwen':'Qwen2.5-14B-Instruct',
    'Llama': 'Meta-Llama-3-8B-Instruct'
}

pipeline = None
tokenizer = None

def get_probability_from_llm(args, prefix, candidates, save_response=True, show_progress=True):
    global pipeline
    if pipeline: pass
    else: 
        if args.llm_model =='Qwen':
            pipeline = transformers.pipeline("text-generation", model=LLM_DIR[args.llm_model], 
                                        model_kwargs={'dtype':'auto'}, device_map="auto")
        else:
            pipeline = transformers.pipeline("text-generation", model=LLM_DIR[args.llm_model], 
                                        model_kwargs={'dtype':torch.bfloat16}, device_map="auto")
    fix_seed(args.random_seed)
    
    llm_model = pipeline.model 
    tokenizer = pipeline.tokenizer

    # Tokenize the prefix to find where h starts
    s_inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
    device  = llm_model.device
    s_input_ids = s_inputs['input_ids'].to(device)
    prefix_len = s_input_ids.shape[1]

    score = {}
    response = {}
    # Get model outputs
    with torch.no_grad():
        if show_progress: _iter = tqdm(candidates, desc='Compute probability')
        else: _iter = candidates
        for h in _iter:
            # Tokenize the full text
            inputs = tokenizer(prefix+h+',', return_tensors="pt", add_special_tokens=False)
            input_ids = inputs['input_ids'].to(device)

            outputs = llm_model(input_ids=input_ids)
            logits = outputs.logits  # Shape: [1, sequence_length, vocab_size]
        
            # Calculate probabilities using softmax
            probs = torch.softmax(logits, dim=-1)
            
            # Extract probabilities for tokens in h
            token_probs = []
            multiplied_prob = 1
            for i in range(prefix_len-1, len(input_ids[0])):
                # The probability of the next token (i+1) given previous context (up to i)
                token_id = input_ids[0, i].item()
                token_prob = probs[0, i-1, token_id].item()
                token = tokenizer.decode(token_id)
                token_probs.append((token, token_prob))
                multiplied_prob *= token_prob
            score[h] = multiplied_prob
            response[h] = token_probs

    if save_response:
        save_at(response, f'{args.work_dir}/response_output/', file=f'{args.task}_{int(time.time() * 1000) }_response.json')  
    return score


def get_response_from_llm(args, test_text, retrieved_examples, labeled_text, labeled_table, save_response=True, aux_data={}, response_type='table', show_progress=True):
    global pipeline

    # Initialize LLM
    if args.llm_model =='Qwen':
        if not pipeline: pipeline = transformers.pipeline("text-generation", model=LLM_DIR[args.llm_model], model_kwargs={'dtype':'auto'}, device_map="auto")
        terminators = [pipeline.tokenizer.eos_token_id,pipeline.tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    elif args.llm_model=='Llama':
        if not pipeline: pipeline = transformers.pipeline("text-generation", model=LLM_DIR[args.llm_model], model_kwargs={'dtype':torch.bfloat16}, device_map="auto")
        terminators = [pipeline.tokenizer.eos_token_id,pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    else:
        raise NotImplementedError
    fix_seed(args.random_seed)
    batch_size = args.llm_batch_size


    # prepare output
    test_index = list(test_text.keys())
    group_n = len(test_index)
    predicted = defaultdict(dict) 
    all_responses = {}

    with torch.no_grad():
        if show_progress: _iter = tqdm(range(0, group_n, batch_size), desc=args.task)
        else: _iter = range(0, group_n, batch_size)
        for i in _iter:
            batch_index = test_index[i: min(group_n, i+batch_size)]        
            
            messages = []
            for index in batch_index: 
                messages.append(compose_prompt(args, {index: test_text[index]}, labeled_text, labeled_table, retrieved_examples, aux_data))
            
            
            prompts = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.llm_temperature>0:
                outputs = pipeline(prompts, max_new_tokens=args.llm_max_token, eos_token_id=terminators, pad_token_id = pipeline.tokenizer.eos_token_id,
                                do_sample=True, temperature=args.llm_temperature, top_p=args.llm_sample_p)        
            else:
                outputs = pipeline(prompts, max_new_tokens=args.llm_max_token, eos_token_id=terminators, pad_token_id = pipeline.tokenizer.eos_token_id,
                                do_sample=False) 

            # Answer
            responses = [outputs[j][0]["generated_text"][len(prompt):] for j, prompt in enumerate(prompts)]

            for j, r in enumerate(responses):
                all_responses[batch_index[j]] = r
               
                if response_type =='table':
                    pred, valid_json = parse_response_json_to_table(r, domain_entities[args.domain])
                elif response_type=='def': # definition
                    pred, valid_json = parse_response_json_to_definition(r)
                else: # term
                    pred, valid = parse_response_to_term(args.task, r)
                predicted[batch_index[j]] = pred

    if save_response:
        save_at(all_responses, f'{args.work_dir}/response_output/', file=f'{args.task}_{int(time.time() * 1000) }_response.json')   

    return predicted