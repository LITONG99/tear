from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
from torch import Tensor
from tqdm import tqdm
from utils import fix_seed
import argparse
import os
import json

embedder_tokenizer = None
embedder_model = None

def initialize_embedder(args):
    global embedder_tokenizer
    global embedder_model
    if embedder_model:
        return embedder_tokenizer, embedder_model
    
    embedder_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_dir)
    embedder_model = AutoModel.from_pretrained(args.embedding_model_dir, device_map='balanced_low_0')
    fix_seed(args.random_seed)
    return embedder_tokenizer, embedder_model

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_embedding_from_llm(embeds, text, llm_tokenizer, llm_model):
    indexes = list(text.keys())
    
    with torch.no_grad():
        batch_size = 4
        for i in tqdm(range(0, len(text), batch_size)):
            batch_indexes = indexes[i:i+batch_size]
            batch_dict = llm_tokenizer([text[index] for index in batch_indexes], max_length=1024, padding=True, truncation=True, return_tensors="pt").to(llm_model.device)
            outputs = llm_model(**batch_dict)
            sentence_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            # normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1).cpu()
            
            for j, index in enumerate(batch_indexes):
                embeds[index] = sentence_embeddings[j, :]
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model_dir", type=str)
    parser.add_argument("--embedding_path", type=str, default='embedding')
    parser.add_argument("--data_path", type=str, default='data')
    parser.add_argument("--random_seed", type=int, default=2025)
    parser.add_argument("--task", type=str, default='embedding')
    parser.add_argument("--domain", type=str, default='Weather')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="0" 

    fix_seed(args.random_seed)

    with open(f"{args.data_path}/{args.domain}_multi_clean.json", 'r') as fp:
        data = json.load(fp)

    embedder_tokenizer, embedder_model = initialize_embedder(args)
    res = {}
    get_embedding_from_llm(res, text = {idx: data[idx]['text'] for idx in data}, llm_tokenizer=embedder_tokenizer, llm_model=embedder_model)

    torch.save(res, f'{args.embedding_path}/{args.domain}_text_embedding.pt')