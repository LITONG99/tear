# Table Extraction with Attribute Recommendation 

This is the official code and supplementary materials for our paper: **TEAR: Table Extraction with Attribute Recommendation from Texts via Large Language Models.** 

---

## :notebook: Dataset
Our dataset is a revised version, with original texts collected by [CACAPO-English](https://github.com/TallChris91/CACAPO-Dataset). Please give credits to the creators if you use the data.

The data is under `data/`. The `_clean.json` is the reviewed data; `_split.json` contains sample splits for complete schema split (level 0), and exploratory levels 1, 2, and 3; and the `_sd.json` contains the known attributes / new attributes splits for exploratory levels 1, 2, and 3. Under each exploratory level, we ensure that each ground truth test attribute appears in some of the test texts, and each ground truth validation attribute also appears in some of the validation texts, so a reasonable text-driven attribute recommendation system could discover them from text.

---

## :cloud: Dependencies
Please refer to `requirements.txt`.

---

## :snowman: Preparation

### 1. Obtain surrogate model and previews
#### :snowflake: Use our previews
The surrogate model is used for proactive preview generation for demonstration retrieval and extraction prompting. We provide the previews at `surrogate_model/{domain}_{level}_{random_seed}/{header or value}_previews_{stage}.json`.

#### :boom: Train the model and generate previews by yourself
The code to obtain the model is developed based on this [repo](https://github.com/shirley-wu/text_to_table). *Please also refer to the original repo. You may need to set up another environment.*

Here are the steps to train the model and generate the previews by yourself:
1. Download pretrained [bart.large](https://huggingface.co/facebook/bart-large).
2. Enter the working directory `surrogate_model/` and convert labeled samples to sequences (See `convert.py`). Note that our way to serialize the table is different from the original repo. This step creates data under `surrogate_data/`.
3. Configure and run `bash preprocess.sh`, which adds files to `surrogate_data/`.
4. Configure and run `bash train.sh`, which generates `{domain}_{level}_{seed}/checkpoint_average_best-3.pt`.
   - You can download some of the trained models at [tear-surrogate-model](https://huggingface.co/tlice/tear-surrogate-model/tree/main).
   - If this is supposed to be an updated model after recommendation, please set `--restore-file` to the previous checkpoint. Note that the IO to disk for storing checkpoints is the efficiency bottleneck compared to the training, so you may want to set `custom_train.py/min_train_epoch` to ignore some earlier checkpoints and accelerate the process.
5. Configure and run `bash infer.sh`, which adds files to `{domain}_{level}_{seed}/`.
6. Generate previews based on the inference (See `convert.py`).

### 2. Obtain the embedding
The embedding of texts is used for demonstration retrieval. We provide them as `embedding/_text_embedding.pt`. You can generate the embedding by yourself with [SFR-Embedding-Mistral](https://huggingface.co/Salesforce/SFR-Embedding-Mistral) and referring to `embedding.py`. The `embedding/_labeled_embedding.pt` is the embedding for attribute definition, which is only used for the Maximum Schema Similarity (MSS) baseline.

### 3. Configure the agentic LLMs
Download the [Qwen](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) or [Llama](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), and set `LLM_DIR` in `main.py` to your local directory. If you use other LLMs except for Llama and Qwen series, you will also need to set up its terminators in `main.py: get_response_from_llm`.

### 4. Configure for the dataset
If you use other datasets, you will need to provide some meta information about the tables. Specifically, in `schema.py`:
- `domain_entities`: Type of tables in the dataset.
- `schema_text`: The complete schema. If an attribute is not known under some exploratory level, our code will hide it during execution even if it is listed here.
- `discover_schema`: For AR, the tables you would like to expand columns.
- `relevance_prefix`: For AR, the Descriptive Prefix for that dataset.
- `global_attribute_definition`: For AR, the definition of attributes for the known schema. If an attribute is never known under any exploratory level, its item will not be used so you can omit it.

---

## :green_heart: Table Extraction
Configure the `--data_path`, `--work_dir`, `--embedding_path`, `--surrogate_checkpoint` and run:
```bash
python main.py --task extract --level 0
```
---

## :heart: Attribute Recommendation 
Under exploratory setting, level 1, 2 and 3.
```bash
python main.py --task extract --domain Weather --level 3
python main.py --task discover --domain Weather --level 3 --use_extracted <extracted-table>
python main.py --task integrate --domain Weather --level 3 
python main.py --task rank --domain Weather --level 3 
```