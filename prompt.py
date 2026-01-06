import json
SYSTEM_TEXT_EXTRACTOR = {"role": "system", "content": "You are a helpful AI assistant to extract information from text." }

SYSTEM_TEXT_SUMMARIZER = {"role": "system", "content": "You are a helpful AI assistant to summarize information from context." }

#'You are a helpful AI assistant to identify information from texts.

EXP_1 = {
'Incidents':'There can be three types of entities mentioned in the texts: Accident, Victim, and Suspect. There is at most one Accident in the text. There can be multiple Victims or Suspects in the text.',
'Weather':'There is one type of entity mentioned in the texts: Weather. There can be multiple Weather in the text.'
}

EXP_2 = {
'Incidents':'There can be three kinds of entities, Accident, Victim, and Suspect mentioned in the texts, and the attribute must belong to one of the entity and relevant to the domain.',
'Weather':'There is one type of entity, Weather, mentioned in the texts, and the attribute must belong to this entity and relevant to the domain.',
}

TEMPLATE_1 = '''Your task is to analyze the given text of {domain} and extract the relevant information according to the specified JSON structure below.

Provide the extracted information in this JSON structure:
{schema_text}

Explanations about the JSON structure:
- Output only the JSON data.
- {explanation}
- For each entity, there are multiple attributes.
- For each attribute, its value is a list of exact substrings of the input text. You should consider every attribute listed in the above structure. If an attribute is not present or cannot be determined, do not include the attribute in the output.

Here are some examples:
{example_text}
 
Here is the input text:
{text}
{hint_prompt}
'''

TEMPLATE_DISCOVERY = '''Your task is to analyze the given text and discover new information mentioned in the text.

Provide the discovered information in this JSON structure:
{schema_text}

Explanations about the JSON structure:
- Output only the JSON data.
- {explanation}
- For each entity, there may be multiple attributes. 
- For each attribute, its value is a list of exact substrings of the input text. 

Here are some examples:
{example_text}

Explanations about the examples:
- Given an input text and the known information, you are encouraged to discover new information for each entity.
- The new information is qualified as a new attribute if and only if (1) it is relevant and important to the domain, \
(2) it is unique and not overlapped with the known information, and \
(3) it has proper granularity level compared with the known attributes, which means it is not too high-level or low-level concepts.  
- The new attribute must have a meaningful name and actual values from the text. 
- Output qualified new attributes as many as possible.

Here is the input text:
{text}

Here is the known information: 
{extracted_table}
'''

TEMPLATE_DEFINE = '''Your task is to summarize the context of an attribute and write a definition for it. \

Provide the definition in this JSON structure: {schema_text}

Explanations about the JSON structure:
- Output only the JSON data, where the key is the name of the attribute and the value is its definition. 
- {explanation}
- The name should be concise and in the same style with the examples. 
- The name should be of proper granularity. Note that if it is too general, it cannot accurately describe the range of values; if it is too specific, it cannot cover all the values.
- The definition should precisely describe the meaning and characteristics of the attribute, enabling the reliable extraction of the attribute from other texts.
- The definition should be less than 50 words.

{example_text}

Here is the context of the attribute to be defined: {context} 
'''

TEMPLATE_JUDGE = '''Your task is to analyze the given schema and judge whether the given concepts are semantically similar enough to be merged as one attribute. 
Here is the schema: {schema_text}

Here are the concepts to be judged:
{elements}

Explanation about task: 
- The schema includes several unique attributes, which shows the semantic granularity of attributes. \
- If the above concepts (1) are semantically equivalent, or (2) one is semantically contained by the others, or (3) they represent a finer granularity compared with the existing attributes, they need to be merged into one single attribute. Then, output "One". 
- Else, they should be at least two unique attributes, output "Two".    
- Output only "Two" or "One".
'''

TEMPLATE_HINT = '''Here are some hints:
The above text is most likely to contain the attributes: {headers}.
The above text is most likely to contain the following values: {values}.'''

EXAMPLE_1 = '''
Input:
{text}
Output:
{table}'''

EXAMPLE_DISCOVERY = '''
Input: 
Text:
{text}
Known information:
{table1}
Output:
{table2}'''


def compose_example_text(text, table, constrained=False):
    if constrained:
        # convert to prompt style
        table_json = {}
        for entity in table:
            if entity=='Accident':
                if "0" in table[entity]: table_json[entity] = table[entity]["0"]
                else: table_json[entity] = {}
            else:
                table_json[entity] = []
                for row in table[entity]:
                    table_json[entity].append(table[entity][row])
        example_table = json.dumps(table_json)
    else: example_table = json.dumps(table)
    res = EXAMPLE_1.format(text=text, table=example_table)
    return res
   
def compose_discovery_example_text(text, table, headers):
    res_known_table = {}
    res_unknown_table = {}
    for entity in table:
        known_table = {}
        unknown_table = {}

        for rid in table[entity]:
            row = table[entity][rid]
            known_row = {}
            unknown_row = {}
            for att in row:
                if att in headers: known_row[att] = row[att]
                else: unknown_row[att] = row[att]
            
            if len(known_row)>0:
                known_table[rid] = known_row
            unknown_table[rid] = unknown_row

        # the known table always starting at 0
        updated_known_table = {}
        updated_id_map = {}
        for i, rid in enumerate(list(known_table.keys())):
            updated_known_table[str(i)] = known_table[rid]
            updated_id_map[rid] = str(i)
        
        updated_unknown_table = {}
        n = len(updated_known_table)
        for i, rid in enumerate(unknown_table):
            if not rid in updated_id_map:
                updated_unknown_table[n+i] = unknown_table[rid]
            else:
                updated_unknown_table[updated_id_map[rid]] = unknown_table[rid]
                
        res_known_table[entity] = updated_known_table
        res_unknown_table[entity] = updated_unknown_table

    res = EXAMPLE_DISCOVERY.format(text=text, table1=res_known_table, table2=res_unknown_table)
    return res

CONTEXT_DEFINE = 'In the text "{t}", the {h} value are {v}. '

def compose_prompt(args, group, labeled_text, labeled_table, examples={}, aux_data=None):
    example_text = []
    index = list(group.keys())[0]
    if args.task =='extract': # group: text
        for ex in examples[index]:
            example_text += [compose_example_text(labeled_text[ex], labeled_table[ex])]

        if len(aux_data['header_preview'][index])>0:
            hint_prompt = '\n' + TEMPLATE_HINT.format(headers= ', '.join(aux_data['header_preview'][index]), values=', '.join(aux_data['value_preview'][index]))
        else:
            hint_prompt = ''
        
        content = TEMPLATE_1.format(domain=args.domain, explanation=EXP_1[args.domain], schema_text=aux_data['schema_text'], example_text='\n'.join(example_text), 
                                    text=group[index], hint_prompt=hint_prompt)
        
        res = [SYSTEM_TEXT_EXTRACTOR, {"role": "user", "content":content}]  

    elif args.task == 'discover': # discovery
        for ex in examples[index]:        
            example_text += [compose_discovery_example_text(labeled_text[ex], labeled_table[ex], aux_data['headers'][index])]
    
        content = TEMPLATE_DISCOVERY.format(explanation=EXP_1[args.domain], schema_text=aux_data['discover_schema'], example_text='\n'.join(example_text), text=group[index], extracted_table=aux_data['extracted'][index])
        res = [SYSTEM_TEXT_EXTRACTOR, {"role": "user", "content":content}]  

    elif args.task == 'define': # define 
        '''
        examples: h -> List[{'text':, 'values':}]
        '''
        context_text = []
        example_text = aux_data['definition_examples']
    
        for tv in examples[index]: context_text.append(CONTEXT_DEFINE.format(t=tv['text'], h=index, v=tv['values']))
        
        content = TEMPLATE_DEFINE.format(explanation=EXP_2[args.domain], schema_text=aux_data['definition_schema'], example_text=example_text, context=''.join(context_text))
        
        res = [SYSTEM_TEXT_SUMMARIZER, {"role": "user", "content": content}]
    elif args.task == 'judge':
        elements_text = []
        for i, header in enumerate(group[index]):
            elements_text.append(f"{i}. {examples[header]}")
        content = TEMPLATE_JUDGE.format(schema_text=aux_data['schema_text'], elements='\n'.join(elements_text))
        res = [SYSTEM_TEXT_SUMMARIZER, {"role": "user", "content": content}]
    
    return res

