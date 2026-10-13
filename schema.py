import json

ATTRIBUTE_DEFINITION = {
'Incidents':
{
    "Accident date": "Accident date identifies the specific time and/or date when an accident or incident occurred. The value should be a precise temporal description, including exact dates, times, or time ranges, that indicates when the event took place, which can include specific days of the week, months, or years. ",
    "Accident address": "Accident address identifies the specific location where an accident or incident occurred. The value is the most precise address or location mentioned in the text that pinpoints where the event took place, which can include street names, intersections, neighborhoods, parks, or broader geographical locations like cities or districts.",
    "Accident type": "Accident type identifies the specific noun or noun phrase used within a text to refer to an accident. The value should be the most specific term used, like \"shooting\" or \"incident\".",
    "Number of rounds fired": "Number of rounds fired identifies the quantity of ammunition discharged during an event, as stated in the text. The value should be a numerical digit or phrase (e.g., several, a few, 7 or 8) that directly indicates the number of rounds fired or a range of rounds fired, or an estimate of the number of rounds fired, such as \"about five or six\".",
    "Accident number": "number of accident identifies the ordinal or sequential designation of an event, as stated in the text. The value should be a numerical digit or word (e.g., first, second, third, pair, sequence) that directly indicates the order or ranking of the incident. ",
    "Personnel arrived time": "Personnel arrived time identifies the specific time when personnel, typically law enforcement officers, arrived at the scene of an incident or event. The value should be a time expression, such as a numerical value with a unit of measurement (e.g., \"9:00 p.m.\") or a descriptive phrase (e.g., \"early morning,\" \"late evening\") that indicates when the personnel arrived, as stated in the text.",
    "Victim number": "Victim number specifies the quantity, or the ordinal number of victims involved in an event as stated in the text. The value should be a numerical digit or word (e.g., one, two, three, first, second, third) that directly indicates the number of individuals harmed or affected by the incident.",
    "Victim status": "Victim status specifies the condition or outcome of the victims as stated in the text. The value should be the exact phrase describing the victims' health status or condition, such as \"critical condition,\" \"shot,\" \"injured,\" or \"non-life threatening injuries.",
    "Victim gender": "Victim gender identifies the sex or gender of the individual(s) involved in an event as stated in the text. The value should be a term that directly indicates the gender of the victim(s), such as \"man\", \"woman\", \"girl\", \"boy\", \"grandmother\", or \"grandfather\", or a more specific term like \"photographer\" or \"skateboard enthusiast\" if provided in the text.",
    "Victim age": "Victim age identifies the age of an individual involved in an event, typically stated in the text as a numerical value or phrase, such as \"25-year-old\" or \"22 years old\". The value should be the specific age mentioned in the text, which can be a single age or a range of ages (e.g., 17-28) or a descriptive phrase (e.g., late teens, young adults).",
    "Victim based": "Victim based identifies the location, community, or residence of the victims as stated in the text. The value should be a specific place, neighborhood, city, state, or region that directly indicates the geographic origin or connection of the victims to the incident or event. ",
    "Hospital name": "Hospital name identifies the specific name of a medical facility where victims or injured individuals were taken for treatment, as stated in the text. The value should be the exact name of the hospital mentioned in the text, which can include general hospitals, trauma centers, or other types of medical facilities.",
    "Victim name": "Victim name identifies the specific name or names of individuals harmed or affected by an event as mentioned in the text. The value is the exact name or names of the victims, which can be a single name or multiple names separated by commas or other punctuation marks.",
    "Victim race": "Victim race identifies the ethnic or racial category of the victim(s) as described in the text. The value should be a word or phrase that directly indicates the race or ethnicity of the individual(s) harmed or affected by the incident, such as \"black\", \"white\", \"Hispanic\", \"Asian\", or \"Latin\".",
    "Victim occupation": "Victim occupation identifies the profession, role, or position of the victim(s) involved in an event, as stated in the text. The value can be a specific job title, a description of their role, or their status as a student, such as \"student\" or \"freshman at a high school\".",
    "Victim vehicle": "Victim vehicle identifies the type of vehicle involved in an incident where the victims were harmed or affected. The value should be the specific type of vehicle mentioned in the text, such as \"bus\", \"car\", \"motorcycle\", or \"plane\", which was involved in the incident that resulted in harm to the victims.", 
    "Suspect gender": "Suspect gender identifies the biological sex of the individual or individuals involved in an event as perpetrators or alleged perpetrators, as stated in the text. The value should be a word (e.g., woman, man, father, brother) that directly indicates the gender of the suspect.",
    "Suspect number": "Suspect number identifies the quantity or ordinal number of individuals involved in an event as perpetrators or alleged perpetrators, as stated in the text. The value should be a numerical digit or word (e.g., one, two, three, first, second, third) that directly indicates the number of individuals suspected of committing the crime or incident.",
    "Suspect status": "Suspect status identifies the current situation or condition of an individual suspected of committing a crime or incident, as stated in the text. The value should be a descriptive phrase or verb phrase that indicates the suspect's whereabouts, actions, or circumstances, such as \"captured by a nearby surveillance camera\", \"fled the area\", or \"in custody\" ",
    "Suspect age": "Suspect age identifies the age of the individual or individuals involved in an event as a perpetrator or alleged perpetrator, as stated in the text. The value should be a numerical digit or phrase (e.g., \"21-year-old\", \"24\", \"55-year-old\") that directly indicates the age of the suspect or a descriptive term, such as \"teen\", \"adult\", or \"juvenile\", that indicates the age range or category of the suspect(s).",
    "Suspect name": "Suspect name identifies the specific name or names of the individuals involved in an event as perpetrators or alleged perpetrators, as stated in the text. The value should be the exact name or names mentioned in the text, which can include first and last names, nicknames, or initials, and may be accompanied by additional information such as age, address, or other relevant details.",
    "Suspect description": "Suspect description identifies the weight, height, clothing, accessories, or other distinguishing physical features of an individual suspected of being involved in an event, as mentioned in the text. The value is a descriptive phrase or sentence that provides a detailed description of the suspect's appearance or attire. ",
    "Suspect weapon": "Suspect weapon identifies the type of weapon or weapons used by the suspect or alleged perpetrator in an event, as stated in the text. The value can be a specific type of firearm, a general description (e.g., handgun, semi-automatic), or a quantity (e.g., multiple guns).",
    "Suspect vehicle": "Suspect vehicle identifies the type and description of the vehicle driven or used by a suspect or perpetrator in an event, as mentioned in the text. The value should be a specific make, model, color, or other distinctive feature of the vehicle that is used to identify the suspect's mode of transportation during the incident.",
    "Suspect occupation": "Suspect occupation identifies the profession, occupation, or category of individuals involved in an event as alleged perpetrators, as stated in the text. The value should be a specific noun or phrase that describes the suspect's occupation, such as \"business owner,\" \"employee,\" \"juvenile,\" or a specific job title, which helps to provide context and understanding of the individuals' involvement in the incident.",
    "Suspect based": "Suspect based identifies the location, group, or affiliation from which the suspect originates, as stated in the text. The value should be a specific noun or phrase indicating the origin or association of the suspect, such as a city, town, group, or organization.",
    "Suspect race": "Suspect race identifies the racial or ethnic identity of the suspect or suspects mentioned in the text. The value should be a specific racial or ethnic group, such as \"African American,\" \"Hispanic,\" \"Asian,\" \"Caucasian,\" or \"Native American,\" that describes the suspect's race or ethnicity.",
    "Prison name": "Prison name identifies the specific name of the correctional facility where a suspect or offender is being held or has been held, as stated in the text. The value should be the exact name of the prison, jail, or detention center mentioned in the text. "   
},
'Weather':{
    'Weather type': "Weather type identifies the general state of the atmosphere, describing primary conditions like \"sunny,\" \"snow,\" or \"storm.\"",
    'Time': "Time identifies the specific period, date, or moment when a weather condition is expected to occur, including both absolute times and relative references.",
    'Location': "Location identifies the geographical area, such as a city, county, region, or a relative direction, where the reported weather condition is occurring.",
    'Weather area': 'Weather area identifies the spatial distribution or coverage of a weather condition within a broader location, such as "localized," "isolated," or "widespread," describing how the weather is spread out.',
    'Sunrise time': 'Sunrise time identifies the specific time of day when the sun rises above the horizon, marking the beginning of daylight.',
    'Sunset time': 'Sunset time identifies the specific time of day when the sun disappears below the horizon, marking the end of daylight.',
    'Weather occurring chance': 'Weather occurring chance identifies the probability or likelihood that a specific weather event will happen, often expressed as a percentage or descriptive term.',
    'Weather frequency': 'Weather frequency identifies how often a weather condition occurs or repeats within a given time period, using descriptive terms like "occasional," "intermittent," or "continuous."',
    'Weather compass direction': 'Weather compass direction identifies the cardinal or intercardinal direction (e.g., northeast) from which a weather system is approaching or moving towards.', 
    'Minimum temperature': "Minimum temperature identifies the lowest expected temperature value over a specified period, often given with numerical values and units.",
    'Maximum temperature': "Maximum temperature identifies the highest expected temperature value over a specified period, often given with numerical values and units.",
    'Temperature': 'Temperature identifies the value, range, or trend of the general atmospheric warmth or cold, such as "18C," "increasing," or "cold."',    
    'Wind status': "Wind status identifies descriptive details about wind conditions or its general state, such as \"calm\", \"breezy\", or \"moderate\", without specific speed or direction.",
    'Wind direction': "Wind direction identifies the compass direction from which the wind is blowing, such as \"northerly\" or \"north.\"",
    'Wind speed': "Wind speed identifies the quantitative or descriptive measure of how fast the wind is blowing, including numerical values and units.",
    'Snow status':'Snow status identifies descriptive details about snow, including its amount, expected accumulation, change and other detail status.',
    'Rain status': 'Rain status identifies descriptive details about rain, including its intensity, amount, expected change, or other characteristics like "heavy" or "easing."',
    'Cloud status': 'Cloud status identifies descriptive details about cloud cover, including its amount, density, or expected change, such as "mostly cloudy" or "begin to clear."',
    'Cloud type': 'Cloud type identifies the specific classification or form of the clouds present, such as "cumulus" or "high-level clouds."',
},
"Conversation": {
    "Attraction name": "Attraction name identifies the specific name of a tourist site or point of interest (e.g., 'scudamores punting co') as mentioned in the dialogue.",
    "Attraction area": "Attraction area identifies the geographical region or neighborhood of the attraction (e.g., 'centre', 'east'), as specified by the user.",
    "Attraction type": "Attraction type identifies the category or kind of attraction (e.g., 'museum', 'park', 'boat'), as mentioned by the user.",

    "Hotel name": "Hotel name identifies the specific name of the hotel (e.g., 'warkworth house') as explicitly mentioned or confirmed by the user in the dialogue.",
    "Hotel area": "Hotel area identifies the desired geographical region for the hotel (e.g., 'centre', 'east'), as specified by the user.",
    "Hotel type": "Hotel type identifies the accommodation category (e.g., 'hotel', 'guest house') as requested by the user.",
    "Hotel parking": "Hotel parking indicates whether the hotel offers parking facilities, with value 'yes' or 'no', as specified by the user.",
    "Hotel price range": "Hotel price range identifies the price category (e.g., 'cheap', 'moderate', 'expensive') preferred by the user.",
    "Hotel internet": "Hotel internet indicates whether the hotel provides free Wi-Fi, with value 'yes' or 'no', as specified by the user.",
    "Hotel stars": "Hotel stars identifies the star rating of the hotel (e.g., '4', '5') as requested by the user.",
    "Hotel requested day": "Hotel requested day identifies the day of the week (e.g., 'tuesday') the user wants to check in, as the final stated request.",
    "Hotel requested people": "Hotel requested people identifies the number of guests (e.g., '6') the user finally requests for the hotel booking.",
    "Hotel requested stay": "Hotel requested stay identifies the number of nights (e.g., '2') the user finally requests to stay at the hotel.",
    "Hotel confirmed name": "Hotel confirmed name identifies the hotel name (e.g., 'the cambridge belfry') returned by the system upon successful booking confirmation.",
    "Hotel confirmed reference": "Hotel confirmed reference identifies the alphanumeric booking reference number (e.g., '7GAWK763') provided by the system upon successful hotel booking.",

    "Restaurant name": "Restaurant name identifies the specific name of the restaurant (e.g., 'river bar steakhouse and grill') as mentioned or confirmed by the user.",
    "Restaurant area": "Restaurant area identifies the desired neighbourhood (e.g., 'centre', 'east') for the restaurant, as specified by the user.",
    "Restaurant food": "Restaurant food identifies the type of cuisine (e.g., 'modern european', 'Italian') requested by the user; may contain multiple values.",
    "Restaurant price range": "Restaurant price range identifies the price category (e.g., 'cheap', 'moderate') preferred by the user.",
    "Restaurant requested day": "Restaurant requested day identifies the day of the week (e.g., 'thursday') the user finally requests for the reservation.",
    "Restaurant requested people": "Restaurant requested people identifies the number of people (e.g., '3') the user finally requests for the restaurant booking.",
    "Restaurant requested time": "Restaurant requested time identifies the reservation time (e.g., '13:00') in 24-hour format, as the user's final requested time.",
    "Restaurant confirmed name": "Restaurant confirmed name identifies the restaurant name returned by the system upon successful booking confirmation.",
    "Restaurant confirmed reference": "Restaurant confirmed reference identifies the alphanumeric booking reference number (e.g., 'A36LRSL9') provided by the system upon successful restaurant booking.",

    "Taxi departure": "Taxi departure identifies the pick-up location for the taxi (e.g., 'restaurant 2 two') as specified by the user.",
    "Taxi destination": "Taxi destination identifies the drop-off location for the taxi (e.g., 'museum of classical archaeology') as specified by the user.",
    "Taxi leave at": "Taxi leave at identifies the desired departure time for the taxi (e.g., '10:00') in 24-hour format, as requested by the user.",
    "Taxi arrive by": "Taxi arrive by identifies the latest arrival time for the taxi (e.g., '19:45') in 24-hour format, as specified by the user.",

    "Train departure": "Train departure identifies the departure station for the train journey (e.g., 'cambridge') as specified by the user.",
    "Train destination": "Train destination identifies the final station for the train journey (e.g., 'bishops stortford') as specified by the user.",
    "Train day": "Train day identifies the day of the week (e.g., 'friday') for the train trip, as specified by the user.",
    "Train leave at": "Train leave at identifies the departure time for the train (e.g., '17:29') in 24-hour format, as requested by the user.",
    "Train arrive by": "Train arrive by identifies the latest arrival time for the train (e.g., '19:45') in 24-hour format, as specified by the user.",
    "Train requested people": "Train requested people identifies the number of passengers (e.g., '4') the user finally requests for the train booking.",
    "Train confirmed reference": "Train confirmed reference identifies the alphanumeric booking reference number (e.g., 'UIFV8FAS') provided by the system upon successful train booking."
}
}


relevance_prefix = {
    'Incidents': '''Here is a schema designed to document and manage detailed information and facts about criminal incidents or accidents, particularly those involving violence, such as shootings or other crimes. The attributes are organized into three main entities: the accident itself, the victims, and the suspects. For each entity, there are multiple attributes. The attributes are unique and in consistent styles. Here are the attributes in this schema: "''',    
    'Weather': '''Here is a schema designed to document and manage detailed information and facts about weather conditions from news reports, particularly those about location, time, temperature, wind, cloud, rain, and snow. The attributes are organized into one main entity: the Weather itself, and there are multiple attributes. The attributes are unique and in consistent styles. Here are the attributes in this schema: "''',
    'Conversation': '''Here is a schema designed to document and manage detailed information from multi-turn task-oriented dialogues, covering five service domains: Attraction, Hotel, Restaurant, Taxi, and Train. \
        For each domain, there are multiple attributes recording both descriptive facts and final booking details (user's final requests and system's confirmed reservations). The attributes are unique and in consistent styles. Here are the attributes in this schema: "'''
    }
       

schema_text = {
'Weather': '''    
{
    "Weather": {
        "0":{
            "Temperature": [],
            "Snow status": [],
            "Cloud type": [],
            "Sunset time": [],
            "Rain status": [],
            "Weather area": [],
            "Time": [],
            "Weather type": [],
            "Wind speed": [],
            "Weather compass direction": [],
            "Maximum temperature": [],
            "Weather frequency": [],
            "Wind status": [],
            "Minimum temperature": [],
            "Sunrise time": [],
            "Weather occurring chance": [],
            "Wind direction": [],
            "Location": [],
            "Cloud status": []
        },
        "1":{...},
        ...
    }
}
''',

'Incidents':'''
{
    "Accident": {
        "0":{
            "Accident type":  [],
            "Accident date":  [],
            "Accident address":  [],
            "Number of rounds fired":  [],
            "Accident number":  [],
            "Personnel arrived time":  [],
        }
    },
    "Victim": {
        "0": {
            "Victim number": [],
            "Victim status": [],
            "Victim gender": [],
            "Victim age": [],
            "Victim based": [],
            "Hospital name": [],
            "Victim name": [],
            "Victim race": [],
            "Victim occupation": [],
            "Victim vehicle": [],
        },
        "1":{...},
        ...
    },
    "Suspect": {
        "0":{
            "Suspect gender": [],
            "Suspect number": [],
            "Suspect status": [],
            "Suspect age": [],
            "Suspect name": [],
            "Suspect description": [],
            "Suspect weapon": [],
            "Suspect vehicle": [],
            "Suspect occupation": [],
            "Suspect based": [],
            "Suspect race": [],
            "Prison name": [],
        },
        "1":{...},
        ...
    }
}''',
'Conversation':'''
{
    "Attraction": {
        "Attraction name": "...",
        "Attraction area": "...",
        "Attraction type": "..."
    },
    "Hotel": {
        "Hotel name": "...",
        "Hotel area": "...",
        "Hotel type": "...",
        "Hotel parking": "...",
        "Hotel price range": "...",
        "Hotel internet": "...",
        "Hotel stars": "...",
        "Hotel requested day": "...",
        "Hotel requested people": "...",
        "Hotel requested stay": "...",
        "Hotel confirmed name": "...",
        "Hotel confirmed reference": "..."
    },
    "Restaurant": {
        "Restaurant name": "...",
        "Restaurant area": "...",
        "Restaurant food": "...",
        "Restaurant price range": "...",
        "Restaurant requested day": "...",
        "Restaurant requested people": "...",
        "Restaurant requested time": "...",
        "Restaurant confirmed name": "...",
        "Restaurant confirmed reference": "..."
    },
    "Taxi": {
        "Taxi departure": "...",
        "Taxi destination": "...",
        "Taxi leave at": "...",
        "Taxi arrive by": "..."
    },
    "Train": {
        "Train departure": "...",
        "Train destination": "...",
        "Train day": "...",
        "Train leave at": "...",
        "Train arrive by": "...",
        "Train requested people": "...",
        "Train confirmed reference": "..."
    }
}'''
}

domain_entities = {
    'Weather': ['Weather'],
    'Incidents':['Accident', 'Victim', 'Suspect'],
    'Conversation':['Attraction','Hotel','Restaurant','Taxi','Train']
}

discover_schema = {
'Weather':
'''
{
    "Weather": {
        "0": {
            "<new attribute 1>": [],
            "<new attribute 2>": []
        },
        "1":{...},
        ...
    },
}
''',
'Incidents':
'''
{
    "Accident": {
        "0":{
            "<new attribute 1>": [],
            "<new attribute 2>": []
        }
    },
    "Victim": {
        "0": {
            "<new attribute 1>": [],
            "<new attribute 2>": []
        },
        "1":{...},
        ...
    },
    "Suspect": {
        "0":{
            "<new attribute 1>": [],
            "<new attribute 2>": []
        },
        "1":{...},
        ...
    }
}''',
'Conversation':'''
{
    "Attraction": {
        "<new attribute 1>": "...",
        "<new attribute 2>": "..."
    },
    "Hotel": {
        "<new attribute 1>": "...",
        "<new attribute 2>": "..."
    },
    "Restaurant": {
        "<new attribute 1>": "...",
        "<new attribute 2>": "..."
    },
    "Taxi": {
        "<new attribute 1>": "...",
        "<new attribute 2>": "..."
    },
    "Train": {
        "<new attribute 1>": "...",
        "<new attribute 2>": "..."
    }
}'''
}

definition_schema = '''
{
    "<name>": "<definition>",
}'''

attribute_schema = '''
{
    "<name of new attribute>":"<definition of new attribute>",
}
'''


def process_entity_schema_text(domain, known_headers, return_json=True):
    res = {h: ATTRIBUTE_DEFINITION[domain][h] for h in known_headers}
    if return_json:
        return json.dumps(res)
    else: return res

def process_schema_text(domain, known_headers):
    known_headers = set(known_headers) 

    res = []
   
    lines = schema_text[domain].split('\n')

    if domain =='Conversation':
        for line in lines:
            if '...' in line:
                cur_h = line[line.find('"')+1:line.rfind('":')]
                if cur_h in known_headers: res.append(line)
            else: res.append(line)
    else:
        lines = schema_text[domain].split('\n')
        for line in lines:
            if '[]' in line:
                cur_h = line[line.find('"')+1:line.rfind('"')]
                if cur_h in known_headers: res.append(line)
            else: res.append(line)
    res = '\n'.join(res)
    return res
    

EXAMPLE_DEFINE = '''Input:
{context}
Output:
{definition}
'''

CONTEXT_DEFINE = 'In the text"{t}", the {h} value is "{v}". '

def process_definition_examples(domain, known_headers, context):
    '''
    context_index: h to indices
    '''
    res = []

    for i, h in enumerate(known_headers):
        context_text = []
        for tv in context[h]: context_text.append(CONTEXT_DEFINE.format(t=tv['text'], h=h, v=tv['values']))
        tmp = EXAMPLE_DEFINE.format(context=''.join(context_text), definition = json.dumps({h:ATTRIBUTE_DEFINITION[domain][h]}))
        res.append(tmp)

    res = '\n'.join(res)
    return res