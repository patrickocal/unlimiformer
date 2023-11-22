from transformers import BartForConditionalGeneration, AutoTokenizer
from datasets import load_dataset
import torch
from rebel import from_text_to_kb
import networkx as nx

def generate_triples(long_text_data):
    kb = from_text_to_kb(long_text_data)
    
    triples = kb.get_triples()
    
    return triples



# example using govreport
dataset = load_dataset("urialon/gov_report_validation")
example_input = dataset['validation'][0]['input']
triples = generate_triples(example_input)
print(triples)
