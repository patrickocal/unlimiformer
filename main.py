#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import BartForConditionalGeneration, AutoTokenizer
from datasets import load_dataset
import torch
from rebel import from_text_to_kb
import networkx as nx

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# example using govreport
modelname = "abertsch/unlimiformer-bart-govreport-alternating"
dataset = load_dataset("urialon/gov_report_validation")

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained(modelname)

# example_input = dataset['validation'][0]['input']
example_input = open('harry_potter.txt','r').read()
print(example_input)


kb = from_text_to_kb(example_input, verbose=True)
print("________________________________________________")
kb.print()
relations = kb.get_relations()


# In[2]:


G = nx.DiGraph()

# Add nodes and edges to the graph
for pair in relations:
    head = pair['head']
    tail = pair['tail']
    relation = pair['type']
    metadata = pair['meta']
    
    # Add nodes
    if head not in G:
        G.add_node(head)
    if tail not in G:
        G.add_node(tail)
    
    # Add edge with metadata
    G.add_edge(head, tail, type=relation, meta=metadata)


# In[3]:


import matplotlib.pyplot as plt

# Increase the size of the plot
plt.figure(figsize=(15, 15))

# Specify the layout for graph - for example, the spring layout
pos = nx.spring_layout(G, k=0.1, iterations=20)

# Draw the nodes 
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')

# Draw the edges 
nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)

# Draw node labels 
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

# Draw edge labels with a specific font size and color
edge_labels = nx.get_edge_attributes(G, 'type')  # Assuming 'type' is an attribute of edges
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')


plt.axis('off')

plt.show()

