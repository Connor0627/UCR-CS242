from transformers import AutoTokenizer, AutoModel
import torch
import json
import time
import pandas as pd
import matplotlib.pyplot as plt

def time_doc_save(time_doc_list):
    try:
        df = pd.read_csv("time_doc.csv")
    except:
        df = pd.DataFrame()
        #print("\n==== numbers of document:",doc_counter , "====")
        #print("\n====The run time of the Lucene index creation process is: ", run_time, "seconds====\n")
    df = pd.DataFrame(time_doc_list)
    df.sort_values(by='x', ascending=True, inplace=True)
    df.to_csv('time_doc.csv', index=False)
    
def graphmaker():
    data = pd.read_csv("time_doc.csv")
    x = data['x']
    y = data['y']
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('runtime-#tweets figure')
    plt.savefig("figure.png")

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
with open("fixed_tweets2.json") as file:
    tweets = json.loads(file.read())

# initialize dictionary to store tokenized tweets
tokens = {'input_ids': [], 'attention_mask': []}

start_time = time.time()
doc_counter = 0
time_doc_list = []
for tweet in tweets:
    # encode each sentence and append to dictionary
    new_tokens = tokenizer.encode_plus(tweet['text'], max_length=512,
                                       truncation=True, padding='max_length',
                                       return_tensors='pt')
    tokens['input_ids'].append(new_tokens['input_ids'][0])
    tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    doc_counter+=1
    time_conter = time.time()
    run_time = time_conter - start_time
    #time_doc_save(run_time,doc_counter)
    time_doc = {'x': doc_counter , 'y': run_time}
    time_doc_list.append(time_doc)  
    ####
    
time_doc_save(time_doc_list)

# reformat list of tensors into single tensor
tokens['input_ids'] = torch.stack(tokens['input_ids'])
tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
with torch.no_grad():
    outputs = model(**tokens)
embeddings = outputs.last_hidden_state

# After we have produced our dense vectors embeddings, we need to perform a mean pooling operation to create
# a single vector encoding (the sentence embedding). To do this mean pooling operation, we will need to multiply
# each value in our embeddings tensor by its respective attention_mask value — so that we ignore non-real tokens.

# resize our attention_mask tensor:
attention_mask = tokens['attention_mask']
mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
masked_embeddings = embeddings * mask
summed = torch.sum(masked_embeddings, 1)
summed_mask = torch.clamp(mask.sum(1), min=1e-9)
mean_pooled = summed / summed_mask


# Cosine Similarity
def convert_to_embedding(query):
    tokens = {'input_ids': [], 'attention_mask': []}
    new_tokens = tokenizer.encode_plus(query, max_length=512,
                                       truncation=True, padding='max_length',
                                       return_tensors='pt')
    tokens['input_ids'].append(new_tokens['input_ids'][0])
    tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    attention_mask = tokens['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask

    return mean_pooled[0]  # assuming query is a single sentence


query = "job"
query_embedding = convert_to_embedding(query)

# FAISS
import faiss  # make faiss available

index = faiss.IndexFlatIP(768)  # build the index
print(index.is_trained)
index.add(mean_pooled)  # add vectors to the index
print(index.ntotal)

distances, indices = index.search(query_embedding[None, :], 5)
for i, idx in enumerate(indices[0]):
    print(f"Rank {i+1}: {tweets[idx]}")

faiss.write_index(index, "sample_code.index")
graphmaker()
