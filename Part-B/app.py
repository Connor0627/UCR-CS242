from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModel
import json
import faiss
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import lucene
from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory, NIOFSDirectory
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer, StandardTokenizer
from org.apache.lucene.queryparser.classic import QueryParser, MultiFieldQueryParser
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions, DirectoryReader, Term
from org.apache.lucene.search import IndexSearcher, BoostQuery, Query, BooleanQuery, BooleanClause, TermQuery
from org.apache.lucene.search.similarities import BM25Similarity, ClassicSimilarity
import jpype
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, isJVMStarted

app = Flask(__name__, static_url_path='/static')

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')

# Load the JSON file containing the tweets
with open('fixed_tweets2.json', 'r') as f:
    tweets = json.load(f)

@app.route('/')
def search():
    return render_template('search.html')


lucene.initVM(vmargs=['-Djava.awt.headless=true'])

# Define the route for query processing
def retrieve(query, topk):
    vm_env = lucene.getVMEnv()
    vm_env.attachCurrentThread()
    storedir = 'lucene_index/'
    searchDir = NIOFSDirectory(Paths.get(storedir))
    searcher = IndexSearcher(DirectoryReader.open(searchDir))
    searcher.setSimilarity(ClassicSimilarity())

    hashtagsQuery = QueryParser("Hashtags", StandardAnalyzer()).parse(query)
    hashtagsBoostQuery = BoostQuery(hashtagsQuery, 2.0)
    contextQuery = QueryParser("Context", StandardAnalyzer()).parse(query)
    # UserIdQuery = QueryParser("UserID", StandardAnalyzer()).parse(query)
    # UserIdBoostQuery = BoostQuery(UserIdQuery, 2.5)

    builder = BooleanQuery.Builder()
    builder.add(hashtagsBoostQuery, BooleanClause.Occur.SHOULD)
    # builder.add(UserIdBoostQuery, BooleanClause.Occur.SHOULD)
    builder.add(contextQuery, BooleanClause.Occur.SHOULD)
    booleanQuery = builder.build()

    topDocs = searcher.search(booleanQuery, topk).scoreDocs
    topkdocs = []
    for hit in topDocs:
        doc = searcher.doc(hit.doc)
        topkdocs.append({
            "score": hit.score,
            # "UserID": doc.get("UserID"),
            "hashtags": doc.get("Hashtags"),
            "text": doc.get("Context")
        })
    return topkdocs

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

def bert(query, top_k):
    index = faiss.read_index("sample_code.index")
    # Define the query
    query_embedding = convert_to_embedding(query)

    # Search for similar documents
    distances, indices = index.search(query_embedding[None, :], top_k)
    topkdocs = []
    for i, idx in enumerate(indices[0]):
        topkdocs.append(tweets[idx])
    return topkdocs

@app.route('/results', methods=['POST'])
def results():
    # Get the user's query and top_k parameter
    query = request.form['query']
    top_k = int(request.form['top_k'])
    engine = request.form['engine']
    print("engine", engine)
    if engine == 'pylucene':
        results = retrieve(query, top_k)
    else:
        results = bert(query, top_k)

    if len(results) == 0:
        return render_template('results.html', results=results, enumerate=enumerate)

    # Generate the word cloud
    text = ' '.join([tweet['text'] for tweet in results])
    wordcloud = WordCloud(
        width=800, height=400, background_color='white', stopwords=None).generate(text)

    # Save the word cloud image to a file
    wordcloud.to_file('static/wordcloud.png')

    # Render the search results
    return render_template('results.html', results=results, enumerate=enumerate)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=False)



