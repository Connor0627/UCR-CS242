import logging, sys
logging.disable(sys.maxsize)

import lucene
import pandas as pd
import matplotlib.pyplot as plt
import os
from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory, NIOFSDirectory
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer,StandardTokenizer
from org.apache.lucene.analysis.core import SimpleAnalyzer, LowerCaseFilter, StopFilter, StopAnalyzer, WhitespaceTokenizer
from org.apache.lucene.analysis import Analyzer
from org.apache.lucene.analysis.synonym import SynonymMap, SynonymGraphFilter
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.queryparser.classic import QueryParser, MultiFieldQueryParser
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions, DirectoryReader, Term
from org.apache.lucene.search import IndexSearcher, BoostQuery, Query, BooleanQuery, BooleanClause, TermQuery
from org.apache.lucene.search.similarities import BM25Similarity, ClassicSimilarity
from java.io import StringReader
from org.apache.lucene.util import CharsRef
import json
import java
import io
from java.lang import Float
import time
import tweepy

# consumer_key = "your own token"
# consumer_secret = "your own token"
# access_token = "your own token"
# access_token_secret = "your own token"
# bearer_token = "your own token"
# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)
# api = tweepy.API(auth)
    
class MyAnalyzer(StandardAnalyzer):
    # def __init__(self, synonyms: SynonymMap):
    #     self.synMap = synonyms
    def createComponents(self, fieldName):
        source = WhitespaceTokenizer()
        result = LowerCaseFilter(source)
    #   filter = SynonymGraphFilter(source, self.synMap, True)
        return StandardAnalyzer.TokenStreamComponents(source, result)
    
def create_index(dir):
    
    start_time = time.time()
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    store = SimpleFSDirectory(Paths.get(dir))
    analyzer = StandardAnalyzer()

    config = IndexWriterConfig(StandardAnalyzer())
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    config.setSimilarity(ClassicSimilarity())
    writer = IndexWriter(store, config)

    contextType = FieldType()
    contextType.setStored(True)
    contextType.setTokenized(True)
    contextType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
    
    HashtagsType = FieldType()
    HashtagsType.setStored(True)
    HashtagsType.setTokenized(True)
    HashtagsType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
    
    # UseridsType = FieldType()
    # UseridsType.setStored(True)
    # UseridsType.setTokenized(False)
    # UseridsType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
    
    with open('fixed_tweets2.json', 'r') as f:
        tweets = json.load(f)
        
    doc_counter = 0
    time_doc_list = []
    for tweet in tweets:
        context = tweet['text']
        hashtags = tweet['Hashtags']
        userid = tweet['author_id']
        doc = Document()
        doc.add(Field('Context', str(context), contextType))
        doc.add(Field('Hashtags', str(hashtags), HashtagsType))
        # try:
        #     user = api.get_user(user_id = userid)
        # except Exception as e: 
        #     continue
        # doc.add(Field('UserID', str(user.screen_name), UseridsType))
        ####
        doc_counter+=1
        time_conter = time.time()
        run_time = time_conter - start_time
        #time_doc_save(run_time,doc_counter)
        time_doc = {'x': doc_counter , 'y': run_time}
        time_doc_list.append(time_doc)  
        ####
        writer.addDocument(doc)
        
    writer.close()
    
    time_doc_save(time_doc_list)
    
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
    
    
def retrieve(query, topk):
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    storedir = 'lucene_index/'
    searchDir = NIOFSDirectory(Paths.get(storedir))
    searcher = IndexSearcher(DirectoryReader.open(searchDir))
    searcher.setSimilarity(ClassicSimilarity())
    #searcher.setSimilarity(BM25Similarity(k1 = 1.4, b = 0.4))
    
#     synMap={'aaaaa':'job','job':'interview'}
#     synonymAnalyzer = SynonymAnalyzer(synMap)
#     synonymAnalyzer.createComponents(fieldName = 'Hashtags')
    
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
    
    # (*topkdocs, sep = "\n")
    return topkdocs

lucene.initVM(vmargs=['-Djava.awt.headless=true'])
create_index('lucene_index/')
# retrieve(sys.argv[2])
# graphmaker()
