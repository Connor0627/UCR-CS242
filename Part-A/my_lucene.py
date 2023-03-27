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

#below is the personal api access information we used to retreive username with userid

# consumer_key = "your own token"
# consumer_secret = "your own token"
# access_token = "your own token"
# access_token_secret = "your own token"
# bearer_token = "your own token"
# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)
# api = tweepy.API(auth)
    
#below is the custom analyzer that we tried to implement with various filters, but eventually 
#we realized that lots of the features were already implemented in the standard analyzer
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
    
    #Below is the hashtag field
    HashtagsType = FieldType()
    HashtagsType.setStored(True)
    HashtagsType.setTokenized(True)
    HashtagsType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
    
    #Below is the userid filed, note that we comment this out because there is a limit 
    #amount of size that we could use this feature and specific reasons and limitations 
    #will be discussed in the report :)

    # UseridsType = FieldType()
    # UseridsType.setStored(True)
    # UseridsType.setTokenized(False)
    # UseridsType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
    
    #here we read the .json data file 
    with open(sys.argv[1]) as file:
        tweets = json.loads(file.read())
        
    doc_counter = 0
    time_doc_list = []
    for tweet in tweets:

        #this if statement is commented out because it was used to graph our data for analysis

        # if doc_counter > 100:
        #     break

        context = tweet['text']
        hashtags = tweet['Hashtags']
        #userid = tweet['author_id']
        doc = Document()
        doc.add(Field('Context', str(context), contextType))
        doc.add(Field('Hashtags', str(hashtags), HashtagsType))

        #below is the method that we used to retreive username using userid
        #uncomment this if userid feature is desired with a small data size

        # try:
        #     user = api.get_user(user_id = userid)
        # except Exception as e: 
        #     continue
        # doc.add(Field('UserID', str(user.screen_name), UseridsType))
    
        doc_counter+=1
        time_conter = time.time()
        run_time = time_conter - start_time
        time_doc = {'x': doc_counter , 'y': run_time}
        time_doc_list.append(time_doc)  
        writer.addDocument(doc)
        
    writer.close()
    
    time_doc_save(time_doc_list)

#this is the function that saves our runtime analysis data into .csv
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
    
#this is the function that graphs our runtime analysis
def graphmaker():
    data = pd.read_csv("time_doc.csv")
    x = data['x']
    y = data['y']
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('runtime-#tweets figure')
    plt.savefig("figure.png")
    
def retrieve(storedir, query):
    searchDir = NIOFSDirectory(Paths.get(storedir))
    searcher = IndexSearcher(DirectoryReader.open(searchDir))
    searcher.setSimilarity(ClassicSimilarity())
    #searcher.setSimilarity(BM25Similarity(k1 = 1.4, b = 0.4))
    
    #below we tried the synonym analyzer and we didn't get it to work

#     synMap={'aaaaa':'job','job':'interview'}
#     synonymAnalyzer = SynonymAnalyzer(synMap)
#     synonymAnalyzer.createComponents(fieldName = 'Hashtags')
    
    hashtagsQuery = QueryParser("Hashtags", StandardAnalyzer()).parse(query)
    hashtagsBoostQuery = BoostQuery(hashtagsQuery, 2.0)
    contextQuery = QueryParser("Context", StandardAnalyzer()).parse(query)

    #the same as the previous, uncomment this if username feature is in use

    # UserIdQuery = QueryParser("UserID", StandardAnalyzer()).parse(query)
    # UserIdBoostQuery = BoostQuery(UserIdQuery, 2.5)
    
    builder = BooleanQuery.Builder()
    builder.add(hashtagsBoostQuery, BooleanClause.Occur.SHOULD)
    # builder.add(UserIdBoostQuery, BooleanClause.Occur.SHOULD)
    builder.add(contextQuery, BooleanClause.Occur.SHOULD)
    booleanQuery = builder.build()
    
    topDocs = searcher.search(booleanQuery, 10).scoreDocs
    topkdocs = []
    for hit in topDocs:
        doc = searcher.doc(hit.doc)
        topkdocs.append({
            "score": hit.score,
            # "UserID": doc.get("UserID"),
            "hashtags": doc.get("Hashtags"),
            "text": doc.get("Context")
        })
    
    print(*topkdocs, sep = "\n")

lucene.initVM(vmargs=['-Djava.awt.headless=true'])
create_index('lucene_index/')
retrieve('lucene_index/', sys.argv[2])
graphmaker()
