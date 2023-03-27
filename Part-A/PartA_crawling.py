import concurrent.futures
import json
import os
import re

import tweepy
from tweepy import StreamRule


class Tweet_crawling(tweepy.StreamingClient):
    def __init__(self, bearer_token, filename, chunk_size=100,file_size=512,num_workers=4):
        super().__init__(bearer_token)
        self.filename = filename
        self.chunk_size = chunk_size
        self.tweets = []
        self.file_size = file_size
        self.num_workers = num_workers

    def on_tweet(self, tweet):
        #follow the api, didn't find the hashtags, we decide to use the re.findall to extract the hashtags
        # print("get a tweet")
        try:
            # Ensure that the tweet text is properly encoded
            tweet_text = tweet.text.encode('utf-8').decode('utf-8')
        except Exception as e:
            print(f'Error encoding tweet text: {e}')
            return
        tweet_data = {
            "id": tweet.id,
            "created_at": str(tweet.created_at),
            "author_id": tweet.author_id,
            "text": tweet_text,
            "Hashtags": re.findall(r'#\w+', tweet.text),
        }
        if tweet.geo is not None:
            tweet_data['place'] = tweet.geo
        else:
            tweet_data['place'] = []
        self.tweets.append(tweet_data)

        # Check the size of the file after each chunk of tweets is received
        if len(self.tweets) % self.chunk_size == 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers)as executor:
                print("start saving")
                executor.submit(self._save_chunk)

    def _save_chunk(self):
        print(f"Saving chunk of {self.chunk_size} tweets")
        # To fix the problem of file structure,we first need to load the file and then append the new data.
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                existing_data = json.load(f)
            with open(self.filename, 'w') as f:
                existing_data.extend(self.tweets[-self.chunk_size:])
                json.dump(existing_data, f, indent=4,ensure_ascii=False)
            if os.path.getsize(self.filename) >= file_size * 1024 * 1024:
                print("reach "+file_size+" mb, stop")
                self.disconnect()
        else:
            print("create a new json file\n")
            with open(self.filename, 'w') as f:
                json.dump(self.tweets[-self.chunk_size:], f, indent=4, ensure_ascii=False)

#entery of the program
if __name__ == '__main__':
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
        bearer_token = config['bearer_token']
        filename = config['json_name']
        chunk_size = config['chunk_size']
        file_size = config['file_size_mb']
        num_workers = config['num_workers']
    # Create a TweetCrawler instance
    printer = Tweet_crawling(bearer_token, filename, chunk_size, file_size, num_workers)
    # Create a StreamRules instance
    rules = ["job", "career", "hiring", "employment", "job opportunities", "job search", "resume", "interview","linkedin.com/jobs"]
    streamrules = []
    # add new rules
    for elem in rules:
        streamrules.append(StreamRule(value=elem))
    # Create a StreamFilter instance
    printer.add_rules(streamrules)
    # Start the stream
    printer.filter(expansions="author_id", tweet_fields="created_at")
    # printer.filter(track="job", expansions="author_id", tweet_fields="created_at")