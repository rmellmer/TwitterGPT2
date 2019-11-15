import markovify
import pandas as pd

import gpt_2_simple as gpt2
import io
from flask import Flask
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify

app = Flask(__name__)
api = Api(app)

class GPTGenerator(Resource):
    def get(self, context=''):
        run_name = 'run3'

        sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(sess, run_name=run_name)

        results = gpt2.generate(sess, run_name=run_name, 
                        prefix=context,
                        nsamples=1,
                        length=200,
                        batch_size=1,
                        temperature=1,
                        top_k=1,
                        include_prefix=True,
                        return_as_list=True
                    )

        all_tweets = []

        for result in results:
            subtweets = result.splitlines()
            all_tweets = list(set(all_tweets + subtweets))
            
        with io.open('tweets_unseparated.txt', 'r', encoding="utf-8") as tweet_file:
            original_tweets = tweet_file.readlines()

        original_tweets = [x.strip() for x in original_tweets] 
        
        all_tweets = list(set(all_tweets) - set(original_tweets))
        
        result = {'predicted_text': all_tweets} 
        return jsonify(result)

class MarkovifyGenerator(Resource):
    def get(self):
        with io.open('tweets_unseparated.txt', 'r', encoding="utf-8") as tweet_file:
            tweets = tweet_file.readlines()

        tweets = [x.strip() for x in tweets]

        text_model = markovify.NewlineText(tweets)

        result = {'predicted_text': text_model.make_sentence()} 
        return jsonify(result)

api.add_resource(GPTGenerator, '/GPT2/<context>', '/')
api.add_resource(MarkovifyGenerator, '/Markovify', '/')

if __name__ == '__main__':
     app.run(port='8080')