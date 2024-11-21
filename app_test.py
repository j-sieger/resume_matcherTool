from flask import Flask
from flask import request
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L12-v2')

app = Flask(__name__)


def resume_fetching(job_description,job_role):
    try:
            # Path to Obsidian vault
            csv_file = "resume.csv"

            data=pd.read_csv(csv_file)
            data["similarity"]=data.apply(lambda x:fuzz.partial_ratio(job_role,x['Category']),axis=1)
            data.sort_values(by=['similarity'],inplace=True,ascending=False)
            data['Full_Resume']="Category : "+data['Category'] +"\n " + "Experience + Skills:" +"\n " + data['Resume']
            jd_embedding=model.encode(job_description)
            top_data=data.head(10)
            top_data['sims_transformer']=top_data.apply(lambda x : cosine_similarity(model.encode(x['Full_Resume']).reshape(1,-1),jd_embedding.reshape(1,-1)),axis=1)
            top_data.sort_values(by="sims_transformer",inplace=True,ascending=False)
            return top_data.head(5)['Email'].values.tolist()
            # # return data.head(3).to_dict(orient="records")

            # # return the filename
            # return data.head(3).to_dict(orient="records")
    except Exception as e:
            print("tttt",job_description)
            print(str(e))
            return "Error with the input for the tool."

@app.route('/', methods = ['GET', 'POST', 'DELETE'])
def homepage():
          return "This is a test page"

@app.route('/get_resumes', methods = ['GET', 'POST', 'DELETE'])
def user():
    if request.method == 'POST':
          content=request.json
          print("------",content['job_description'])
          print("---------- jr",content["job_role"])
          return resume_fetching(content['job_description'],content["job_role"])

if __name__ == '__main__':

    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(host="0.0.0.0",port="8004",debug=True)
