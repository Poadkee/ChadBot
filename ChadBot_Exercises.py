import faiss
import requests
from pyngrok import ngrok

from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage

from sentence_transformers import SentenceTransformer

from neo4j import GraphDatabase
import json

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2') 

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = "neo4j://localhost"
AUTH = ("neo4j", "password")

url = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json"
}
llama_model = "supachai/llama-3-typhoon-v1.5"

def run_query(query, parameters=None):
   with GraphDatabase.driver(URI, auth=AUTH) as driver:
       driver.verify_connectivity()
       with driver.session() as session:
           result = session.run(query, parameters)
           return [record for record in result]
   driver.close()

cypher_query = '''
MATCH (n) WHERE (n:Exercises OR n:Greeting OR n:Grateful OR n:Goodbye) RETURN n.name as name, n.msg_reply as reply;
'''
input_corpus = []
results = run_query(cypher_query)
for record in results:
   input_corpus.append(record['name'])
input_corpus = list(set(input_corpus))
print(input_corpus)  

# Encode the input corpus into vectors using the sentence transformer model
input_vecs = model.encode(input_corpus, convert_to_numpy=True, normalize_embeddings=True)

# Initialize FAISS index
d = input_vecs.shape[1]  # Dimension of vectors
index = faiss.IndexFlatL2(d)  # L2 distance index (cosine similarity can be used with normalization)
index.add(input_vecs)  # Add vectors to FAISS index

def compute_similar_faiss(sentence):
    try:
        # Encode the query sentence
        ask_vec = model.encode([sentence], convert_to_numpy=True, normalize_embeddings=True)
        D, I = index.search(ask_vec, 1)  # Return top 1 result
        return D[0][0], I[0][0]
    except Exception as e:
        print("Error during FAISS search:", e)
        return None, None

def neo4j_search(neo_query):
   results = run_query(neo_query)
   # Print results
   for record in results:
       response_msg = record['reply']
   return response_msg     

def compute_response(sentence):
    distance, index = compute_similar_faiss(sentence)
    Match_input = input_corpus[index]
    print(f"\n==========Distance : {distance}==========")
    if distance > 0.5:
        my_msg = llama_response(sentence)
        my_msg = "ไม่พบคำถามนี้ในคลังข้อมูล คำตอบที่คุณได้รับจะมาจาก Ollama :\n" + my_msg
        # my_msg = f'ไม่พบผลลัพธ์ที่ต้องการ หรือคุณกำลังจะพูดว่า "{Match_input}"'
    else:
        My_cypher = f"MATCH (n) where (n:Exercises OR n:Greeting OR n:Grateful OR n:Goodbye) AND n.name ='{Match_input}' RETURN n.msg_reply as reply"
        my_msg = neo4j_search(My_cypher)
    # print(my_msg)
    return my_msg

def llama_response(msg):
    print("======Using Ollama=====")
    payload = {
        "model": llama_model,
        "prompt": msg + " ตอบเป็นภาษาไทย ให้คุณตอบผมในฐานะที่คุณเป็นผู้เชี่ยวชาญด้านนั้นๆที่ผมถาม ตอบสั้นไม่เกิน 30 คำ และไม่ต้องตอบแบบทวนว่าคุณเป็นผู้เชียวชาญ",
        "stream": False
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        res_JSON = json.loads(response.text)
        res_text = res_JSON["response"]
        return res_text
    else:
        print("error:", response.status_code, response.text)
        return "ขออภัย ไม่สามารถเชื่อมต่อกับเซิฟเวอร์ได้"
    
# ... Update inbound traffic via APIs to use the public-facing ngrok URL
port = "5000"
ngrok.set_auth_token("<AUTH_TOKEN>")
public_url = ngrok.connect(port).public_url
# Open a ngrok tunnel to the HTTP server
print(f"ngrok tunnel {public_url} -> http://127.0.0.1:{port}")

app = Flask(__name__)
app.config["BASE_URL"] = public_url
app.config['JSON_AS_ASCII'] = False

@app.route("/")
def home():
    return {"Hello":"World"}

@app.route("/chat", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)                    
    try: # check if request from LINE
        json_data = json.loads(body)                         
        access_token = '<ACCESS_TOKEN>'
        secret = '<SECRET>'
        line_bot_api = LineBotApi(access_token)              
        handler = WebhookHandler(secret)                     
        signature = request.headers['X-Line-Signature']      
        handler.handle(body, signature)
        msg = json_data['events'][0]['message']['text']      
        tk = json_data['events'][0]['replyToken']   

        response_msg = compute_response(msg)
        # response_msg = llama_response(msg)

        line_bot_api.reply_message(tk, TextSendMessage(text=response_msg)) 
        print(msg, tk)                      
        print("-"*20)
        print(response_msg)                
    except:
        print(body)                 
    return 'OK'                 

@app.route("/api", methods=["POST"]) # api
def api_response():
    body = request.get_data(as_text=True)                    
    json_data = json.loads(body)                         
    print("Data received:", json_data)
    response_msg = llama_response(json_data["prompt"]) 
    return response_msg 

if __name__ == '__main__':
    app.run(port=5000)