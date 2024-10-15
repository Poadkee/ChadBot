import faiss
import requests
from pyngrok import ngrok

from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage, QuickReply, QuickReplyButton, MessageAction

from sentence_transformers import SentenceTransformer

from neo4j import GraphDatabase
import json

from bs4 import BeautifulSoup
from selenium import webdriver

#model = SentenceTransformer('bert-base-nli-mean-tokens')
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2') 
# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = "neo4j://localhost"
AUTH = ("neo4j", "password")

url = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json"
}
llama_model = "supachai/llama-3-typhoon-v1.5"


pz_url = "https://www.1112.com/th/"

# setup chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless') # ensure GUI is off
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

def run_query(query, parameters=None):
   with GraphDatabase.driver(URI, auth=AUTH) as driver:
       driver.verify_connectivity()
       with driver.session() as session:
           result = session.run(query, parameters)
           return [record for record in result]
   driver.close()

def create_query(query, parameters=None):
   with GraphDatabase.driver(URI, auth=AUTH) as driver:
       driver.verify_connectivity()
       with driver.session() as session:
           result = session.run(query, parameters)
           return result
   driver.close()

cypher_query = '''
MATCH (n) WHERE (n:Greeting OR n:Grateful OR n:Goodbye OR n:pizza OR n:quick_reply) RETURN n.name as name, n.msg_reply as reply;
'''
input_corpus = []
results = run_query(cypher_query)
for record in results:
   print(record)
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

# /============ QUICK REPLY ==============
quick_reply_start = QuickReply(items=[
    QuickReplyButton(action=MessageAction(label="แนะนำโปร pizza", text="แนะนำโปร pizza ให้หน่อย")),
    QuickReplyButton(action=MessageAction(label="กำหนดช่วงราคาที่ต้องการ", text="ขอกำหนดช่วงราคาที่ต้องการ"))
])

quick_reply_price = QuickReply(items=[
    QuickReplyButton(action=MessageAction(label="ไม่กำหนดราคา", text="ไม่กำหนดราคา")),
])

quick_reply_keep_looking = QuickReply(items=[
    QuickReplyButton(action=MessageAction(label="ต้องการ", text="เลือกดูเพิ่มเติม")),
    QuickReplyButton(action=MessageAction(label="ไม่ต้องการ", text="ไม่ต้องการ"))
])

quick_reply_menu = QuickReply(items=[
    QuickReplyButton(action=MessageAction(label="พิซซ่าไบท์", text="พิซซ่าไบท์")),
    QuickReplyButton(action=MessageAction(label="พิซซ่าสุดคุ้ม", text="พิซซ่าสุดคุ้ม")),
    QuickReplyButton(action=MessageAction(label="ชุดอิ่มเดี่ยว", text="ชุดอิ่มเดี่ยว")),
    QuickReplyButton(action=MessageAction(label="ชุดอิ่มกลุ่ม", text="ชุดอิ่มกลุ่ม")),

    QuickReplyButton(action=MessageAction(label="ไก่", text="ไก่")),
    QuickReplyButton(action=MessageAction(label="พาสต้า", text="พาสต้า")),
    QuickReplyButton(action=MessageAction(label="อาหารทานเล่น", text="อาหารทานเล่น")),
    QuickReplyButton(action=MessageAction(label="สลัด", text="สลัด")),
    QuickReplyButton(action=MessageAction(label="สเต็ก", text="สเต็ก")),
    QuickReplyButton(action=MessageAction(label="เครื่องดื่ม", text="เครื่องดื่ม")),
    QuickReplyButton(action=MessageAction(label="ของหวาน", text="ของหวาน")),
])
# ============ QUICK REPLY ==============/

def reset_history_chat(userid):
    create_query(f'MATCH (n:Price) WHERE n.userId="{userid}" SET n.min_price = 0, n.max_price = 9999')
    create_query(f'MATCH (n:UserId)-[r:Conversation]->(chat:ChatHistory) WHERE n.userId = "{userid}" DETACH DELETE chat')
    default_history = run_query(f"MATCH (n:DefaultChatHistory) RETURN n.msg_send as msg_send, n.msg_reply as msg_reply")

    for i in range(len(default_history)):
        # print(default_history[i]['msg_send'])
        create_query(f"MATCH (n:UserId) WHERE n.userId='{userid}' CREATE (n)-[:Conversation]->(:ChatHistory {{msg_send: '{default_history[i]['msg_send']}', msg_reply: '{default_history[i]['msg_reply']}'}})")

default_prompt = ' ตอบเป็นภาษาไทย ไม่เกิน 20 คำ ถ้าหากถามนอกเหนือจากนั้นให้ถือว่าคุณไม่รู้เรื่องนั้นและตอบว่าไม่ทราบ'
def compute_response(sentence, userid):
    quick_reply = None

    # check if any userID in history
    result = run_query(f"MATCH (n:UserId) WHERE n.userId='{userid}' RETURN n;")
    if(result == []): # if no userID in history chat => create one!
        create_query(f"CREATE (user:UserId {{userId:'{userid}'}})")

    print("\n=====got history=====")
    history = run_query(f"MATCH (u:UserId {{userId:'{userid}'}})-[r:Conversation]->(c:ChatHistory) RETURN c.msg_send as msg_send, c.msg_reply as msg_reply;")
    chat_history = "".join([f"User: {history[i]['msg_send']}\nChad Bot: {history[i]['msg_reply']}\n" for i in range(0, len(history))])
    prompt_history = chat_history + f"User: {sentence}"
    print(prompt_history)
        
    distance, index = compute_similar_faiss(sentence)
    Match_input = input_corpus[index]
    print(f"\n==========Distance : {distance}==========" + Match_input)
    try: # check if price range
        remove_space = sentence.replace(" ", "")
        price = remove_space.split("-")
        for i in range(len(price)):
            price[i] = int(price[i])
        sorted_p = sorted(price)

        min_price = int(sorted_p[0])
        max_price = int(sorted_p[1])
        print(min_price, max_price)

        price = run_query(f"MATCH (n:Price) WHERE n.userId='{userid}' RETURN n.min_price as min_price, n.max_price as max_price;")
        if(price == []):
            create_query(f'CREATE (:Price {{userId:"{userid}", min_price:"{min_price}", max_price:"{max_price}"}})')
        else: # update price
            create_query(f'MATCH (n:Price) WHERE n.userId="{userid}" SET n.min_price = {min_price}, n.max_price = {max_price}')

        my_msg = "กรุณาเลือกหมวดอาหารที่คุณสนใจครับ"
        quick_reply = quick_reply_menu
        print("end of price range")
    except:
        print("except")
        if distance > 0.2:
            my_msg = llama_response(prompt_history)
            # my_msg = "ไม่พบคำถามนี้ในคลังข้อมูลครับ"
            quick_reply = quick_reply_start
        else:
            My_cypher = f"MATCH (n) where (n:Greeting OR n:Grateful OR n:Goodbye OR n:pizza OR n:quick_reply) AND n.name ='{Match_input}' RETURN n.msg_reply as reply"
            my_msg = neo4j_search(My_cypher)
            if(my_msg[:24] == pz_url): # if response is URL
                price = run_query(f"MATCH (n:Price) WHERE n.userId='{userid}' RETURN n.min_price as min_price, n.max_price as max_price;")
                min_price = price[0]['min_price']
                max_price = price[0]['max_price']
                print(min_price, max_price)
                my_msg = web_scrape(my_msg, min_price, max_price, userid)
                if my_msg == "ขออภัย ไม่พบตัวเลือกที่คุณต้องการค้นหา":
                    quick_reply = quick_reply_keep_looking
                else:
                    quick_reply = quick_reply_keep_looking
                my_msg = my_msg + "\n\nต้องการเลือกดูอย่างอื่นเพิ่มเติมอีกหรือไม่"

            else:
                if my_msg == "แนะนำ":
                    my_msg = "กรุณาเลือกหมวดอาหารที่คุณสนใจครับ"
                    quick_reply = quick_reply_menu
                elif my_msg == "กำหนดราคา":
                    my_msg = 'กรุณากรอกช่วงราคาที่ต้องการครับ เช่น 0-500 หรือ 200-699 หากไม่ต้องการให้กรอกคำว่า "ไม่กำหนดราคา" ครับ'
                    quick_reply = quick_reply_price
                elif my_msg == "ไม่กำหนดราคา":
                    create_query(f'MATCH (n:Price) WHERE n.userId="{userid}" SET n.min_price = 0, n.max_price = 9999')
                    my_msg = "กรุณาเลือกหมวดอาหารที่คุณสนใจครับ"
                    quick_reply = quick_reply_menu
                elif my_msg == "ลบประวัติการค้นหา":
                    reset_history_chat(userid)
                    my_msg = "ลบประวัติการค้นหาสำเร็จ"
                    return my_msg, quick_reply_start
                elif my_msg == "เลือกดูเพิ่มเติม":
                    my_msg = "กรุณาเลือกหมวดอาหารที่คุณสนใจครับ"
                    quick_reply = quick_reply_menu
                elif my_msg == "ไม่ต้องการ":
                    my_msg = "ยินดีที่ได้ช่วยเหลือครับ หากต้องการอะไรเพิ่มเติมสามารถถามได้เลยครับ"
                    quick_reply = quick_reply_start
                else:
                    quick_reply = quick_reply_start
    # print(my_msg)

    # create chat history
    create_query(f"MATCH (n:UserId) WHERE n.userId='{userid}' CREATE (n)-[:Conversation]->(:ChatHistory {{msg_send: '{sentence}', msg_reply: '{my_msg}'}})")
    
    return my_msg, quick_reply

food_type1 = ["chicken", "pasta", "appetizers", "salad", "steak", "drink", "desserts"]

def web_scrape(url, price_min, price_max, userid):
    sub_url = url[24:]
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    driver.implicitly_wait(5)  # Wait for the page to load
    html = driver.page_source   
    mysoup = BeautifulSoup(html, "html.parser")

    msg_reply = "Menu : \n"
    if(sub_url in food_type1):
        job_elements = mysoup.find_all("div",{"class":"relative self-stretch"})
    else:
        job_elements = mysoup.find_all("div",{"class":"w-full py-3 lg:py-6 px-4 lg:px-6 flex flex-col gap-y-2"})
    
    json_results = []
    isDescription = True
    for job_element in job_elements:
        price = job_element.find("span", class_="inline-block")
        price_int = int(price.text[1:])
        if price_min > price_int or price_int > price_max: # if price in range
            continue

        title = job_element.find("h2", class_="card-name")
        description = job_element.find("div", class_="text-body_sm_regular sm:text-body_lg_regular text-text_secondary")

        if description:
            json_results.append({'title':title.text, 'price':price_int, 'description':description.text})
            isDescription = True
        else:
            json_results.append({'title':title.text, 'price':price_int, 'description':None})
            isDescription = False

    if (json_results == []):
        msg_reply = "ขออภัย ไม่พบตัวเลือกที่คุณต้องการค้นหา"
        create_query(f'MATCH (n:Price) WHERE n.userId="{userid}" SET n.min_price = 0, n.max_price = 9999')
        return msg_reply
    else:  
        # sort data from price
        sorted_json = sorted(json_results, key=lambda x: x['price'])
        for json_result in sorted_json:
            msg_reply = msg_reply + f"{json_result['title']}\nราคา : ฿{json_result['price']}\n"
            if isDescription:
                msg_reply = msg_reply + f"รายละเอียด : {json_result['description']}\n"
            msg_reply = msg_reply + "\n"

        return msg_reply[:-2]

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
        userid = json_data['events'][0]['source']['userId']

        response_msg, quick_reply = compute_response(msg, userid)

        line_bot_api.reply_message(tk, TextSendMessage(text=response_msg, quick_reply=quick_reply))
        print(msg, userid)                      
        print("-"*20)
        print(response_msg)                
    except Exception as e:
        print("Error Line Bot : ", e)     
        # print(body)    
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