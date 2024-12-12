from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from ollama import chat, create
from pydantic import BaseModel
from typing import Literal
import pickle
import sklearn
import pandas as pd

MODEL_NAME = 'llama3.2:3b-instruct-q6_K'
app = Flask(__name__)
cors = CORS(app)
rf = "rfmodel.pickle"
products = pd.read_csv("products.csv")
unique = products['product_name'].unique()
listing = pd.read_csv("listing.csv")
factor = pd.factorize(products['product_name'])
prodss = factor[1]

pets = ['bird', 'hamster', 'guinea pig', 'rat', 'dog', 'cat']
allergies = ['almond', 'apple', 'barley', 'beef', 'broccoli', 'carrot', 'cashew',
       'cheese', 'chicken', 'corn', 'cranberry', 'dairy', 'duck', 'egg',
       'fish', 'grain', 'lamb', 'milk', 'none', 'oats', 'pea', 'peanut',
       'potato', 'pork', 'pumpkin', 'poultry', 'rice', 'salmon', 'seafood',
       'shrimp', 'soy', 'sunflower', 'tomato', 'tuna', 'turkey', 'wheat']

class Pet(BaseModel):
  pet: Literal['bird', 'hamster', 'guinea pig', 'rat', 'dog', 'cat'] | None
  allergy: Literal['almond', 'apple', 'barley', 'beef', 'broccoli', 'carrot', 'cashew',
            'cheese', 'chicken', 'corn', 'cranberry', 'dairy', 'duck', 'egg',
            'fish', 'grain', 'lamb', 'milk', 'none', 'oats', 'pea', 'peanut',
            'potato', 'pork', 'pumpkin', 'poultry', 'rice', 'salmon', 'seafood',
            'shrimp', 'soy', 'sunflower', 'tomato', 'tuna', 'turkey', 'wheat'] | None

class PetList(BaseModel):
  pets: list[Pet] | None

def generate_response(input_text):
    response = chat(
        messages=[
        {
        'role': 'system',
        'content': '''Identify the allergies of the pets mentioned by the user. The valid pets are: 'bird', 'hamster', 'guinea pig', 'rat', 'dog', 'cat'. Valid allergies are: 'almond', 'apple', 'barley', 'beef', 'broccoli', 'carrot', 'cashew',
       'cheese', 'chicken', 'corn', 'cranberry', 'dairy', 'duck', 'egg',
       'fish', 'grain', 'lamb', 'milk', 'none', 'oats', 'pea', 'peanut',
       'potato', 'pork', 'pumpkin', 'poultry', 'rice', 'salmon', 'seafood',
       'shrimp', 'soy', 'sunflower', 'tomato', 'tuna', 'turkey', 'wheat'. Ignore pets mentioned that have no allergies. Valid allergies include description that describes the valid allergies.''',
        },
        {
        'role': 'user',
        'content': input_text
        }
        ],
        model=MODEL_NAME,
        format=PetList.model_json_schema(),
    )
    print(response)
    pets = PetList.model_validate_json(response.message.content)
    return process_llm(pets)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/llm', methods=['POST'])
def llm():
    data = request.get_json()
    if 'input' not in data:
        return jsonify({'error': 'No input provided'}), 400

    input_text = data['input']
    llm_output = generate_response(input_text)

    return jsonify({'output': llm_output})

def process_llm(llm_output):
    jsonned = list()
    kv: dict
    for x in llm_output.pets:
        kv = dict()
        kv['pet'] = x.pet
        kv['allergy'] = x.allergy
        jsonned.append(kv)

    return jsonned

@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.get_json()

    if 'num' not in data:
        return jsonify({'error': 'Number must be provided'}), 400
    if data['num'] <= 0:
        return jsonify({'error': 'Number must be >0'}), 400
    
    rfm = pickle.load(open(rf, "rb"))
    toreturn = []
    prodrows = []
    ids = []

    for x in data['list']:
        print(x)
        pet = x['pet']
        allergy = x['allergy']
        print(pet, allergy)

        valid_products = products[products['pet'] == pet]
        
        while True:
            pick = valid_products.sample(n=1)
            pickid = pick.index.tolist()[0]

            if pickid in ids:
                continue

            prodname = pick['product_name'].values[0]
            prodlist = listing[listing['product_name'] == prodname]

            if prodlist.empty:
                continue

            trio = [[pickid, pets.index(pet), allergies.index(allergy)]]
            print(trio)
            print(pick, pickid, prodname)

            pre = rfm.predict(trio)
            print(pre)
            if pre[0] == 0:
                ids.append(pickid)
                #pick = pick.va
                #print(prodlist.values[0].tolist())
                columns = ['product_name', 'image_url', 'url']
                prodrows.append(prodlist[columns].iloc[0].to_dict())
            #print(prodrows)
            '''
        if pickid in ids:
            continue
        else:
            prod = pick['product_name'].tolist()[0]
            print(pickid, prod)
            prodlisting = listing[listing['product_name'] == prod]
            if prodlisting.empty:
                # paniguro lang kung hindi nagtutugma yung products at listing csv
                print("empty skipping")
                continue
            prodlisting = prodlisting.iloc[0]
            prodlistingindex = listing.index[listing['product_name'].isin(pick['product_name'])].values[0]
            #prodlistingindex = int(prod) + 12 # ewan ko bakit +12 pero may 12 row discrepancy e
            
            print(prodlisting, prodlistingindex + 12)
            
            #trio = [[prodss.tolist().index(prod), pets.index(pet), allergies.index(allergy)]]
            #print(trio)

            trio = [[0,0,0]]
            pre = rfm.predict(trio)
            print(pre)
            if pre[0] == 0:
                ids.append(0)
            '''


            if len(ids) >= data['num']:
                break
        toreturn.append(prodrows)
        ids = []
        prodrows = []

    return jsonify({'recommend': toreturn})

if __name__ == '__main__':
    app.run(debug=True)