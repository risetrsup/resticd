#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install   streamlit 


# In[6]:


#!pip install fastapi 


# In[7]:


#!pip install   uvicorn 


# In[5]:


#!pip install    joblib


# In[11]:


#!pip install gdown


# In[24]:



#import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import re
import pickle 
import os
import gdown

from fastapi import FastAPI
from pydantic import BaseModel


# In[18]:


#paremeter Setup
bert_model_name = 'indobenchmark/indobert-base-p1'
label_encoder_name = 'label_encoder.pkl'
saved_model_name = 'indobert_bigru_model.pth'
model_id = '1sMDAEvH4tHDLAYCaFLBnj3BN4hzY5NQd'

model_id = '1-4f9SfmaTLFKPliRnf95HKG2apJI6NWe'
bert_model_name = ''bert-base-uncased'
label_encoder_name = 'label_encoder.pkl'
saved_model_name = 'bert_bigru_model.pth'
# In[13]:


#download Model
def download_model():
    url = 'https://drive.google.com/uc?id='+model_id
    if not os.path.exists(saved_model_name):
         gdown.download(url, saved_model_name, quiet=False)         


# In[14]:


def download_encoder():
    url = 'https://drive.google.com/uc?id=1rZWBHHybuqQBrpWQP_57k81YO2qAF3W-'
    if not os.path.exists(label_encoder_name):
         gdown.download(url, label_encoder_name, quiet=False) 


# In[15]:


#Roman Number IV A into IVA

def standardize_roman_numerals(text):
  if not isinstance(text, str):
    return text  # or you can return a default value like '' if you want to replace non-strings
  roman_pattern = re.compile(r"(?<=\b)(IV|IX|XL|XC|CD|CM|I{1,3}|V|X{1,3}|L|C{1,3}|D|M{1,3})([-\s])([A-Za-z])(?=\b)")
  return roman_pattern.sub(r"\1\3", text)

#Letter PS 1 into PS1

def standardize_ps(text):
  if not isinstance(text, str):
    return text  # or you can return a default value like '' if you want to replace non-strings
  regex_pattern = r'(PS)([\s-]*)(\d+)'
  return re.sub(regex_pattern, r'\1\3', text)

#Roman dictionary
roman_numerals = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
    'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
    'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
    'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20
  }


def roman_to_number(text) :
  return roman_numerals.get(text.upper(), text)

# Standardize "tipe" or "type" to "type"

def standardize_typetest(text):

  def replace_with_standardized(match):
    type_word = match.group(1)
    number = match.group(2).replace('-', '').replace(' ', '')
    if number.upper() in roman_numerals:
      number = roman_to_number(number.upper())
    return f'{type_word}{number}'

  text = re.sub(r'\b(tipe|type)\b[\s-]*(\b(?:I{1,3}|IV|V|VI{0,3}|IX|X{1,3}|XI{0,3}|IX|XX|[0-9]+)\b)', replace_with_standardized, text, flags=re.IGNORECASE)
  return text

def normalize_text(text):
  if not isinstance(text, str):
    return text  # or you can return a default value like '' if you want to replace non-strings

  # Standardize Covid variations to 'Covid19'
  text = re.sub(r'\bcovid\b', 'Covid', text, flags=re.IGNORECASE)
  text = re.sub(r'Covid[\s-]*19', 'Covid19', text, flags=re.IGNORECASE)
  text = re.sub(r'Covid(?!\d+)', 'Covid19', text, flags=re.IGNORECASE)

  # Replace CKR and its variants with 'cidera kepala ringan'
  text = re.sub(r'\bckr\b', 'cidera kepala ringan', text, flags=re.IGNORECASE)

  # Replace CKR and its variants with 'cidera kepala ringan'
  text = re.sub(r'\bcks\b', 'cidera kepala ringan', text, flags=re.IGNORECASE)

  # Replace stg, stage and its variants with 'stage'
  text = re.sub(r'\bstg\b|\bstage\b', 'stage', text, flags=re.IGNORECASE)

  return text


# In[ ]:


# Function to load the saved LabelEncoder
def load_label_encoder():    
    with open(label_encoder_name, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder


# In[19]:


# Function to load the model
@st.cache_resource
def load_model():    
    hidden_dim = 256
    n_layers = 2
    num_classes = 10
    dropout = 0.2
    bidirectional = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StackedGRUModel(bert_model_name, hidden_dim, n_layers, num_classes, bidirectional, dropout).to(device)
    model.load_state_dict(torch.load(saved_model_name, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model, device


# In[17]:


# Define the model class (ensure it matches your training code)
class StackedGRUModel(nn.Module):
    def __init__(self, bert_modelname, hidden_dim, n_layers, num_classes, bidirectional=True, dropout=0.2):
        super(StackedGRUModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_modelname)
        self.gru = nn.GRU(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embedded = bert_outputs.last_hidden_state
        embedded = self.dropout(embedded)
        gru_out, _ = self.gru(embedded)
        gru_out = self.dropout(gru_out)
        logits = self.fc(gru_out[:, -1, :])
        probs = self.softmax(logits)
        return probs


# In[28]:


# Load the trained model

download_model()
download_encoder()

# Load the model and device
model, device = load_model()
tokenizer = BertTokenizer.from_pretrained( bert_model_name )
label_encoder =  load_label_encoder()

#model = joblib.load( saved_model_name )

app = FastAPI()

class PredictionRequest(BaseModel):
    input_data: str

@app.post("/predict")
  
async def predict(request: PredictionRequest):
    # Process the input data as needed
    input_text = request.input_data
    
    if input_text:
        try:
            # Preprocess the text
            preprocess = input_text
            preprocess = standardize_roman_numerals(preprocess)
            preprocess = standardize_ps(preprocess)
            preprocess = normalize_text(preprocess)
            preprocess = standardize_typetest(preprocess)
            text_input = preprocess        

             # Tokenize the input
            inputs = tokenizer(text_input, return_tensors='pt', padding=True, truncation=True, max_length=128)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                predicted_class = torch.argmax(outputs, dim=1).cpu().numpy()
                predicted_label = label_encoder.inverse_transform(predicted_class)[0]

            return {"prediction": predicted_label}
            
        except Exception as e:
            return {"error": "Error saat mengklasifikasikan :" +  e } 
    else: 
        return {"error": "Silahkan masukkan text untuk prediksi klasifikasi."  }    
     


# In[ ]:




