#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:51:54 2024

@author: harish8
"""
import spacy
import streamlit as st
import spacy_streamlit

from spacy_streamlit import load_model

spacy_model = st.sidebar.selectbox("Model name", ["en_core_web_sm"])
nlp = load_model(spacy_model)

# nlp = spacy.load("en_core_web_sm")

# sample text1
text1 = "Apple is looking at buying U.K. startup for $1 billion"

# sample text2
text2 = "SpaceX launches Falcon 9 rocket carrying Startlink satellites"

# sample text3
text3 = "Introducing the new iPhone 13 with advanced camera technology"

# sample text4
text4 = "Just booked tickets for a vacation in Paris with my friends!"

# sample text5
text5 = "This agreement is made between Company X and Company Y for the sale of goods."


# Process the text with the loaded model. 
def get_named_entities(text):
    
    # load the text in doc variable. 
    doc = nlp(text)
    # Print the named entities
    for ent in doc.ents:
        print (ent.text, ent.label_)
    
    
# output the named entities
get_named_entities(text2)
