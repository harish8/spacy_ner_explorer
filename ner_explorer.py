#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:35:43 2024

@author: harish8
"""

import pandas as pd
import spacy
import streamlit as st
import spacy_streamlit
import io

from spacy_streamlit import load_model
from spacy import displacy

import subprocess

#Fix to install the spacy models on fly https://discuss.streamlit.io/t/how-to-include-en-core-web-sm-2-2-0-in-deployment/37673/2
@st.cache_resource
def download_en_core_web_sm_md():
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])


#%%
st.title("SpaCy NER Explorer")

st.markdown('''
        Named Entity Recognition (NER) project using SpaCy. NER is a task in NLP where we identify and classify entities in text, 
        like names of people, organizations, locations, dates, and other important information. 
        This project will help you get familiar with SpaCy and its capabilities in handling text data.    
        ''')

st.markdown('''
            ## **`This dataset contains around 38000 lines of articles from CNN news from the year 2011 to 2022.`**

            Source: https://www.kaggle.com/datasets/hadasu92/cnn-articles-after-basic-cleaning/data

            ''')
            
#%%
            
spacy_model = st.sidebar.selectbox("Model name", ["en_core_web_md", "en_core_web_sm"])
nlp = load_model(spacy_model)


            
#%%
# load the articles dataset in a var `cnn_articles_df`
cnn_articles_df = pd.read_csv('./data/CNN_Articels_clean.csv').set_index('Index').reset_index(drop=True)


# print(f"Unique word count: {len(set(' '.join(cnn_articles_df['Headline']).split()))}")

#%%
st.markdown(" ## Get summary of the dataset's structure")
buffer = io.StringIO()
cnn_articles_df.info(buf=buffer)
df_cnn_articles_info = buffer.getvalue()

st.text(df_cnn_articles_info)
# cnn_articles_df.info()

#%%

cnn_articles_df.rename(columns = {
    'Date published' : 'DatePublished',
    'Second headline' : 'SecondHeadline',
    'Article text' : 'ArticleText'
        }, inplace=True)

st.markdown('''
            For this NLP project the dataset looks good with all the object types. 
            I will change the column names with spaces to CamelCase to keep it ideal to work with like the following. 
            `'Date Published' : 'DatePublished',
            'Second headline' : 'SecondHeadline',
            'Article text' : 'ArticleText'`
            ''')
            

st.markdown(" ## Get statistics of the dataset")
st.write(cnn_articles_df.describe())

#%%
st.markdown(" ## let's take a look in first few rows how the data is looking like.")
pd.set_option("display.max_columns",200)
st.dataframe(cnn_articles_df.head())

st.markdown('''
            Looks neat so far. 
            now we can start choosing some interesting features to explore and analyse the articles in the dataset.
            ''')
            
#%%
# Check for the null values in the dataset.
st.markdown("## Check for Null values")
null_values = cnn_articles_df.isna().sum()
if null_values.any():
    st.write(" Null values found in the dataset")
    st.write(null_values[null_values > 0])
else:
    st.write(" No null values are found in the dataset")
    
#%%
# Choosing the intresting features to analyse in the dataset.
st.markdown(" ## Types of articles covered in sections under each category")
group_cat_sections = cnn_articles_df.groupby(['Category']).agg({
    'Section' : lambda x : list(x.unique())
    }).reset_index()

st.markdown(" (Double click on the cell to see the full list) ")
st.dataframe(group_cat_sections, use_container_width=True)

#%%
st.markdown('''
            Pretty standard list of article sections under each category and pretty straight forward 
            that `health` and `politics` are just as their name suggests.
            \n 
            Although there are many intresting sections to dive into, for now I'll fetch the articles in
            `tech` under `business` category to explore the Named Entities. Wonder how tech is evolved since 2011 to 2022. 
            ''')


#%%
st.write('## Fetch all the articles of `tech` in `business` category')
cnn_articles_in_tech = cnn_articles_df[
    (cnn_articles_df['Category'] == 'business') &
    (cnn_articles_df['Section'] == 'tech')
    ]

st.write(f" Total number of articles in tech = {len(cnn_articles_in_tech)} ")
st.dataframe(cnn_articles_in_tech)

#%%

st.markdown('''
            Here we will use the SpaCy built-in `displaCy visualizer` highlighting the entities and their labels in a text from the choosen article. 
            ''')
        
        
st.markdown('''
            There will be two SpaCy pretrained models to choose on the left sidebar. 
            `en_core_web_sm` is a small size model
            `en_core_web_md` is a medium size model
            
            Feel free to play around choosing different models. 
            ''')

# Process the text with spaCy
# create a dropdown selectbox with a list of headlines in tech. 

option = st.selectbox("Choose an Article", cnn_articles_in_tech['Headline'])
ArticleRecord = cnn_articles_in_tech[cnn_articles_in_tech['Headline']== option]

doc = nlp(ArticleRecord.iloc[0]['ArticleText'])
ent_html = displacy.render(doc, style="ent", jupyter=False)
st.html(ent_html)

#%%

st.markdown('''
            ------
            I will conclude this project here since the main purpose of this project is to fetch the text and explore the Named Entity Recognition. 
            We will dive deeper into NLP with BERT, SpaCy and NLTK in the future projects which is already in progress. 
            ''')