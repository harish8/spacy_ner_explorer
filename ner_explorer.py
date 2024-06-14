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


#%%
st.title("SpaCy NER Explorer")

st.markdown(
    '''
Named Entity Recognition (NER) project using SpaCy. NER is a task in NLP where we identify and classify entities in text, 
like names of people, organizations, locations, dates, and other important information. 
This project will help you get familiar with SpaCy and its capabilities in handling text data.    '''
    )
#%%
# load the articles dataset in a var `cnn_articles_df`
cnn_articles_df = pd.read_csv('./data/CNN_Articels_clean.csv')

#%%
st.markdown(f" ## Get summary of the dataset's structure")
buffer = io.StringIO()
cnn_articles_df.info(buf=buffer)
df_cnn_articles_info = buffer.getvalue()

st.text(df_cnn_articles_info)
# cnn_articles_df.info()

st.markdown(f" ## Get statistics of the dataset")
st.write(cnn_articles_df.describe())

#%%
st.markdown(f" ## let's take a look of first few rows how the data is looking like.")
pd.set_option("display.max_columns",200)
st.dataframe(cnn_articles_df.head())
