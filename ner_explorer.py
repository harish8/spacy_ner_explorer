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
# load the articles dataset in a var `cnn_articles_df`
cnn_articles_df = pd.read_csv('./data/CNN_Articels_clean.csv').set_index('Index').reset_index(drop=True)

#%%
st.markdown(" ## Get summary of the dataset's structure")
buffer = io.StringIO()
cnn_articles_df.info(buf=buffer)
df_cnn_articles_info = buffer.getvalue()

st.text(df_cnn_articles_info)
# cnn_articles_df.info()


cnn_articles_df.rename(columns = {
    'Date Published' : 'DatePublished',
    'Second headline' : 'SecondHeadline',
    'Article text' : 'ArticleText'
        })

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
            Although there are many intresting sections to dive into I'm curious to dig deeper 
            into `tech` under `business` category. Wonder how tech is evolved since 2011 to 2022. 
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
