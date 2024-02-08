import pandas as pd
import glob
import os
import numpy as np
import re
from nltk import sent_tokenize
import pdfplumber
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
import spacy


def extract(path):
    all_page_texts = ""
    
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            print(f"Page {i + 1}:\n{text}")
            
            # Concatenate the text for the current page to the accumulated text
            all_page_texts += f"Page {i + 1}:\n{text}\n\n"
    
    lines = all_page_texts.split('\n')
    # Remove empty lines and strip whitespace
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    
    return cleaned_lines