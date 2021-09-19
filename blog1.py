import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from google_drive_downloader import GoogleDriveDownloader as gdd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import joblib
import os
import pickle


gdd.download_file_from_google_drive(file_id='13VXNiG3d98apB7L8f3luF-L4wyF3mxxx',
                                    dest_path='/app/cs1-blog/text_corpus.csv',
                                    unzip=False)


gdd.download_file_from_google_drive(file_id='1LjjAWYIoEhMwOsIzdAuj0oBlGnz07Eqy',
                                    dest_path='/app/cs1-blog/micro-f1.png',
                                    unzip=False)

gdd.download_file_from_google_drive(file_id='16xqcoh3Tf4FOTEQdzVqqOI6izf-QWr8R',
                                    dest_path='/app/cs1-blog/download.png',
                                    unzip=False)

gdd.download_file_from_google_drive(file_id='1DC2bLhjAOLiZLSJVG4JhHjviO2xtAVvG',
                                    dest_path='/app/cs1-blog/roc.png',
                                    unzip=False)

gdd.download_file_from_google_drive(file_id='14dMuT_7VHMDejyB_DckEIIad4_JxwQyn',
                                    dest_path='/app/cs1-blog/auc.png',
                                    unzip=False)


gdd.download_file_from_google_drive(file_id='1GydE6I9rfqgdBHrsYUi0b3J73V_t2f-O',
                                    dest_path='/app/cs1-blog/ai.jpg',
                                    unzip=False)

df = pd.read_csv('text_corpus.csv')

st.title('Multi-class Sentiment Classification of customer text in Customer Support')
st.write('\n')
st.image('ai.jpg')
st.write('\n')
st.markdown("""“We have seen AI providing conversation and comfort to the lonely; we have also seen AI engaging in racial discrimination. Yet the biggest harm that AI is likely to do to individuals in the short term is job displacement, as the amount of work we can automate with AI is vastly larger than before. As leaders, it is incumbent on all of us to make sure we are building a world in which every individual has an opportunity to thrive.”""")
st.markdown("Andrew NG - Co-founder and lead of Google Brain")

st.header('Overview:')
st.markdown("Customer Support across the globe face multiple challenges when try to solve customer queries. Very often customer support agents face a lot of issues identifying sentiments and the intent of the customer at hand. Customers across the globe have faced issues at certain point in their life when the customer support did not fully understand their issue, and did not comprehend how the customer felt emotion-wise. This left the customers feeling either incomplete or not convinced about the solution that they got at the end of their conversation with the support team. We will build a sub-system which will identify the customer’s emotion. To build such a system which can understand the context given by the user, and identify the underlying emotions expressed we need to annotate such texts manually and train our algorithm on huge corpus of data. Our data should have words which resemeble various human emotions as well as the context needed to understand what the user is really trying to convey. Identifying emotion beforehand and proceeding with the customers cautiously can not just solve a lot of business problems but also elevate the customer’s state of mind, as well as help us in resolving their query.")
st.write('\n')
st.header('ML Formulation of Business Problem:')
st.markdown('Customer Support across the globe is trying to resolve customer issues. Identifying such issues, and solving them can be tricky. One needs to have subject expertise, but very often subject expertise isn’t enough if customers aren’t handled well enough emotionally. We will try to build a sub-system which can identify the customer emotion before a particular response is sent to the customer. Depending on the emotional state of the customer, we will carefully use a sentence that can address the emotions of the customer and then send them a first response as the emotion is identified. This type of a system can also be used on forums, when customers sometimes express their grievances on a forum.')
st.markdown("Tourism industry is something that needs to handle their customers more courteously as the industry solely depends on how a customer is being treated. Customers who are willing to pay, want the services that they pay for, and are typically in rush and excited. We want to address these customers with care.")
st.write('\n')
st.header('Source of Data:')
st.markdown('Verint Next IT designs and builds IVAs on behalf of other companies, typically  for customer service automation. This allows them to access large number of IVA-Human conversation that vary widely across domains. Ian Beaver, Cynthia Freeman from Verint and Abdullah Mueen from University of New Mexico were kind enough to share this data on Kaggle so this problem can be explored and solved by people from various domains across the globe.')
st.markdown('TripAdvisor.com is commonly used in literature as a source of travel-related data (Banerjee and Chua 2016; Valdivia, Luz´on, and Herrera 2017). The TripAdvisor.com airline forum includes discussions of airlines and polices, flight pricing and comparisons, flight booking websites, airports, and general flying tips and suggestions. The authors observed that the intentions of requests posted by users were very similar to that of requests handled by airline travel IVA. While a forum setting is a different type of interaction than chatting with a customer service representative (user behavior is expected to differ when the audience is not paid to respond), it was the best fit that we could obtain for our study. A random sample of 2,000 threads from the 62,736 present during August 2016 was taken, and the initial post containing the problem statement was extracted. We use request hereafter to refer to the complete text of an initial turn or post extracted as described.')
st.header('Business Constraints:')
st.markdown('1. Time constraint is not a constraint as we want to take a minimum of half hour to address the customer.')
st.markdown('2. Incorrect response will affect the travel agency')
st.write('\n')
st.header('Dataset Column Analysis:')
st.markdown('Text - Consists of texts by the customer. This can be a query, or a grievance or issues related to flight/train/bus booking. One-Hot Encoded Labels.')
st.markdown('Greeting - If the text contains e.g. “Good morning”, “Good evening” etc such emotions it is marked with ‘1’ else ‘0’.')
st.markdown('Backstory - If a customer gives us some background of what they are saying e.g. “..My son’s graduation..”, “..husband likes to complain..”')
st.markdown('Justification - If the customer gives a reason for taking some actions or booking or any related things. “..the booking was created by mistake in rush..”, “..cancel booking..”')
st.markdown('Gratitude - If a customer appreciated the services or response. E.g “thank you!”, “god bless”')
st.markdown('Ranting - expressing dissatisfaction, negative emotions')
st.markdown('Expressing emotions - Expressions not covered by rant like “ugh!”, ‘aah’, ‘i love that’')
st.markdown('Other - Repetitive statements, dates and time, not relational.')
st.write('\n')
st.header("Performance Metrics:")
st.markdown('1. Micro-Averaged F1-Score (Mean F Score):')
st.markdown("* The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.")
st.markdown("* The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)")
st.markdown("* In the multi-class and multi-label case, this is the weighted average of the F1 score of each class.")
st.markdown("* Micro f1 score: Calculate metrics globally by counting the total true positives, false negatives and false positives. This is a better metric when we have class imbalance.")
st.markdown("* Macro f1 score: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.")
st.image('micro-f1.png')
st.write('\n')
st.markdown("2. Hamming loss:")
st.markdown("* The Hamming loss is the fraction of labels that are incorrectly predicted.")
st.markdown("* Hence, for the binary case (imbalanced or not), HL=1-Accuracy")
st.image('download.png')
st.write('\n')
st.markdown("3. ROC-AUC:")
st.markdown("* An ROC curve is a graph showing the performance of a classification model at all classification thresholds.")
st.markdown("""* AUC stands for "Area under the ROC Curve." That is, AUC measures the entire two-dimensional area underneath the entire ROC curve (think integral calculus) from (0,0) to (1,1).""")
st.markdown("""* An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives.""")
st.markdown("""* AUC provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example.""")
st.markdown('* ROC Curve')
st.image('roc.png')
st.write('\n')
st.markdown('* AUC Curve')
st.image('auc.png')

import string
import re
from collections import Counter
from collections import defaultdict

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
  
for i in range(df.shape[0]):
  df.at[i, 'Text'] = decontracted(df['Text'].values[i])
  
def Find(string):
  # findall() has been used 
  # with valid conditions for urls in string
  regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
  url = re.findall(regex,string) 
  temp = ''
  for x in url:
    temp+=''.join(x[0])
  return temp
  
 
def clean_text(df, feature):
  cleaned_text = []
    
  for i in tqdm(range(df.shape[0])):
    
    doc = df[feature].values[i]
        
    url = Find(doc)
        
    doc = re.sub(url, '', doc)
        
    doc = re.findall(r'\w+', doc)
        
    table = str.maketrans('', '', string.punctuation)
        
    stripped = [w.translate(table) for w in doc]
        
    doc = ' '.join(stripped)
        
    doc = doc.lower()

    # remove text followed by numbers
    doc = re.sub('[^A-Za-z0-9]+', ' ', doc)

    # remove text which appears inside < > or text preceeding or suceeding <, >
    doc = re.sub(r'< >|<.*?>|<>|\>|\<', ' ', doc)

    # remove anything inside brackets
    doc = re.sub(r'\(.*?\)', ' ', doc)
        
    # remove digits
    doc = re.sub(r'\d+', ' ', doc)
    cleaned_text.append(doc)
        
  return cleaned_text
  
df.drop(['MergedSelections','Unselected','Selected','Threshold','SentenceID'], axis=1, inplace=True)

st.write('\n\n')
st.header('Dataset Preview')
st.dataframe(df.head())
