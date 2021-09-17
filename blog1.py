import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import joblib
import os
import pickle

st.title('Multi-class Sentiment Classification of customer text in Customer Support')
st.header('Overview:')
st.markdown("Customer Support across the globe face multiple challenges when try to solve customer queries. Very often customer support agents face a lot of issues identifying sentiments and the intent of the customer at hand. We will build a sub-system which will identify the customer’s emotion. To build a system which can understand the text given by a user, and identify the underlying emotions expressed by the user our text data has words which reflect very close to Greetings, Backstory, Justification, Gratitude, Rants, or Expressing Emotions. Identifying emotion beforehand and proceeding with the customer cautiously can elevate the customer’s mood, as well as help us in resolving their query.")
st.write('\n')
st.header('ML Formulation of Business Problem:')
st.markdown('Customer Support across the globe is trying to resolve customer issues. Identifying such issues, and solving them can be tricky. One needs to have subject expertise, but very often subject expertise isn’t enough if customers aren’t handled well enough emotionally. We will try to build a sub-system which can identify the customer emotion before a particular response is sent to the customer. Depending on the emotional state of the customer, we will carefully use a sentence that can address the emotions of the customer and then send them a first response as the emotion is identified. This type of a system can also be used on forums, when customers sometimes express their grievances on a forum.')
st.markdown("Tourism industry is something that needs to handle their customers more courteously as the industry solely depends on how a customer is being treated. Customers who are willing to pay, want the services that they pay for, and are typically in rush and excited. We want to address these customers with care.")
st.write('\n')
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
st.markdown("2. Hamming loss: The Hamming loss is the fraction of labels that are incorrectly predicted.")