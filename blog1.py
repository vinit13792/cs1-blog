import streamlit as st
import numpy as np
import seaborn as sns
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
plt.style.use('seaborn')


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
st.markdown('TripAdvisor.com is commonly used in literature as a source of travel-related data (Banerjee and Chua 2016; Valdivia, Luz´on, and Herrera 2017). The TripAdvisor.com airline forum includes discussions of airlines and polices, flight pricing and comparisons, flight booking websites, airports, and general flying tips and suggestions. The authors observed that the intentions of requests posted by users were very similar to that of requests handled by airline travel IVA. While a forum setting is a different type of interaction than chatting with a customer service representative (user behavior is expected to differ when the audience is not paid to respond), it was the best fit that authors could obtain for their study. A random sample of 2,000 threads from the 62,736 present during August 2016 was taken, and the initial post containing the problem statement was extracted.')
st.header('Business Constraints:')
st.markdown('1. Time constraint is not a constraint as we want to take a minimum of half hour to address the customer.')
st.markdown('2. Incorrect response will affect the travel agency')
st.write('\n')
st.header('Dataset Column Analysis:')
st.markdown('1. Text - Consists of texts by the customer. This can be a query, or a grievance or issues related to flight/train/bus booking. One-Hot Encoded Labels.')
st.markdown('2. Greeting - If the text contains e.g. “Good morning”, “Good evening” etc such emotions it is marked with ‘1’ else ‘0’.')
st.markdown('3. Backstory - If a customer gives us some background of what they are saying e.g. “..My son’s graduation..”, “..husband likes to complain..”')
st.markdown('4. Justification - If the customer gives a reason for taking some actions or booking or any related things. “..the booking was created by mistake in rush..”, “..cancel booking..”')
st.markdown('5. Gratitude - If a customer appreciated the services or response. E.g “thank you!”, “god bless”')
st.markdown('6. Ranting - expressing dissatisfaction, negative emotions')
st.markdown('7. Expressing emotions - Expressions not covered by rant like “ugh!”, ‘aah’, ‘i love that’')
st.markdown('8. Other - Repetitive statements, dates and time, not relational.')
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
st.write('\n')
st.header('Existing Solutions:')
st.markdown('Mondher Bouazizi and Tomoaki Ohtsuki in their 2019 IEEE paper came up with a really interesting approach to preprocess text data on tweets. Their approach was to break up a sentence into multiple portions, take each element and count its occurence. A very simple approach, yet very powerful. They use a tool called SENTA which extracts multiple characteristics of a sentence like punctuations, syntactic features, unigram features, sentiment related features, pattern related features.')
st.markdown('Apart from the common methods known at the time, they also came up with new application of something known as synsets. Synsets is just a fancy way of saying synonym of a synonym. When you take synonym of a word, you have one synonym, but when you find multiple synonyms of a single word, you a get a set of words which are synonyms of a root word. This level of Depth is Depth 1 Synsets. And when you do this again on but this time on each word in this set, this level of depth is called Depth 2 Synsets. The authors came up with an approach to find synsets till Depth 4.')
st.markdown('This approach achieved a overall score of 60.2% accuracy on tweets for a Multiclass problem when the authors tested this on dataset having 7 sentiments. And on a binary classification acheived 81.3% accuracy.')
st.header('Exploratory Data Analysis:')
st.markdown('Exploratory data analysis is a set of techniques that have been principally developed by John Wilder Tukey, since 1970. The philosophy behind this approach is to examine the data before applying a specific probability model. According to J.W. Tukey, exploratory data analysis is similar to detective work. In exploratory data analysis, these clues can be numerical and (very often) graphical. Indeed, Tukey introduced several new semigraphical data representation tools to help with exploratory data analysis, including the “box and whisker plot” (also known as the box plot)in 1972, and the stem and leaf diagram in 1977. This diagram is similar to the histogram, which dates from the eighteenth century.')
st.markdown('This process involves investigating the data, to get to the core of it, and observe the patterns, behaviors, dependencies, anomalies, test hypothesis and generate summaries about it through statistical and graphical tools.')



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
    
  for i in range(df.shape[0]):
    
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

df['clean_text'] = clean_text(df, 'Text')

st.write('\n\n')
st.header('Data Preprocessing: ')

st.markdown('Our data has a lot of text. So the first step should be preprocessing. Preprocessing is one of the most important in NLP based applications as everything you do later depends on how thoroughly do the data cleaning. Mistakes in this step often cause error in modelling as most of the ML or DL models require cleaned data. Uncleaned data often causes failures/errors before modelling step.')
st.subheader('1. Removing Decontractions: ')
st.markdown("""Decontractions is a process of expanding words which are shortened words, words like "can't", "should've", "I'll" etc. ML libraries are advanced enough to encode words, but those words also need to be simplified for ingestion This process is one of those where we need to simplify shortened words.""")
st.markdown('We will use regex for this as it is fast and results are accurate.')
st.markdown("The line re.sub takes three most important arguments i.e. current_word, replacement_word, variable which has the text. It's really that simple.")
with st.echo(code_location='below'):
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
  
st.subheader('2. Cleaning the data: ')
st.markdown("""Now that we have decontracted the words in our dataset, the next step should be to remove punctuations, urls, digits, and words appearing inside brackets. """)
with st.echo(code_location='below'):
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
    
    for i in range(df.shape[0]):
    
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
st.markdown("One thing to add here is we don't need all the preprocessing concepts that exist in NLP always. Our preprocessing steps should depend on our use case. As we can see above I have not used preprocessing techniques like stemming, lemmatization, or removing stop words. The reason we did not use these techniques is because we need to capture the pattern features, syntactical features, as well as parts of speech. Cleaning the data with such techniques will cause us to loose imformation which is critical to us in this case.")


def get_sent_dict(df):
  emotions = ['Greeting', 'Backstory', 'Justification', 'Rant', 'Gratitude', 'Other', 'Express Emotion']
  sent_dict = dict()

  for i in range(len(emotions)):
    sent_dict[emotions[i]] = df[df[emotions[i]]==1].shape[0]
    sent_dict[emotions[i] + ' ' + '&' + ' ' + emotions[i-1]] = df[(df[emotions[i]]==1) & (df[emotions[i-1]]==1)].shape[0]
    sent_dict[emotions[i] + ' ' + '&' + ' ' + emotions[i-1] + ' ' + '&' + ' ' + emotions[i-2]] = df[(df[emotions[i]]==1) & (df[emotions[i-1]]==1) & (df[emotions[i-2]]==1)].shape[0]
    sent_dict[emotions[i] + ' ' + '&' + ' ' + emotions[i-1] + ' ' + '&' + ' ' + emotions[i-2] + ' ' + '&' + ' ' + emotions[i-3]] = df[(df[emotions[i]]==1) & (df[emotions[i-1]]==1) & 
                                                                                                                                                               (df[emotions[i-2]]==1)
                                                                                                                                                               & (df[emotions[i-3]]==1)].shape[0]
    sent_dict[emotions[i] + ' ' + '&' + ' ' + emotions[i-1] + ' ' + '&' + ' ' + emotions[i-2] + ' ' + '&' + ' ' + emotions[i-3]  + ' ' + '&' + ' ' + emotions[i-4]] = df[(df[emotions[i]]==1) & 
                                                                                                                                                                       (df[emotions[i-1]]==1) & 
                                                                                                                                                               (df[emotions[i-2]]==1)
                                                                                                                                                               & (df[emotions[i-3]]==1)
                                                                                                                                                               & (df[emotions[i-4]]==1)].shape[0]
    sent_dict[emotions[i] + ' ' + '&' + ' ' + emotions[i-1] + ' ' + '&' + ' ' + emotions[i-2] + ' ' + '&' + ' ' + emotions[i-3] + ' ' + '&' + ' ' + emotions[i-4] + ' ' + '&' + ' ' + emotions[i-5]] = df[(df[emotions[i]]==1) & 
                                                                                                                                                                       (df[emotions[i-1]]==1) & 
                                                                                                                                                               (df[emotions[i-2]]==1)
                                                                                                                                                               & (df[emotions[i-3]]==1)
                                                                                                                                                               & (df[emotions[i-4]]==1)
                                                                                                                                                               & (df[emotions[i-5]]==1)].shape[0]
    sent_dict[emotions[i] + ' ' + '&' + ' ' + emotions[i-1] + ' ' + '&' + ' ' + emotions[i-2] + ' ' + '&' + ' ' + emotions[i-3] + ' ' + '&' + ' ' + emotions[i-4] + ' ' + '&' + ' ' + emotions[i-5] + ' ' + '&' + ' ' + emotions[i-6]] = df[(df[emotions[i]]==1) & 
                                                                                                                                                                       (df[emotions[i-1]]==1) & 
                                                                                                                                                               (df[emotions[i-2]]==1)
                                                                                                                                                               & (df[emotions[i-3]]==1)
                                                                                                                                                               & (df[emotions[i-4]]==1)
                                                                                                                                                               & (df[emotions[i-5]]==1)
                                                                                                                                                               & (df[emotions[i-6]]==1)].shape[0]
  return sent_dict
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
sent_dict = get_sent_dict(df)

def get_plot(sentence_dict):
  
  keys = list(sentence_dict.keys())
  vals = list(sentence_dict.values())

  gen_plot = plt.figure(figsize=(20,5))
  plt.bar(keys, vals, align='center', edgecolor='black')

  for i in range(len(vals)):
    plt.text(i, vals[i], vals[i], ha='center', Bbox = dict(facecolor = 'indianred', alpha =.4))

  plt.xlabel('Sentiments')
  plt.ylabel('Datapoint')
  plt.xticks(rotation=90)
  plt.title('Count of datapoints per sentiment')
  
  return gen_plot

gen_plot = get_plot(sent_dict)
st.header('Exploratory Data Analysis:')
st.markdown("Let's see how our dataset looks")
st.dataframe(df.head())
st.markdown('Our text has multiple labels. These labels are the target variables. But we need to check how many of these sentiments we have per sentence. So we need to know the distribution of these sentiments. We also need to know if each sentence is associated with multiple sentiments or not.')
st.markdown("To find out combination of sentiments associated with each sentence, let's write a custom function which would get us exactly that.")
with st.echo(code_location='below'):
  def get_sent_dict(df):
    emotions = ['Greeting', 'Backstory', 'Justification', 'Rant', 'Gratitude', 'Other', 'Express Emotion']
    sent_dict = dict()

    for i in range(len(emotions)):
      sent_dict[emotions[i]] = df[df[emotions[i]]==1].shape[0]
      sent_dict[emotions[i] + ' ' + '&' + ' ' + emotions[i-1]] = df[(df[emotions[i]]==1) & (df[emotions[i-1]]==1)].shape[0]
      sent_dict[emotions[i] + ' ' + '&' + ' ' + emotions[i-1] + ' ' + '&' + ' ' + emotions[i-2]] = df[(df[emotions[i]]==1) & (df[emotions[i-1]]==1) & (df[emotions[i-2]]==1)].shape[0]
      sent_dict[emotions[i] + ' ' + '&' + ' ' + emotions[i-1] + ' ' + '&' + ' ' + emotions[i-2] + ' ' + '&' + ' ' + emotions[i-3]] = df[(df[emotions[i]]==1) & (df[emotions[i-1]]==1) & 
                                                                                                                                                               (df[emotions[i-2]]==1)
                                                                                                                                                               & (df[emotions[i-3]]==1)].shape[0]
      sent_dict[emotions[i] + ' ' + '&' + ' ' + emotions[i-1] + ' ' + '&' + ' ' + emotions[i-2] + ' ' + '&' + ' ' + emotions[i-3]  + ' ' + '&' + ' ' + emotions[i-4]] = df[(df[emotions[i]]==1) & 
                                                                                                                                                                       (df[emotions[i-1]]==1) & 
                                                                                                                                                               (df[emotions[i-2]]==1)
                                                                                                                                                               & (df[emotions[i-3]]==1)
                                                                                                                                                               & (df[emotions[i-4]]==1)].shape[0]
      sent_dict[emotions[i] + ' ' + '&' + ' ' + emotions[i-1] + ' ' + '&' + ' ' + emotions[i-2] + ' ' + '&' + ' ' + emotions[i-3] + ' ' + '&' + ' ' + emotions[i-4] + ' ' + '&' + ' ' + emotions[i-5]] = df[(df[emotions[i]]==1) & 
                                                                                                                                                                       (df[emotions[i-1]]==1) & 
                                                                                                                                                               (df[emotions[i-2]]==1)
                                                                                                                                                               & (df[emotions[i-3]]==1)
                                                                                                                                                               & (df[emotions[i-4]]==1)
                                                                                                                                                               & (df[emotions[i-5]]==1)].shape[0]
      sent_dict[emotions[i] + ' ' + '&' + ' ' + emotions[i-1] + ' ' + '&' + ' ' + emotions[i-2] + ' ' + '&' + ' ' + emotions[i-3] + ' ' + '&' + ' ' + emotions[i-4] + ' ' + '&' + ' ' + emotions[i-5] + ' ' + '&' + ' ' + emotions[i-6]] = df[(df[emotions[i]]==1) & 
                                                                                                                                                                       (df[emotions[i-1]]==1) & 
                                                                                                                                                               (df[emotions[i-2]]==1)
                                                                                                                                                               & (df[emotions[i-3]]==1)
                                                                                                                                                               & (df[emotions[i-4]]==1)
                                                                                                                                                               & (df[emotions[i-5]]==1)
                                                                                                                                                               & (df[emotions[i-6]]==1)].shape[0]
    return sent_dict

st.pyplot(gen_plot)
