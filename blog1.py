import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import glob

from google_drive_downloader import GoogleDriveDownloader as gdd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import joblib
import os
import pickle
import zipfile
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
st.markdown("It is evident that people love to talk, and give a back story in tourism. Of course, they need the services, so its important that the customer representative understand where they're coming from, and how important it is for them to put their point forward in a very descriptive way. We see that they are quite some ranting, and other, which is a mix of emotions. We also see very few people being thankful for the services or query resolution. Quite a lot of people do greet during the query, which is good, although it will be interesting to see which of the datapoints have multiple sentiments associated with it.")
st.write('\n')
st.subheader("Distribution of Punctuations per sentiment")
st.markdown("As per the 2019 IEEE paper, the authors used punctuation count as a  feature. We will bed using the same in our case and plot the distribution of punctuations for each sentiment")
st.markdown('We will write a function based punctuations. We will create a dictionary of punctuations. Then we will index through each punctuations in the sentence and count them, and later plot them.')

@st.cache()
def get_punctuations(data, feature):
    
    # get unique punctuations into the list
    # We are adding one to each punctuation since we want to find ratio of punc
    # we may encounter divide by zero error 
    p_dict = dict([(k,1) for k in string.punctuation])
    
    # Get all the punctuations from text and count them
    for i in range(data.shape[0]):
        
        text = data[feature].values[i]
        
        pattern = r'\~|\!|\@|\#|\$|\%|\^|\&|\*|\(|\)|\-\+|\,|\"|\?|\.|\:|\;|\=|\[|\]|\{|\}|\_|\\|\/'
        
        punctuations = re.findall(pattern, text)
        
        for p in punctuations:
            
            if p in p_dict.keys():
                
                p_dict[p] +=1           
    
    # https://stackoverflow.com/a/26367880/10632473
    punc_per_text = defaultdict(list)
    
    for j in range(data.shape[0]):
            
        text = data[feature].values[j]
            
        punc = re.findall(pattern, text)
        
        counts = Counter(punc)
        
        for punctuation in p_dict:
                
            if counts.get(punctuation)==None:
                    
                punc_per_text[punctuation].append(1)
                
            if counts.get(punctuation)!=None:
                
                punc_per_text[punctuation].append(counts.get(punctuation))
            
    return p_dict, punc_per_text

with st.echo(code_location='below'):
  def get_punctuations(data, feature):
    
    # get unique punctuations into the list
    # We are adding one to each punctuation since we want to find ratio of punc
    # we may encounter divide by zero error 
    p_dict = dict([(i,1) for i in string.punctuation])
    
    # Get all the punctuations from text and count them
    for i in range(data.shape[0]):
        
        text = data[feature].values[i]
        
        pattern = r'\~|\!|\@|\#|\$|\%|\^|\&|\*|\(|\)|\-\+|\,|\"|\?|\.|\:|\;|\=|\[|\]|\{|\}|\_|\\|\/'
        
        punctuations = re.findall(pattern, text)
        
        for p in punctuations:
            
            if p in p_dict.keys():
                
                p_dict[p] +=1           
    
    # https://stackoverflow.com/a/26367880/10632473
    punc_per_text = defaultdict(list)
    
    for i in range(data.shape[0]):
            
        text = data[feature].values[i]
            
        punc = re.findall(pattern, text)
        
        counts = Counter(punc)
        
        for punctuation in p_dict:
                
            if counts.get(punctuation)==None:
                    
                punc_per_text[punctuation].append(1)
                
            if counts.get(punctuation)!=None:
                
                punc_per_text[punctuation].append(counts.get(punctuation))
            
    return p_dict, punc_per_text
  

def get_punc_plot(sentiment, feature):
  sentim = df[df[sentiment]==1]
  punc_dict_sentim, _ = get_punctuations(sentim, feature)
  keys = list(punc_dict_sentim.keys())
  vals = list(punc_dict_sentim.values())
  fig = plt.figure(figsize=(20,5))
  plt.bar(keys, vals, align='center', edgecolor='black')
  for i in range(len(vals)):
    plt.text(i, vals[i], vals[i], ha='center', Bbox = dict(facecolor = 'indianred', alpha =.4))
  plt.xlabel('Punctuations')
  plt.ylabel('Datapoint')
  plt.title('Count of punctuations in class')
  return fig

senti_plot_greet = get_punc_plot('Greeting', 'Text')
st.subheader('Punctuation Distribution in Greeting Class')
st.pyplot(senti_plot_greet)

senti_plot_justifn = get_punc_plot('Justification', 'Text')
st.subheader('Punctuation Distribution in Justification Class')
st.pyplot(senti_plot_justifn)

senti_plot_rant = get_punc_plot('Rant', 'Text')
st.subheader('Punctuation Distribution in Rant Class')
st.pyplot(senti_plot_rant)

senti_plot_grat = get_punc_plot('Gratitude', 'Text')
st.subheader('Punctuation Distribution in Gratitude Class')
st.pyplot(senti_plot_grat)

senti_plot_other = get_punc_plot('Other', 'Text')
st.subheader('Punctuation Distribution in Other Class')
st.pyplot(senti_plot_other)

senti_plot_expemo = get_punc_plot('Express Emotion', 'Text')
st.subheader('Punctuation Distribution in Express Emotion Class')
st.pyplot(senti_plot_expemo)

st.markdown("As we can say that when people express emotions, they use lot of periods, exclamation. While expressing emotions, people do often ask questions as well, so we can see quite some question marks.")
st.subheader("Statistics of words in each Sentiment: ")
st.markdown("When we write something to someone, we often express it in a way we see fit. Some people are less expressive, and some people are more intune with their emotions so they know exactly what to say. And there are people who just say things without keeping in context of question being asked or topic of discussion")
st.markdown('That being said, we will see how many words are spoken with respect to our dataset for each sentiment to get a sense how the population which represents our dataset express their emotions and in how many words')
min_dict = dict()
max_dict = dict()
mean_dict = dict()
  

def get_word_stats(sentiment):
  sentim = df[df[sentiment]==1]
  word_length = []

  for i in range(sentim.shape[0]):
    sent = sentim['clean_text'].values[i]

    words = re.findall(r'\w+', sent)

    word_length.append(len(words))
  
  min_words = np.min(word_length)
  max_words = np.max(word_length)
  mean_words = np.mean(word_length)
  
  return min_words, max_words, np.round(mean_words)

greet_min_words, greet_max_words, greet_mean_words = get_word_stats('Greeting')
min_dict['Greeting'] = greet_min_words
max_dict['Greeting'] = greet_max_words
mean_dict['Greeting'] = greet_mean_words

rant_min_words, rant_max_words, rant_mean_words = get_word_stats('Rant')
min_dict['Rant'] = rant_min_words
max_dict['Rant'] = rant_max_words
mean_dict['Rant'] = rant_mean_words

justifn_min_words, justifn_max_words, justifn_mean_words = get_word_stats('Justification')
min_dict['Justification'] = justifn_min_words
max_dict['Justification'] = justifn_max_words
mean_dict['Justification'] = justifn_mean_words


grat_min_words, grat_max_words, grat_mean_words = get_word_stats('Gratitude')
min_dict['Gratitude'] = grat_min_words
max_dict['Gratitude'] = grat_max_words
mean_dict['Gratitude'] = grat_mean_words

other_min_words, other_max_words, other_mean_words = get_word_stats('Other')
min_dict['Other'] = other_min_words
max_dict['Other'] = other_max_words
mean_dict['Other'] = other_mean_words

expemo_min_words, expemo_max_words, expemo_mean_words = get_word_stats('Express Emotion')
min_dict['Express Emotion'] = expemo_min_words
max_dict['Express Emotion'] = expemo_max_words
mean_dict['Express Emotion'] = expemo_mean_words


def get_word_plots(dict_):
  keys = list(dict_.keys())
  vals = list(dict_.values())

  fig = plt.figure(figsize=(20,5))
  plt.bar(keys, vals, align='center', edgecolor='black')
  for i in range(len(vals)):
    plt.text(i, vals[i], vals[i], ha='center', Bbox = dict(facecolor = 'indianred', alpha =.4))
  plt.xlabel('Emotions')
  plt.ylabel('Word Count')
  plt.title('Word count across all sentiments')
  return fig

min_fig = get_word_plots(min_dict)
st.markdown('Min Word Count across all sentiments')
st.pyplot(min_fig)
st.write('\n')

max_fig = get_word_plots(max_dict)
st.markdown('Max Word Count across all sentiments')
st.pyplot(max_fig)
st.write('\n')

mean_fig = get_word_plots(mean_dict)
st.markdown('Mean Word Count across all sentiments')
st.pyplot(mean_fig)
st.write('\n')
st.markdown("Its evident from the above graphs that Rant, Gratitude and Express Emotions are the sentiments are the classes having longer sentences. Surprisingly backstory has less number of sentences, even justification has less number of words.")

st.write("\n")
st.header("Distribution of Parts of Speech: ")
st.markdown(""""These are the "building blocks" of the language. Think of them like the parts of a house. When we want to build a house, we use concrete to make the foundations or base. We use bricks to make the walls. We use window frames to make the windows, and door frames to make the doorways. And we use cement to join them all together. Each part of the house has its own job. And when we want to build a sentence, we use the different types of word.""")
st.markdown("""There are 9 basic types of word, and they are called "parts of speech".""")
st.markdown("""The most important parts of speech are the BIG FOUR, and the verb is the king of parts of speech. Here they are, each with an example and its basic "job":""")
st.markdown(" * verb (deliver - expresses action)")
st.markdown(" * noun (computer - expresses a thing)")
st.markdown(" * adjective (yellow - tells us more about a noun)")
st.markdown(" * adverb (quickly - tells us more about a verb)")
st.markdown("The other parts of speech are mostly small words:")
st.markdown(" * pronoun (it - replaces a noun)")
st.markdown(" * preposition (on - links a noun to another word)")
st.markdown(" * determiner (the - limits a noun)")
st.markdown(" * conjunction (and - joins words)")
st.markdown(" * interjection (ouch! - expresses feeling)")

st.markdown("Before we dive deep into isnpecting distribution of the parts of speech across all the sentiments, we need to know the abbreviation NLTK library uses for tagging parts of speech.")
st.markdown("The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written in the Python programming language.")
st.markdown('Currently NLTK supports Lexical analysis: Word and text tokenizer, n-gram and collocations, Part-of-speech tagger, Tree model and Text chunker for capturing, Named-entity recognition')
pos_list =  ['CC', 'CD','DT','EX','FW','IN','JJ','JJR','JJS','LS',"MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRPS","RB","RBR","RBS","RP","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"]
desc_list = ['coordinating conjunction','cardinal digit','determiner',"existential there (like: “there is” … think of it like “there exists”)",'foreign word','preposition/subordinating conjunction',"adjective ‘big’","adjective, comparative ‘bigger’","adjective, superlative ‘biggest’","list marker 1","modal could, will","noun, singular ‘desk’","noun plural ‘desks’","proper noun, singular ‘Harrison’","proper noun, plural ‘Americans’","predeterminer ‘all the kids’","possessive ending parent‘s","personal pronoun I, he, she","possessive pronoun my, his, hers","adverb very, silently","adverb, comparative better","adverb, superlative best","particle give up","to go ‘to‘ the store","interjection errrrrrrrm","verb, base form take","verb, past tense took","verb, gerund/present participle taking","verb, past participle taken","verb, sing. present, non-3d take","verb, 3rd person sing. present takes","wh-determiner which","wh-pronoun who, what","possessive wh pronoun whose","wh-abverb where, when"]
pos_tab = pd.DataFrame()
pos_tab['POS'] = pos_list
pos_tab['Description'] = desc_list
st.markdown("Following is the table all the parts of speech tagging available in NLTK and its description so we understand the nomenclatures correctly.")
st.table(pos_tab)

st.markdown("Now that we know what are parts of speech, and why they are important, let's see how they are distributed acrosss different sentiments in our dataset.")

from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))


def pos_vec(sentiment):
  sentim = df[df[sentiment]==1]
  pos_vector = []

  for i in range(sentim.shape[0]):

    doc = sentim['Text'].values[i]

    tokenized = sent_tokenize(doc)
    for i in tokenized:

      # Word tokenizers is used to find the words 
      # and punctuation in a string
      wordsList = nltk.word_tokenize(i)
  
      # removing stop words from wordList
      wordsList = [w for w in wordsList if not w in stop_words] 
  
      #  Using a Tagger. Which is part-of-speech 
      # tagger or POS-tagger. 
      tagged = nltk.pos_tag(wordsList)
  
      for tag in tagged:
        pos_vector.append(tag[1])
    
  return pos_vector

def plot_pos_dist(data):
  pos = dict(Counter(data))

  keys = list(pos.keys())
  vals = list(pos.values())

  fig = plt.figure(figsize=(20,5))
  plt.bar(keys, vals, align='center', edgecolor='black')
  
  for i in range(len(vals)):
    plt.text(i, vals[i], vals[i], ha='center', Bbox = dict(facecolor = 'indianred', alpha =.4))
  plt.xlabel('Parts of Speech')
  plt.ylabel('Count')
  plt.title('Parts of Speech Count')
  return fig

with st.echo(code_location='below'):
  import nltk
  from nltk.corpus import stopwords
  from nltk.tokenize import word_tokenize, sent_tokenize
  nltk.download('stopwords')
  nltk.download('punkt')
  nltk.download('averaged_perceptron_tagger')
  stop_words = set(stopwords.words('english'))

  def pos_vec(sentiment):
    sentim = df[df[sentiment]==1]
    pos_vector = []

    for i in range(sentim.shape[0]):
      doc = sentim['Text'].values[i]

      tokenized = sent_tokenize(doc)
      for i in tokenized:
        
        # Word tokenizers is used to find the words 
        # and punctuation in a string
        wordsList = nltk.word_tokenize(i)
  
        # removing stop words from wordList
        wordsList = [w for w in wordsList if not w in stop_words] 
  
        #  Using a Tagger. Which is part-of-speech 
        # tagger or POS-tagger. 
        tagged = nltk.pos_tag(wordsList)
  
        for tag in tagged:
            pos_vector.append(tag[1])
    
    return pos_vector
            
st.write('\n')
st.markdown('Parts of Speech Count of Greeting Class')
greet_pos_vector = pos_vec('Greeting')
greet_pos = plot_pos_dist(greet_pos_vector)
st.pyplot(greet_pos)

st.write('\n')
st.markdown('Parts of Speech Count of Justification Class')
justifn_pos_vector = pos_vec('Justification')
justifn_pos = plot_pos_dist(justifn_pos_vector)
st.pyplot(justifn_pos)

st.write('\n')
st.markdown('Parts of Speech Count of Rant Class')
rant_pos_vector = pos_vec('Rant')
rant_pos = plot_pos_dist(rant_pos_vector)
st.pyplot(rant_pos)

st.write('\n')
st.markdown('Parts of Speech Count of Gratitude Class')
grat_pos_vector = pos_vec('Gratitude')
grat_pos = plot_pos_dist(grat_pos_vector)
st.pyplot(grat_pos)

st.write('\n')
st.markdown('Parts of Speech Count of Other Class')
other_pos_vector = pos_vec('Other')
other_pos = plot_pos_dist(other_pos_vector)
st.pyplot(other_pos)

st.write('\n')
st.markdown('Parts of Speech Count of Express Emotion Class')
expemo_pos_vector = pos_vec('Express Emotion')
expemo_pos = plot_pos_dist(expemo_pos_vector)
st.pyplot(expemo_pos)
st.write('\n')

            
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

def get_wordcloud(sentiment):
  sentim = df[df[sentiment]==1]
  comment_words = ''
  stopwords = set(STOPWORDS)
  
  # iterate through the csv file
  for val in sentim['clean_text']:
      
      # typecaste each val to string
      val = str(val)
  
      # split the value
      tokens = val.split()
      
      # Converts each token into lowercase
      for i in range(len(tokens)):
          tokens[i] = tokens[i].lower()
      
      comment_words += " ".join(tokens)+" "

  wordcloud = WordCloud(width = 2048, height = 1024,background_color ='black', stopwords = stopwords, min_font_size = 20).generate(comment_words)
  
  # plot the WordCloud image                       
  fig = plt.figure(figsize = (10, 8), facecolor = None)
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.tight_layout(pad = 0)
  return fig

st.header("Inspecting most common words using Wordcloud per Sentiment: ")
st.markdown("We need to see for ourself that which are the most common words used in our dataset. It shouldn't be that surprising since its a travel data, so most often words used could be related to travelling, flights, trains, reservations, bookings and so forth.")
with st.echo(code_location='below'):
  def get_wordcloud(sentiment):
    sentim = df[df[sentiment]==1]
    comment_words = ''
    stopwords = set(STOPWORDS)
  
    # iterate through the csv file
    for val in sentim['clean_text']:
      # typecaste each val to string
      val = str(val)
  
      # split the value
      tokens = val.split()
      
      # Converts each token into lowercase
      for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
      
      comment_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width = 2048, height = 1024,background_color ='black', stopwords = stopwords, min_font_size = 20).generate(comment_words)
  
    # plot the WordCloud image                       
    fig = plt.figure(figsize = (10, 8), facecolor = None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad = 0)
    return fig
            
greet_wc = get_wordcloud('Greeting')
st.subheader("Frequent words in the Greeting Class")
st.pyplot(greet_wc)

st.write('\n')
justifn_wc = get_wordcloud('Justification')
st.subheader("Frequent words in the Justification Class")
st.pyplot(justifn_wc)

st.write('\n')
rant_wc = get_wordcloud('Rant')
st.subheader("Frequent words in the Rant Class")
st.pyplot(rant_wc)

st.write('\n')
grat_wc = get_wordcloud('Gratitude')
st.subheader("Frequent words in the Gratitude Class")
st.pyplot(grat_wc)
            
st.write('\n')
other_wc = get_wordcloud('Other')
st.subheader("Frequent words in the Other Class")
st.pyplot(other_wc)

st.write('\n')
expemo_wc = get_wordcloud('Express Emotion')
st.subheader("Frequent words in the Express Emotion Class")
st.pyplot(expemo_wc)

st.markdown("As I said it shouldn't be surprising to find travel related words more often even though the words associated to sentiments were less seen.")

st.write('\n')
st.markdown("On inspecting our dataset, I came to know that a lot of files do not have labels. So inorder to build model, we need only those datapoints which have labels. So lets remove the ones which do not have labels.")

with st.echo(code_location='below'):
  # Ref: https://stackoverflow.com/a/43421391/10632473
  index_list = list(df[(df['Greeting']==0) & (df['Backstory']==0) & (df['Justification']==0) & (df['Rant']==0) & (df['Gratitude']==0) & (df['Other']==0) & (df['Express Emotion']==0)].index)
  indexes_to_keep = set(range(df.shape[0])) - set(index_list)
  df_sliced_multi = df.take(list(indexes_to_keep))

st.markdown("Now we need to verify if those datapoints are removed which do not have labels. Verification is important else we would be giving just garbage to our model.")
with st.echo(code_location='below'):
  try:
    df_sliced_multi.loc[index_list]
  except KeyError:
    print('Indexes doesnt not exist')
    print('Indexes from index lists removed')

st.write('\n')
st.markdown("Now that we have verified that we have all the valid datapoints, let's split our dataset into 70% train and 30% test")
with st.echo(code_location='below'):
  from sklearn.model_selection import train_test_split
  
  X_multi = df_sliced_multi[[df_sliced_multi.columns[0]]]
  y_multi = df_sliced_multi[df_sliced_multi.columns[1:-1]]
  
  X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.33, random_state=42)
  X_train_multi, X_cv_multi, y_train_multi, y_cv_multi = train_test_split(X_train_multi, y_train_multi, test_size=0.33, random_state=42)

st.write('\n')
st.markdown("Till now we saw data cleaning, and data visualisation. Now we have reached at the most important part of the blog, Feature Engineering.")
st.markdown("Feature Engineering is like the art and creative part of a Machine Learning project. The more creative you get, the more features we get more descriptive our dataset becomes for our model. Although this does'nt hold true in all the cases.")
st.header("Feature Engineering: ")
st.subheader("Data Augmentation on Text Data:-")
st.markdown("Data Augmentation on text data is actually quite simple. You find the words which has synonyms, replace that word without changing the meaning of the sentence.")
st.markdown("We will use a library called __nlpaug__ which uses bert model and different versions of bert to find and replace words.")
st.markdown("We will use contextual word embeddings and synonym word augmenter to find and replace such words.")
st.markdown("Let's see how this works from a code perspective")
code = """
  import nlpaug.augmenter.char as nac
  import nlpaug.augmenter.word as naw
  import nlpaug.augmenter.sentence as nas
  import nlpaug.flow as nafc

  from nlpaug.util import Action
  
  
  text = df_sliced_multi['Text'].values[20]
  con_aug = naw.ContextualWordEmbsAug(model_path='bert-large-uncased', action="insert", device='cuda')
  augmented_text = con_aug.augment(text)
  print("Original:")
  print(text)
  print("Augmented Text:")
  print(augmented_text)

  Output:
  Original:
  Does anyone know if there is a particular time of year when airlines tend to have good prices on airfare? We are going on a Carnival Cruise in June 2007 and want to find the best deal on airfare. Thanks!
  Augmented Text:
  does anyone know if there is a particular time of year when european airlines tend to quite have good prices on airfare? they we present are going her on a carnival cruise in her june with 2007 and want congratulations to find possibly the best deal on their airfare. thanks!
  """
st.code(code, language='python')

st.markdown("As you can see the contextual word embeddings keep the context and meaning of the sentence as it is, but add certain words like 'european' which add a new meaning, but at the same time didnt change what the original sentence meant. Similarly it changed other parts of the sentence as well.")
st.write('\n')
st.markdown("Let's look at synonym augmenter to see how it augments the sentence")
code_1 = """import nltk
nltk.download('wordnet')
syn_aug = naw.SynonymAug(aug_src='wordnet')
augmented_text = syn_aug.augment(text)
augmented1 = syn_aug.augment(augmented_text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)
print(augmented1)

Original:
Does anyone know if there is a particular time of year when airlines tend to have good prices on airfare? We are going on a Carnival Cruise in June 2007 and want to find the best deal on airfare. Thanks!
Augmented Text:
Does anyone have it away if at that place is a special time of year when airlines tend to hold salutary terms on airfare? We are going on a Carnival Cruise in June 2007 and want to find the best deal on airfare. Thanks!
Does anyone ingest it away if at that station be a special time of twelvemonth when airlines tend to bear salutary footing on airfare? We are going on a Carnival Cruise in June 2007 and want to find the best deal on airfare. Thanks

"""
st.code(code_1, 'python')

st.markdown("""As you can see synonym augmenter just replaces some words here and there without keeping the grammatical structure intact. It does make the sentence flawed the way any humans are, which makes it slightly authentic, but not entirely, although since we have less dataset and we need more data to train our model, so this works.""")

gdd.download_file_from_google_drive(file_id='1kFgT7fbdVeTYpi9HRTqgB-N6ROsYyuZ6',
                                    dest_path='/app/cs1-blog/data.zip',
                                    unzip=False)


zipfiles = glob.glob("/app/cs1-blog/*.zip")
#st.write(zipfiles)
for file in zipfiles:
    with zipfile.ZipFile(f'{file}', 'r') as zip_ref:
        zip_ref.extractall()
text_multi = pickle.load(open("/app/cs1-blog/data/text_multi.pkl", "rb"))
labels_multi = pickle.load(open("/app/cs1-blog/data/labels_multi.pkl", "rb"))

syn_text_multi = pickle.load(open("/app/cs1-blog/data/syn_text_multi.pkl", "rb"))
syn_labels_multi = pickle.load(open("/app/cs1-blog/data/syn_labels_multi.pkl", "rb"))

cols = ['Greeting','Backstory','Justification','Rant','Gratitude','Other','Express Emotion']
y_dict = {cols[0]: syn_labels_multi[:,0], cols[1]:syn_labels_multi[:,1], cols[2]:syn_labels_multi[:,2], cols[3]:syn_labels_multi[:,3],
          cols[4]:syn_labels_multi[:,4], cols[5]:syn_labels_multi[:,5], cols[6]:syn_labels_multi[:,6]}

y_aug_dict = {cols[0]: labels_multi[:,0], cols[1]:labels_multi[:,1], cols[2]:labels_multi[:,2], cols[3]:labels_multi[:,3],
          cols[4]:labels_multi[:,4], cols[5]:labels_multi[:,5], cols[6]:labels_multi[:,6]}

y_syn_train = pd.DataFrame(y_dict, columns=cols)
y_new_train = pd.DataFrame(y_aug_dict, columns=cols)

y_new_train = y_new_train.append(y_syn_train)

y_train_multi = y_train_multi.append(y_new_train)

y_train_multi = y_train_multi.append(y_new_train)

syn_dict_multi = {'Text':syn_text_multi}

syn_df_multi = pd.DataFrame(syn_dict_multi.get('Text'))
syn_df_multi.rename(columns={0:'Text'}, inplace=True)

text_dict_multi = {'Text':text_multi}
aug_df_multi = pd.DataFrame(text_dict_multi.get('Text'))
aug_df_multi.rename(columns={0:'Text'}, inplace=True)

X_train_multi = X_train_multi.append(aug_df_multi)
X_train_multi = X_train_multi.append(syn_df_multi)


st.write('\n')
st.subheader("Word Count per Sentence:")
st.markdown("Emotions are expressed in different way. Some emotions require less words, and some more. But it is really dependent on people how to explain, in few words or more words. Capturing this count of words per emotion or per multiple emotions will help us build a better model")

def get_word_count(data, feature):
    
    counts = []
    
    for i in range(data[feature].shape[0]):
        
        text = data[feature].values[i]
        pattern = r'[a-zA-Z]+'
        
        words = re.findall(pattern, text)
        
        counts.append(len(words))
        
    return counts
  
code_2 = """def get_word_count(data, feature):
    
    counts = []
    
    for i in range(data[feature].shape[0]):
        
        text = data[feature].values[i]
        pattern = r'[a-zA-Z]+'
        
        words = re.findall(pattern, text)
        
        counts.append(len(words))
        
    return counts
    
    
    word_counts_train = get_word_count(X_train_multi, 'Text')
    word_counts_cv = get_word_count(X_cv_multi, 'Text')
    word_counts_test = get_word_count(X_test_multi, 'Text')

    X_train_multi['word_counts'] = word_counts_train
    X_cv_multi['word_counts'] = word_counts_cv
    X_test_multi['word_counts'] = word_counts_test"""
st.code(code_2, 'python')

word_counts_train = get_word_count(X_train_multi, 'Text')

X_train_multi['word_counts'] = word_counts_train

st.write('\n')
st.subheader("Get the Number of particles, interjections, verbs, noun, pronoun, adverb, adjective:")

import itertools

def pos_count(data, feature):

  POS_List = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'PRP', 'PRPS', 'RB', 'RBR', 'RP', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP','VBZ', 'WP', 'WP$']

  pos_per_text = defaultdict(list)

  for i in range(data.shape[0]):

    doc = data['Text'].values[i]

    info =[]
    tokenized = sent_tokenize(doc)
    for i in tokenized:

      # Word tokenizers is used to find the words 
      # and punctuation in a string
      wordsList = nltk.word_tokenize(i)
  
      # removing stop words from wordList
      wordsList = [w for w in wordsList if not w in stop_words] 
  
      #  Using a Tagger. Which is part-of-speech 
      # tagger or POS-tagger. 
      tagged = nltk.pos_tag(wordsList)
      for tag in tagged:
        info.append(tag[1])
    
    #print(info)
    
    
    counts = Counter(info)
    
    keys = dict(counts).keys()
    #print(list(keys))
    #break
    for pos in POS_List:
      
      if pos in list(keys):
        
        pos_per_text[pos].append(counts.get(pos))
            
      else:
        pos_per_text[pos].append(0)
      
  return pos_per_text

code_3 = """ def pos_count(data, feature):

  POS_List = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'PRP', 'PRPS', 'RB', 'RBR', 'RP', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP','VBZ', 'WP', 'WP$']

  pos_per_text = defaultdict(list)

  for i in tqdm(range(data.shape[0])):

    doc = data['Text'].values[i]

    info =[]
    tokenized = sent_tokenize(doc)
    for i in tokenized:

      # Word tokenizers is used to find the words 
      # and punctuation in a string
      wordsList = nltk.word_tokenize(i)
  
      # removing stop words from wordList
      wordsList = [w for w in wordsList if not w in stop_words] 
  
      #  Using a Tagger. Which is part-of-speech 
      # tagger or POS-tagger. 
      tagged = nltk.pos_tag(wordsList)
      for tag in tagged:
        info.append(tag[1])
    
    #print(info)
    
    
    counts = Counter(info)
    
    keys = dict(counts).keys()
    #print(list(keys))
    #break
    for pos in POS_List:
      
      if pos in list(keys):
        
        pos_per_text[pos].append(counts.get(pos))
            
      else:
        pos_per_text[pos].append(0)
      
  return pos_per_text 
  
parts_of_speech_counts_train_multi = pos_count(X_train_multi, 'Text')
parts_of_speech_counts_cv_multi = pos_count(X_cv_multi, 'Text')
parts_of_speech_counts_test_multi = pos_count(X_test_multi, 'Text')

POS_List = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'PRP', 'PRPS', 'RB', 'RBR', 'RP', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP','VBZ', 'WP', 'WP$']

for pos in POS_List:
  X_train_multi[pos] = parts_of_speech_counts_train_multi.get(pos)
  X_cv_multi[pos] = parts_of_speech_counts_cv_multi.get(pos)
  X_test_multi[pos] = parts_of_speech_counts_test_multi.get(pos)
  
  """

st.code(code_3, 'python')

st.write('\n')
st.subheader('Get Negations in a sentence')
st.markdown("""Negations are words like not or contractions with 'nt. In simple words negative words are called as negations. We will create a 1 or 0 if a negative words exists in a sentence or not respectively. """)
def check_negation(df):
    
    negex_ = []
    
    for i in range(df.shape[0]):
        sent = df['Text'].values[i].split()
        if ('not' in sent) or ('never' in sent):
            negex_.append(1)
        else:
            negex_.append(0)
    return negex_

code_4 = """ def check_negation(df):
    
    negex_ = []
    
    for i in tqdm(range(df.shape[0])):
        sent = df['Text'].values[i].split()
        if ('not' in sent) or ('never' in sent):
            negex_.append(1)
        else:
            negex_.append(0)
    return negex_
    
    X_train_multi['Negation'] = check_negation(X_train_multi)
    X_cv_multi['Negation'] = check_negation(X_cv_multi)
    X_test_multi['Negation'] = check_negation(X_test_multi)"""

st.code(code_4, 'python')


X_train_multi['Negation'] = check_negation(X_train_multi)

st.header("Getting unigrams from Synonyms, Hyponyms, WordNet")
st.markdown("We get unigrams when we get each word as feature and create a one hot encoded vector. Then we get synonyms of these unigrams this is Depth level 1 of synonyms. When we find synonyms of these Depth level 1, we get Depth level 2 words, these are called hyponyms. We get these synonyms of various depths using wordnet.")

import nltk
nltk.download('wordnet')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn

train_clean_multi = clean_text(X_train_multi, 'Text')
X_train_multi['clean_text'] = train_clean_multi

def get_seed_words(data, feature):
    
    pos_type = ['NN', 'NNS', 'PRP', 'PRPS', 'RB', 'RBR','RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP','VBZ']
    
    seed_words = []
    
    for i in range(data.shape[0]):
        
        doc = data[feature].values[i]
        info =[]
        tokenized = sent_tokenize(doc)
        for i in tokenized:

          # Word tokenizers is used to find the words 
          # and punctuation in a string
          wordsList = nltk.word_tokenize(i)
  
          # removing stop words from wordList
          wordsList = [w for w in wordsList if not w in stop_words] 
  
          #  Using a Tagger. Which is part-of-speech 
          # tagger or POS-tagger. 
          tagged = nltk.pos_tag(wordsList)
        
        for tag in tagged:

          if tag[1] in pos_type:

            seed_words.append(tag[0])
    return seed_words

seed_words_multi = get_seed_words(X_train_multi, 'clean_text')

def get_words_based_on_prob(words):
    
    word_list = set()
    sid_obj = SentimentIntensityAnalyzer()
    
    for i in range(len(words)):
        
        scores = sid_obj.polarity_scores(words[i])
        
        if scores.get('compound')>0.5 or scores.get('compound')<-0.5:
            
            word_list.add(words[i])
    return word_list
  
seed_words_prob_multi = get_words_based_on_prob(seed_words_multi)
def depth_1_syn(words):
    
    synsets = []
    for word in words:
        for s in wn.synsets(word):
            synsets.append(s.hyponyms())
    
    words = []
    for s in synsets:
        t1 = re.findall(r"\b([a-zA-Z]+)\b", str(s))
        for temp in t1:
            if len(temp)>1 and temp!='Synset':
                words.append(temp)
    return set(words)
  
synsets_d1_multi = depth_1_syn(seed_words_prob_multi)
synsets_d2_multi = depth_1_syn(synsets_d1_multi)
synsets_d3_multi = depth_1_syn(synsets_d2_multi)
synsets_d4_multi = depth_1_syn(synsets_d3_multi)

word_depth_multi = dict()

word_depth_multi['Seed Words']       = len(seed_words_prob_multi)
word_depth_multi['Synsets Depth 1']  = len(synsets_d1_multi)
word_depth_multi['Synsets Depth 2']  = len(synsets_d2_multi)
word_depth_multi['Synsets Depth 3']  = len(synsets_d3_multi)
word_depth_multi['Synsets Depth 4']  = len(synsets_d4_multi)


def get_synset_plot(word_depth):
  # https://www.geeksforgeeks.org/adding-value-labels-on-a-matplotlib-bar-chart/

  keys = list(word_depth.keys())
  vals = list(word_depth.values())

  fig = plt.figure(figsize=(20,5))
  plt.bar(keys, vals, align='center', edgecolor='black')

  for i in range(len(vals)):
    plt.text(i, vals[i], vals[i], ha='center', Bbox = dict(facecolor = 'indianred', alpha =.8))
  plt.xlabel('Unigrams')
  plt.ylabel('Count')
  plt.title('Unigram Count')
  return fig

synset_fig = get_synset_plot(word_depth_multi)

code_5 = """ 
import nltk
nltk.download('wordnet')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn

train_clean_multi = clean_text(X_train_multi, 'Text')
X_train_multi['clean_text'] = train_clean_multi

def get_seed_words(data, feature):
    
    pos_type = ['NN', 'NNS', 'PRP', 'PRPS', 'RB', 'RBR','RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP','VBZ']
    
    seed_words = []
    
    for i in range(data.shape[0]):
        
        doc = data[feature].values[i]
        info =[]
        tokenized = sent_tokenize(doc)
        for i in tokenized:

          # Word tokenizers is used to find the words 
          # and punctuation in a string
          wordsList = nltk.word_tokenize(i)
  
          # removing stop words from wordList
          wordsList = [w for w in wordsList if not w in stop_words] 
  
          #  Using a Tagger. Which is part-of-speech 
          # tagger or POS-tagger. 
          tagged = nltk.pos_tag(wordsList)
        
        for tag in tagged:

          if tag[1] in pos_type:

            seed_words.append(tag[0])
    return seed_words

seed_words_multi = get_seed_words(X_train_multi, 'clean_text')

def get_words_based_on_prob(words):
    
    word_list = set()
    sid_obj = SentimentIntensityAnalyzer()
    
    for i in range(len(words)):
        
        scores = sid_obj.polarity_scores(words[i])
        
        if scores.get('compound')>0.5 or scores.get('compound')<-0.5:
            
            word_list.add(words[i])
    return word_list
  
seed_words_prob_multi = get_words_based_on_prob(seed_words_multi)
  
def depth_1_syn(words):
    
    synsets = []
    for word in words:
        for s in wn.synsets(word):
            synsets.append(s.hyponyms())
    
    words = []
    for s in synsets:
        t1 = re.findall(r"\b([a-zA-Z]+)\b", str(s))
        for temp in t1:
            if len(temp)>1 and temp!='Synset':
                words.append(temp)
    return set(words)
 """

st.code(code_5, 'python')
st.pyplot(synset_fig)

st.write('\n')
st.header("Getting the words using TFIDF Vectorizer")
st.markdown("TFIDF takes in account the less frequent occuring words, as well the more frequent occuring words in such a way that less frequently occuring words gets more importance and more frequent words gets less importance, effectively getting log probabilities.")

from sklearn.feature_extraction.text import TfidfVectorizer

gdd.download_file_from_google_drive(file_id='1KLIH1mhEKqCl4OymTOkl4ChG_b511TfD',
                                    dest_path='/app/cs1-blog/tf_model.sav',
                                    unzip=False) # tfidf model fitted on training data


def tfidf(train, feature, tfidf_mod):  
    train_tfidf = tfidf_model.transform(train[feature].values)
    return train_tfidf
  
tfidf_model = joblib.load('/app/cs1-blog/tf_model.sav')
train_tf = tfidf(X_train_multi, 'clean_text', tfidf_model)

code_6 = """ 
from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf(train, cv, test, feature, ngram_range, max_features, smooth_idf):
    tfidf_model = TfidfVectorizer(min_df=10, ngram_range=ngram_range, max_features=max_features, smooth_idf=smooth_idf)
    print(f'Feature name: {feature}')
    print("Before Vectorization:-")
    print(f'Shape of our training data: {train.shape}')
    print(f'Shape of our cross-validation data: {cv.shape}')   
    print(f'Shape of our test data: {test.shape}')
    print("-"*75)
    print(f'Initialised TFIDF model')
    print(f'\nVectorizing Data....')
    
    tfidf_model.fit(train[feature].values)
    #features = tfidf_model.get_feature_names()
    train_tfidf = tfidf_model.transform(train[feature].values)
    cv_tfidf = tfidf_model.transform(cv[feature].values)
    test_tfidf = tfidf_model.transform(test[feature].values)
    print(f'\nVectorizing Complete!')
    
    print("\nAfter Vectorization:-")
    print(f'Shape of our training data: {train_tfidf.shape}')
    print(f'Shape of our cross-validation data: {cv_tfidf.shape}')   
    print(f'Shape of our test data: {test_tfidf.shape}')
    print("-"*75)
    return train_tfidf, cv_tfidf, test_tfidf, tfidf_model
    
train_tf, cv_tf, test_tf, tf_model = tfidf(X_train_multi, X_cv_multi, X_test_multi, 'clean_text', (1,4), 5000, True)
    """

st.code(code_6, 'python')

st.write('\n')
st.header("Getting Pattern Based Features")
st.markdown("""The idea of our pattern-related features in which we proposed an approach that relies on Part of Speech tags (PoS-tags) to extract sarcastic patterns. We rely on PoS-Tag of words to extract similar patterns. However, instead of dividing words into two categories, we divide them into three:

Referred to as EI, containing words which might have emotional content.
A second one, referred to as “CI”, containing non emotional words whose content is important
And a third one, referred to as “GFI”, containing the words whose grammatical function is important.
If a word belongs to the first category, it is replaced by the corresponding expression shown in TABLE II along with its polarity (e.g., the word “good” would be replaced by POS-ADJECTIVE); if it belongs to the second, it is lemmatized and replaced by its lemma; and if it belongs to the third, it is replaced by the corresponding expression shown in TABLE II. """)

gdd.download_file_from_google_drive(file_id='1zhqF3YfALiRN5Dkm-xkfMFTLPw2d7cts',
                                    dest_path='/app/cs1-blog/pattern feat.png',
                                    unzip=False)

st.image('pattern feat.png')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

def get_pos_vec(df, feature):
  ci = ['CC', 'DT', 'EX', 'IN', 'MD', 'PDT', 'POS', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'WDT', 'WP', 'WP$', 'WRB']
  gfi = ['CD', 'FW', 'LS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'SYM', 'UH']
  ei = ['JJ', 'JJR','JJS','NN','NNS','VB','VBD','VBD','VBG','VBN','VBP','VBZ']
  pos_cat = defaultdict(list)
  
  for i in range(df.shape[0]):

    doc = df[feature].values[i]
    pattern =[]
    tokenized = sent_tokenize(doc)
    for i in tokenized:

      # Word tokenizers is used to find the words 
      # and punctuation in a string
      wordsList = nltk.word_tokenize(i)
  
      # removing stop words from wordList
      wordsList = [w for w in wordsList if not w in stop_words] 
  
      #  Using a Tagger. Which is part-of-speech 
      # tagger or POS-tagger. 
      tagged = nltk.pos_tag(wordsList)

      for tag in tagged:
        pattern.append(tag[1]) 
    
    
    pat_check_ci = [pat for pat in pattern if pat in ci]
    pat_check_ei = [pat for pat in pattern if pat in ei]
    pat_check_gfi = [pat for pat in pattern if pat in gfi]
    #print('-----')
    #print(pat_check_ci, 'ci')
    #print(pat_check_gfi, 'gfi')
    #print(pat_check_ei, 'ei')
    
    if (len(pat_check_ci)!=0) and (len(pat_check_ei)!=0) and (len(pat_check_gfi)!=0):
      pos_cat['CI'].append(1)
      pos_cat['EI'].append(1)
      pos_cat['GFI'].append(1)
    elif (len(pat_check_ci)==0) and (len(pat_check_ei)!=0) and (len(pat_check_gfi)!=0):
      pos_cat['CI'].append(0)
      pos_cat['EI'].append(1)
      pos_cat['GFI'].append(1)
    elif (len(pat_check_ci)!=0) and (len(pat_check_ei)==0) and (len(pat_check_gfi)!=0):
      pos_cat['CI'].append(1)
      pos_cat['EI'].append(0)
      pos_cat['GFI'].append(1)
    elif (len(pat_check_ci)!=0) and (len(pat_check_ei)!=0) and (len(pat_check_gfi)==0):
      pos_cat['CI'].append(1)
      pos_cat['EI'].append(1)
      pos_cat['GFI'].append(0)
    elif (len(pat_check_ci)!=0) and (len(pat_check_ei)==0) and (len(pat_check_gfi)==0):
      pos_cat['CI'].append(1)
      pos_cat['EI'].append(0)
      pos_cat['GFI'].append(0)
    elif (len(pat_check_ci)==0) and (len(pat_check_ei)==0) and (len(pat_check_gfi)!=0):
      pos_cat['CI'].append(0)
      pos_cat['EI'].append(0)
      pos_cat['GFI'].append(1)
    else:
      pos_cat['CI'].append(0)
      pos_cat['EI'].append(1)
      pos_cat['GFI'].append(0)

  return pos_cat

train_pos_cat_multi = get_pos_vec(X_train_multi, 'Text')
X_train_multi['CI'] = train_pos_cat_multi.get('CI')
X_train_multi['GFI'] = train_pos_cat_multi.get('GFI')
X_train_multi['EI'] = train_pos_cat_multi.get('EI')

code_7 = """ import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

def get_pos_vec(df, feature):
  ci = ['CC', 'DT', 'EX', 'IN', 'MD', 'PDT', 'POS', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'WDT', 'WP', 'WP$', 'WRB']
  gfi = ['CD', 'FW', 'LS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'SYM', 'UH']
  ei = ['JJ', 'JJR','JJS','NN','NNS','VB','VBD','VBD','VBG','VBN','VBP','VBZ']
  pos_cat = defaultdict(list)
  
  for i in range(df.shape[0]):

    doc = df[feature].values[i]
    pattern =[]
    tokenized = sent_tokenize(doc)
    for i in tokenized:

      # Word tokenizers is used to find the words 
      # and punctuation in a string
      wordsList = nltk.word_tokenize(i)
  
      # removing stop words from wordList
      wordsList = [w for w in wordsList if not w in stop_words] 
  
      #  Using a Tagger. Which is part-of-speech 
      # tagger or POS-tagger. 
      tagged = nltk.pos_tag(wordsList)

      for tag in tagged:
        pattern.append(tag[1]) 
    
    
    pat_check_ci = [pat for pat in pattern if pat in ci]
    pat_check_ei = [pat for pat in pattern if pat in ei]
    pat_check_gfi = [pat for pat in pattern if pat in gfi]
    #print('-----')
    #print(pat_check_ci, 'ci')
    #print(pat_check_gfi, 'gfi')
    #print(pat_check_ei, 'ei')
    
    if (len(pat_check_ci)!=0) and (len(pat_check_ei)!=0) and (len(pat_check_gfi)!=0):
      pos_cat['CI'].append(1)
      pos_cat['EI'].append(1)
      pos_cat['GFI'].append(1)
    elif (len(pat_check_ci)==0) and (len(pat_check_ei)!=0) and (len(pat_check_gfi)!=0):
      pos_cat['CI'].append(0)
      pos_cat['EI'].append(1)
      pos_cat['GFI'].append(1)
    elif (len(pat_check_ci)!=0) and (len(pat_check_ei)==0) and (len(pat_check_gfi)!=0):
      pos_cat['CI'].append(1)
      pos_cat['EI'].append(0)
      pos_cat['GFI'].append(1)
    elif (len(pat_check_ci)!=0) and (len(pat_check_ei)!=0) and (len(pat_check_gfi)==0):
      pos_cat['CI'].append(1)
      pos_cat['EI'].append(1)
      pos_cat['GFI'].append(0)
    elif (len(pat_check_ci)!=0) and (len(pat_check_ei)==0) and (len(pat_check_gfi)==0):
      pos_cat['CI'].append(1)
      pos_cat['EI'].append(0)
      pos_cat['GFI'].append(0)
    elif (len(pat_check_ci)==0) and (len(pat_check_ei)==0) and (len(pat_check_gfi)!=0):
      pos_cat['CI'].append(0)
      pos_cat['EI'].append(0)
      pos_cat['GFI'].append(1)
    else:
      pos_cat['CI'].append(0)
      pos_cat['EI'].append(1)
      pos_cat['GFI'].append(0)

  return pos_cat
  
train_pos_cat_multi = get_pos_vec(X_train_multi, 'Text')
X_train_multi['CI'] = train_pos_cat_multi.get('CI')
X_train_multi['GFI'] = train_pos_cat_multi.get('GFI')
X_train_multi['EI'] = train_pos_cat_multi.get('EI')

cv_pos_cat_multi = get_pos_vec(X_cv_multi, 'Text')
X_cv_multi['CI'] = cv_pos_cat_multi.get('CI')
X_cv_multi['GFI'] = cv_pos_cat_multi.get('GFI')
X_cv_multi['EI'] = cv_pos_cat_multi.get('EI')

test_pos_cat = get_pos_vec(X_test, 'Text')
X_test['CI'] = test_pos_cat.get('CI')
X_test['GFI'] = test_pos_cat.get('GFI')
X_test['EI'] = test_pos_cat.get('EI')


"""

st.write('\n')

st.header("Building Dictionary for Converting Text-To-Sequences")
st.markdown("When we convert a sentence to a sequence, we convert words to a series of numbers. But a model cannot do that on its own. We need to encode each word to a corresponding unique words, and then based on those unique words, we will construct a datapoint which will correspond to a sentence.")
st.markdown("We will use all the words in our synsnets to construct this.")

code_8 = """

wordcloud_backstory = set(comment_words_backstory.split())
wordcloud_expemo = set(comment_words_expemo.split())
wordcloud_gratitude = set(comment_words_gratitude.split())
wordcloud_greetings = set(comment_words_greetings.split())
wordcloud_justifn = set(comment_words_justifn.split())
wordcloud_rant = set(comment_words_rant.split())
wordcloud_other = set(comment_words_other.split())

vocab_multi = synsets_d1_multi.union(synsets_d2_multi.union(synsets_d3_multi.union(synsets_d4_multi)))

wordclouds = {'backstory':wordcloud_backstory, 'Express Emotion':wordcloud_expemo,'Gratitude':wordcloud_gratitude,
              'Greeting':wordcloud_gratitude, 'Justification':wordcloud_justifn,'Rant':wordcloud_rant,
              'Other':wordcloud_other}
synset_dict_multi={'synsets_d1':synsets_d1_multi, 'synsets_d2':synsets_d2_multi,'synsets_d3':synsets_d3_multi,'synsets_d4':synsets_d4_multi}

def get_unigram_features(voc, sentiments_cloud,synsets, df):

  int1 = voc.intersection(sentiments_cloud.get('backstory'))
  int2 = voc.intersection(sentiments_cloud.get('Express Emotion'))
  int3 = voc.intersection(sentiments_cloud.get('Greeting'))
  int4 = voc.intersection(sentiments_cloud.get('Justification'))
  int5 = voc.intersection(sentiments_cloud.get('Rant'))
  int6 = voc.intersection(sentiments_cloud.get('Gratitude'))
  int7 = voc.intersection(sentiments_cloud.get('Other'))
  unigrams = set()
  for row in range(df.shape[0]):

    doc = df['clean_text'].values[row].split()

    for word in doc:

      if word in voc:

        if (word in int1) or (word in int2) or (word in int3) or (word in int4) or (word in int5) or (word in int6) or (word in int7):

          unigrams.add(word)

  return unigrams
  
unigram_feat_multi = get_unigram_features(vocab_multi, wordclouds, synset_dict_multi, X_train_multi)


def get_word_dict(unigram_feat_multi):
  puncs = [i for i in string.punctuation]
  unigram_feat_multi = puncs + list(unigram_feat_multi)
  dictionary_multi = list(unigram_feat_multi)

  word_index_multi= dict()

  for i in range(len(dictionary_multi)):
    if i==0:
       word_index_multi['OOV'] = 1
    else:
        word_index_multi[dictionary_multi[i]]=i+1
  return word_index_multi

word_ind_multi = get_word_dict(unigram_feat_multi)
"""
st.code(code_8, 'python')
st.write('\n')

st.header("Converting Text-to-Sequences")
st.markdown("Now we need to convert the text into a sequence since our model cannot understand words, it needs to see numbers. That's how models work. So let's give it what it needs.")

code_9 = """

from tensorflow.keras.preprocessing.sequence import pad_sequences

def find_max_len(df):
  length = []
  for i in tqdm(range(df.shape[0])):
    text = re.findall(r'[a-zA-Z]+', df['clean_text'].values[i])

    length.append(len(text))

  return length

def text_to_seq(vocab, data):

  sequences = []
  for i in tqdm(range(data.shape[0])):
    text = re.findall(r'[a-zA-Z]+|\~|\!|\@|\#|\$|\%|\^|\&|\*|\(|\)|\-\+|\,|\"|\?|\.|\:|\;|\=|\[|\]|\{|\}|\_|\\|\/', data['Text'].values[i])
    seq = []
    for i in text:
      if vocab.get(i)==None:
        seq.append(1)
      else:
        seq.append(vocab.get(i))
    sequences.append(seq)

  return np.array(sequences)

length_words_multi = find_max_len(X_train_multi)

maxlen = int(np.percentile(length_words_multi, 99))

maxlen = int(np.percentile(length_words_multi, 99))

train_array_multi = text_to_seq(word_index_multi, X_train_multi)
train_array_multi = pad_sequences(train_array_multi, maxlen=maxlen, dtype='int32', padding='pre',truncating='post')

cv_array_multi = text_to_seq(word_index_multi, X_cv_multi)
cv_array_multi = pad_sequences(cv_array_multi, maxlen=maxlen, dtype='int32', padding='pre',truncating='post')

test_array_multi = text_to_seq(word_index_multi, X_test_multi)
test_array_multi = pad_sequences(test_array_multi, maxlen=maxlen, dtype='int32', padding='pre',truncating='post')


"""
st.code(code_9, 'python')
st.write('\n')

st.header("Getting everything at one place")
st.markdown("Now that we have engineered all the required features, lets get everything in an array to make it ready for model building.")

code_10 = """
X_train_multi.drop('clean_text', axis=1, inplace=True)
X_cv_multi.drop('clean_text', axis=1, inplace=True)
X_test_multi.drop('clean_text', axis=1, inplace=True)
keep_columns = list(X_train_multi.columns[1:])

train_cols_multi = X_train_multi[keep_columns].values
cv_cols_multi = X_cv_multi[keep_columns].values
test_cols_multi = X_test_multi[keep_columns].values

train_multi = np.hstack((train_array_multi, train_tf.toarray(), train_cols_multi))
cv_multi = np.hstack((cv_array_multi, cv_tf.toarray(), cv_cols_multi))
test_multi = np.hstack((test_array_multi, test_tf.toarray(), test_cols_multi))

"""
st.code(code_10, 'python')
st.write('\n')

st.header("Building and evaluating ML models: ")
st.markdown(""" We will now build a ML model, to be precise a stacking classifier for our problem. In stacking classifier we build multiple models, and their output is given to a metaclassifier which will then output. This approach improves the model predictions, but with that we get flexibility of choosing which models we choose to stack.""")
st.subheader("Why stacking and why not a single classifier?")
st.markdown("""In my experiments, I got to know that single classifiers require MultiOutput Classifier to wrap around them to able to train and predcit. This method is not that flexible with regards to calibration and the model predictions are below average at the best. Models like Random Forest has multioutput inbuilt with them but even their predictions were below average.""")
st.markdown("""This is the reason why I chose Stacking Classifier per class. That way we will solving binary classification problem, and training and development process becomes much easier. Above all calibrating binary classifier is trivial and fast with new sklearn as it provides parallel processing. """)
st.subheader("""How to ingest data for stacking classifier? """)
st.markdown("""Well, we cannot give the entire data for stacking, so here's how we will split the data for stacking classifier. 
1. Split the data into 50-50 with stratified sampling.
2. Randomly sample the dataset such that we get equal indices in each of the set. For e.g. if we want to randomly sample 5 sets, then these 5 sets should have equal number of datapoints.
3. We will train these sets individually using five different models by calibrating them. These calibrated models are then used against the other 50% of the dataset and evaluated.
4. Say for example we are training 6 models, we will train 6 models with calibration, get its probability predictions and train a metaclassifier. This metaclassifier is also calibrated for probabilities.""")
st.subheader("Splitting the dataset")
st.markdown("Since we have too many features and we want to train faster at the same time get accurate results, we will used statistical methods like chi-squared test and selectkbest to reduce the number of features. This will calculate how our features and labels are independent of each other, with small values will translate to our labels being independent and large values will translate to feature being important for us as it will provide value to our model.")

code_11 = """
from sklearn.feature_selection import chi2, SelectKBest

select_k = SelectKBest(chi2, k=int(train_norm_multi.shape[1]/2))
select_k.fit(train_norm_multi, y_train_multi)

train_new = select_k.transform(train_norm_multi)

cv_new = select_k.transform(cv_norm_multi)

test_new = select_k.transform(test_norm_multi)


train_d1, train_d2, y_d1, y_d2 = train_test_split(train_new, y_train_multi, stratify=y_train_multi, test_size=0.5, random_state=42)

train_indices = []
for i in range(7):
  temp = random.sample(range(train_d1.shape[0]), k=train_d1.shape[0])
  train_indices.append(temp)
  
train_d1_k0 = np.take(train_d1, train_indices[0], axis=0)
train_d1_k1 = np.take(train_d1, train_indices[1], axis=0)
train_d1_k2 = np.take(train_d1, train_indices[2], axis=0)
train_d1_k3 = np.take(train_d1, train_indices[3], axis=0)
train_d1_k4 = np.take(train_d1, train_indices[4], axis=0)
train_d1_k5 = np.take(train_d1, train_indices[5], axis=0)
train_d1_k6 = np.take(train_d1, train_indices[6], axis=0)

y_d1_k0 = np.take(y_d1, train_indices[0], axis=0)
y_d1_k1 = np.take(y_d1, train_indices[1], axis=0)
y_d1_k2 = np.take(y_d1, train_indices[2], axis=0)
y_d1_k3 = np.take(y_d1, train_indices[3], axis=0)
y_d1_k4 = np.take(y_d1, train_indices[4], axis=0)
y_d1_k5 = np.take(y_d1, train_indices[5], axis=0)
y_d1_k6 = np.take(y_d1, train_indices[6], axis=0)
"""
st.code(code_11, 'python')
st.write('\n')

st.write('insert flowchart here')


st.subheader(" Training different Models with each of the classes ")
st.markdown("It is understood that each dataset that we have now will have 7 sentiments. So training each model for each of those classes and then doing the same for all datasets is very cumbersome. What we chose to do instead is from each dataset we will choose only majority class i.e. Class which has higher number of variables.")
st.markdown(" We will create stacking classifier for each of the 7 sentiments and then predict for each sentiment using these stacking classifier. ")
st.markdown("Each of our stacking classifier will contain a Decision Tree, Stochastic GRadient Descent, Logistic Regression, Naive Bayes and Logistic Regression as the Meta-Classifier")

code_12 = """
dtc = DecisionTreeClassifier()

random_grid = {'splitter': ['best', 'random'],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'criterion': ['gini', 'entropy']}

dtc_random = RandomizedSearchCV(estimator = dtc, param_distributions = random_grid, scoring='roc_auc',
                               n_iter = 50, cv = 3,refit=True, verbose=3, random_state=42, n_jobs = -1)
                               
                               
### OUTPUT
Fitting 3 folds for each of 50 candidates, totalling 150 fits
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=-1)]: Done  24 tasks      | elapsed:    1.6s
[Parallel(n_jobs=-1)]: Done 143 out of 150 | elapsed:    3.7s remaining:    0.2s
[Parallel(n_jobs=-1)]: Done 150 out of 150 | elapsed:    3.7s finished
RandomizedSearchCV(cv=3, error_score=nan,
                   estimator=DecisionTreeClassifier(ccp_alpha=0.0,
                                                    class_weight=None,
                                                    criterion='gini',
                                                    max_depth=None,
                                                    max_features=None,
                                                    max_leaf_nodes=None,
                                                    min_impurity_decrease=0.0,
                                                    min_impurity_split=None,
                                                    min_samples_leaf=1,
                                                    min_samples_split=2,
                                                    min_weight_fraction_leaf=0.0,
                                                    presort='deprecated',
                                                    random_state=None,
                                                    splitter='best'),
                   iid='deprecated', n_iter=50, n_jobs=-1,
                   param_distributions={'criterion': ['gini', 'entropy'],
                                        'max_features': ['auto', 'sqrt'],
                                        'min_samples_leaf': [1, 2, 4],
                                        'min_samples_split': [2, 5, 10],
                                        'splitter': ['best', 'random']},
                   pre_dispatch='2*n_jobs', random_state=42, refit=True,
                   return_train_score=False, scoring='roc_auc', verbose=3)
"""

st.code(code_12, 'python')






