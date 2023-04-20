#dataset_table = specific_sa_in_ne
#sa_in_ne2 = sa_in_noteevents
#new_table = 
import pandas as pd
import numpy as np
import os
import spacy
from tqdm import tqdm
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import CountVectorizer
#spacy.cli.download("en_core_web_sm")
# 
nlp = spacy.load("en_core_web_sm")

phrase_matcher = PhraseMatcher(nlp.vocab)
phrases = ['sleep apnea', 'OSA','slp apnea', 'SLEEP APNEA', 'Sleep apnea', 'Sleep Apnea']
patterns = [nlp(text) for text in phrases]
phrase_matcher.add('AI', None, *patterns)

    
def sleep_apnea_matcher(text):
    matches = []
    doc = nlp(text)
    for sent in doc.sents:
        for match_id, start, end in phrase_matcher(nlp(sent.text)):
            if nlp.vocab.strings[match_id] in ["AI"]:
                matches.append(sent.text)
    return matches

def read_patients(path):
    patients = pd.read_csv(path)
    print("Patients")
    print(patients.head())
    print(len(patients["SUBJECT_ID"].unique()))
    return patients

def read_shortandlongtitle(path):
    shortandlongtitle_diagnoses = pd.read_csv(path)
    print("Short and long title diagnoses")
    print(shortandlongtitle_diagnoses.head())
    print(len(shortandlongtitle_diagnoses))
    return shortandlongtitle_diagnoses

def read_matchingdiagnoses(path):
    matching_diagnoses = pd.read_csv(path)
    print("Matching diagnoses")
    print(matching_diagnoses.head())
    print(len(matching_diagnoses))
    return matching_diagnoses

def find_sainlongtitle(path):
    sa_longtitle =     shortandlongtitle_diagnoses[shortandlongtitle_diagnoses["LONG_TITLE"].str.lower().str.contains(path)]
    print("Sleep apnea in long title")
    print(sa_longtitle.head())
    print(len(sa_longtitle))
    return sa_longtitle

def find_saintitleandcode(path):
    sa_in_titleandcode = matching_diagnoses.merge(sa_longtitle, on = "ICD9_CODE", how = path)
    print("Sleep apnea in long title text and diagnoses code")
    print(sa_in_titleandcode.head())
    print(len(sa_in_titleandcode["SUBJECT_ID"].unique()))
    return sa_in_titleandcode

# 

def read_noteevents_unigram(path):
    noteevents = pd.read_csv(path)
    print("Note events")
    # print(noteevents.head(10))
    print(len(noteevents))
    text = noteevents['TEXT'].head(5)
    # model = CountVectorizer(ngram_range = (1, 1), max_features = 100, stop_words='english')
    # CountVectorizer = CountVectorizer(lowercase=False)
    model = CountVectorizer(ngram_range = (1, 1), stop_words='english', lowercase=True)
    matrix = model.fit_transform(text).toarray()
    noteevents_output = pd.DataFrame(data = matrix, columns = model.get_feature_names())
    ne_sum = pd.DataFrame(noteevents_output)
    sum_column = ne_sum.sum(axis=0)
    # sum_column.columns=['unigram','nb_times']
    print (sum_column)
    print(sum_column.shape)
    sum_column.to_csv("~/workspace/osa_supervised_ml/data/intermediate_data_files/unigram_ne1(head5).csv", index=True)
    # neo_list = noteevents_output.columns[1:2083181]
    # noteevents_output['nb_times'] = noteevents_output[neo_list].sum(axis=1)
    # print(noteevents_output.T.tail(5))
    # print(noteevents_output.shape)
    return noteevents

def read_noteevents_bigram(path):
    noteevents = pd.read_csv(path)
    print("Note events")
    text = noteevents['TEXT'][1011595:2023191]
    model = CountVectorizer(ngram_range = (2, 2), stop_words='english', min_df = 5, lowercase=True)
    # matrix = model.fit_transform(text).toarray()
    # noteevents_output = pd.DataFrame(data = matrix, columns = model.get_feature_names())
    # ne_sum = pd.DataFrame(noteevents_output)
    # sum_column = ne_sum.sum(axis=0)
    # sum_column.columns=['bigram','nb_times']
    # print (sum_column)
    # print(sum_column.shape)
    # sum_column.to_csv("~/workspace/osa_supervised_ml/data/intermediate_data_files/bigram_ne1(head5).csv", index=True)
    # print(noteevents_output.shape)
    # return noteevents
    counts = np.array(model.fit_transform(text).sum(axis=0))[0]
    bigrams = model.get_feature_names_out()
    print(counts)
    print(bigrams)
    data = [[counts[i],bigrams[i]] for i in range(len(bigrams))]
    noteevents_output = pd.DataFrame(data, columns = ["counts", "bigrams"])
    print(noteevents_output)
    print(noteevents_output.shape)
    noteevents_output = noteevents_output.sort_values('counts', ascending = False) 
    print (noteevents_output)
    print(noteevents_output.shape)
    noteevents_output.to_csv("~/workspace/osa_supervised_ml/data/intermediate_data_files/bigram_ne1(part2of2)ordered.csv", index=True)
    print(noteevents_output.shape)
    return noteevents

def read_noteevents_trigram(path):
    noteevents = pd.read_csv(path)
    print("Note events")
    text = noteevents['TEXT'][1820871:2023191]
    model = CountVectorizer(ngram_range = (3, 3), stop_words='english', min_df = 5, lowercase=True)
    # matrix = model.fit_transform(text).toarray()
    # noteevents_output = pd.DataFrame(data = matrix, columns = model.get_feature_names())
    # ne_sum = pd.DataFrame(noteevents_output)
    # sum_column = ne_sum.sum(axis=0)
    # sum_column.columns=['trigram','nb_times']
    # print (sum_column)
    # print(sum_column.shape)
    # sum_column.to_csv("~/workspace/osa_supervised_ml/data/intermediate_data_files/trigram_ne1(head5).csv", index=True)
    # print(noteevents_output.shape)
    # return noteevents
    counts = np.array(model.fit_transform(text).sum(axis=0))[0]
    trigrams = model.get_feature_names_out()
    print(counts)
    print(trigrams)
    data = [[counts[i],trigrams[i]] for i in range(len(trigrams))]
    noteevents_output = pd.DataFrame(data, columns = ["counts", "trigrams"])
    print(noteevents_output)
    print(noteevents_output.shape)
    noteevents_output = noteevents_output.sort_values('counts', ascending = False) 
    print (noteevents_output)
    print(noteevents_output.shape)
    noteevents_output.to_csv("~/workspace/osa_supervised_ml/data/intermediate_data_files/trigram_ne1(part10)ordered.csv", index=True)
    print(noteevents_output.shape)
    return noteevents

def read_noteevents_tetragram(path):
    noteevents = pd.read_csv(path)
    print("Note events")
    text = noteevents['TEXT'][1820871:2023191]
    model = CountVectorizer(ngram_range = (4, 4), stop_words='english', min_df = 5, lowercase=True)
    counts = np.array(model.fit_transform(text).sum(axis=0))[0]
    tetragrams = model.get_feature_names_out()
    print(counts)
    print(tetragrams)
    data = [[counts[i],tetragrams[i]] for i in range(len(tetragrams))]
    noteevents_output = pd.DataFrame(data, columns = ["counts", "tetragrams"])
    print(noteevents_output)
    print(noteevents_output.shape)
    noteevents_output = noteevents_output.sort_values('counts', ascending = False) 
    print (noteevents_output)
    print(noteevents_output.shape)
    noteevents_output.to_csv("~/workspace/osa_supervised_ml/data/intermediate_data_files/tetragram_ne1(part10)ordered.csv", index=True)
    print(noteevents_output.shape)
    return noteevents

# def read_noteevents_tetragram_ordered(path):
#     noteevents = pd.read_csv(path)
#     print("Note events")
#     text = noteevents['tetragram']
#     sum_column = sum_column.sort_values(by='nb_times')
#     print (sum_column)
#     print(sum_column.shape)
#     sum_column.to_csv("~/workspace/osa_supervised_ml/data/intermediate_data_files/tetragram_ne1(head5)ordered.csv", index=True)
#     print(noteevents_output.shape)
#     return noteevents

def find_sainnoteevents(path):
    sa_in_noteevents = noteevents[noteevents["TEXT"].str.lower().str.contains(path)]
    print("Sleep Apnea in Note events")
    print(sa_in_noteevents.head())
    print(len(sa_in_noteevents))
    return sa_in_noteevents

def print_stats():
    print("Here are stats")
    print("Unique patients that were assigned sleep apnea in diagnosis codes and long title diagnosis")
    print(len(sa_in_titleandcode["SUBJECT_ID"].unique()))
    print("Unique hospital admissions of patients that were assigned sleep apnea in diagnosis codes and long title diagnosis")
    print(len(sa_in_titleandcode["HADM_ID"].unique()))
    print("Unique patients that had sleep apnea mentioned in the text of their note events")
    print(len(sa_in_noteevents["SUBJECT_ID"].unique()))
    print("Unique hospital admissions that had sleep apnea mentioned in the text of their note events")
    print(len(sa_in_noteevents["HADM_ID"].unique()))
    
def find_specificsainne(sa_in_noteevents):
    specific_sa_in_ne = sa_in_noteevents[["SUBJECT_ID","HADM_ID","ROW_ID", "TEXT"]]
    
    specific_sa_in_ne.columns = ['SUBJECT_ID', 'HADM_ID', 'ROW_ID', 'OSA_MENTION']
    
    # specific_sa_in_ne["OSA_MENTION"]  = specific_sa_in_ne["OSA_MENTION"].apply(sleep_apnea_matcher)
    matches = []
    for x in tqdm(specific_sa_in_ne["OSA_MENTION"]):
        matches.append(sleep_apnea_matcher(x))
    specific_sa_in_ne["OSA_MENTION"] = matches
    
    specific_sa_in_ne = specific_sa_in_ne.explode("OSA_MENTION", ignore_index=True)
    specific_sa_in_ne = specific_sa_in_ne.reset_index().rename(columns={"index" : "SNIPPET_ID"})
    print("OSA Mentions in Specific Columns table with applied SA Matcher function")
    print(specific_sa_in_ne.head())
    print(len(specific_sa_in_ne["OSA_MENTION"].unique()))
    return specific_sa_in_ne

# comment out below for original

# def read_osamention(path):
#     osa_mention = pd.read_csv(path)
#     print("OSA Mention")
#     print(osa_mention.head(10))
#     print(len(osa_mention["SUBJECT_ID"].unique()))
#     text = osa_mention['OSA_MENTION']
#     model = CountVectorizer(ngram_range = (1, 1))
#     matrix = model.fit_transform(text).toarray()
#     osa_mention_output = pd.DataFrame(data = matrix, columns = model.get_feature_names())
#     print(osa_mention_output.T.tail(5))
#     return osa_mention


if __name__ == "__main__":
    
#     patients = read_patients("~/workspace/osa_supervised_ml/data/raw/PATIENTS.csv") 
    
#     shortandlongtitle_diagnoses = read_shortandlongtitle("~/workspace/osa_supervised_ml/data/raw/D_ICD_DIAGNOSES.csv")
    
#     matching_diagnoses = read_matchingdiagnoses("~/workspace/osa_supervised_ml/data/raw/DIAGNOSES_ICD.csv")
    
#     sa_longtitle = find_sainlongtitle("sleep apnea")
    
    # noteevents = read_noteevents_unigram("~/workspace/osa_supervised_ml/data/raw/NOTEEVENTS.csv")
    
     noteevents = read_noteevents_bigram("~/workspace/osa_supervised_ml/data/raw/NOTEEVENTS.csv")
    
    # noteevents = read_noteevents_trigram("~/workspace/osa_supervised_ml/data/raw/NOTEEVENTS.csv")
    
    # noteevents = read_noteevents_tetragram("~/workspace/osa_supervised_ml/data/raw/NOTEEVENTS.csv")
    
#     sa_in_titleandcode = find_saintitleandcode("inner")
    
#     sa_in_noteevents = find_sainnoteevents("sleep apnea")
    
#     print_stats()
    
#     specific_sa_in_ne = find_specificsainne(sa_in_noteevents)
    #specific_sa_in_ne.to_csv("~/workspace/osa_supervised_ml/data/intermediate_data_files/OSA_inputtableof10(OSA_MENTION)6.csv", index=False)
    
#     osa_mention = read_osamention("~/workspace/osa_supervised_ml/data/intermediate_data_files/OSA_inputtableof10(OSA_MENTION)6.csv")
    # noteevents.to_csv("~/workspace/osa_supervised_ml/data/intermediate_data_files/unigram_ne1(head1).csv", index=True)

    