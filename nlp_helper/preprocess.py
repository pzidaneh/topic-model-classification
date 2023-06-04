from sklearn.model_selection import RepeatedStratifiedKFold

def cross_split(X, y, n_splits=5, random_state=55):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=random_state)
    rsfk_split = rskf.split(X=X, y=y)
    cv_index_dict = dict()

    for enum, (train_index, test_index) in enumerate(rsfk_split):
        cv_index_dict[f"train_{enum}"] = train_index
        cv_index_dict[f"test_{enum}"] = test_index

    return cv_index_dict


from hunspell import Hunspell
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
from string import punctuation

import unicodedata

from . import custom_timer


class Cleaner:
    def __init__(self, slang, slang_replacement, emoji=[None], emoji_replacement=[None], punctuation=punctuation, stopwords=stopwords.words("indonesian")):
        self.slang_dict = dict(zip(slang, slang_replacement))
        
        if (emoji == [None]) ^ (emoji_replacement == [None]):  # ^ is XOR logic gate
            raise Exception("emoji and emoji_replacement must BOTH be None or not None")
        else:
            self.emoji_dict = dict(zip(emoji, emoji_replacement))
        
        self.punctuation = punctuation
        self.stopwords = stopwords
        self.stopwords_selection = ["enggak", "tidak"]
        self.stopwords_selected = self.omit_substr(stopwords, self.stopwords_selection)

        path_data = "C:/Users/LENOVO/Documents/00 IPB/Tugas Akhir/Data/"
        self.HS = Hunspell(path_data + "id_ID")


    def omit_substr(self, input_list, ref_list):
        output = np.array(input_list)
        
        for substr in ref_list:
            output = output[~(np.char.find(output, substr) > -1)]
        
        return list(output)


    def strip_accents(self, doc):
        doc_clean = []

        for char in unicodedata.normalize('NFD', doc):
            if unicodedata.category(char) != 'Mn':
                doc_clean.append(char)

        return ''.join(doc_clean)


    def hunspell_stemmer(self, word):
        word_temp = str(word)
        word = str()
        other_char = str()

        #if word.isalnum():
        if word.isalpha():
            word = word_temp
        else:
            for char in word_temp:
                #if char.isalnum():
                if char.isalpha():
                    word += char
                elif char.isnumeric():  # remove numbers
                    pass
                else:
                    other_char += char

            other_char_temp = other_char
            other_char = str()
            
            for char in other_char_temp:
                other_char += " "
                other_char += self.emoji_dict.get(char, char)


        try:
            stems = self.HS.stem(word)
        except UnicodeEncodeError:
            try:
                stems = self.HS.stem(self.strip_accents(word))
            except UnicodeEncodeError:
                stems = [word]

        if len(stems) == 0:
            output = word
        else:
            output = stems[0]

        return output + other_char


    def document_wise_cleaning(self, doc):
        doc_clean = doc

        doc_clean = str(doc_clean).lower()
        doc_clean = str(doc_clean).replace("\n", " ")
        doc_clean = re.sub(r"\s+", " ", str(doc_clean))
        doc_clean = str(doc_clean).translate(str.maketrans("", "", self.punctuation))

        return doc_clean


    def strip_excess_characters(self, tokens):
        tokens_clean = tokens

        tokens_clean = np.vectorize(lambda word: re.sub(r'^([a-z])\1+', r'\1', str(word)))(tokens_clean)
        tokens_clean = np.vectorize(lambda word: re.sub(r'([a-z])\1+$', r'\1', str(word)))(tokens_clean)

        return tokens_clean


    def remove_stopwords(self, tokens):
        tokens_clean_class = [word for word in tokens if word not in self.stopwords_selected]
        tokens_clean_clust = [word for word in tokens_clean_class if word not in self.stopwords_selection]

        return tokens_clean_clust, tokens_clean_class


    def word_wise_cleaning(self, doc):
        tokens = doc.split()
        
        if tokens == list():
            tokens = [""]

        tokens_clean = tokens

        try:
            tokens_clean = self.strip_excess_characters(tokens_clean)
            tokens_clean = np.vectorize(lambda word: self.slang_dict.get(word, word))(tokens_clean)
            tokens_clean = np.vectorize(lambda word: self.emoji_dict.get(word, self.hunspell_stemmer(word)))(tokens_clean)
            
            
            tokens_clean_clust, tokens_clean_class = self.remove_stopwords(tokens_clean)

        except ValueError:
            print(f"ERROR AT: {doc}")
            
            tokens_clean_clust = np.array([""])
            tokens_clean_class = np.array([""])

        doc_clean_clust =  " ".join(tokens_clean_clust)
        doc_clean_class =  " ".join(tokens_clean_class)

        return doc_clean_clust, doc_clean_class


    def corpus_wise_cleaning(self, X):
        X_tokenized = pd.Series(X).copy().apply(lambda doc: str(doc).split())
        token_counts = X_tokenized.explode().value_counts()
        
        threshold = 1
        rare_words = token_counts.index[token_counts <= threshold]
        rare_words = rare_words.to_series().reset_index(drop=True).to_list()
        
        X_filtered = X_tokenized.apply(lambda doc: [word for word in doc if word not in rare_words])
        X_clean = np.vectorize(lambda doc: " ".join(doc))(X_filtered)
        X_clean = pd.Series(X_clean)

        return X_clean, rare_words


    def adjust_placement(self, char, ref):
        return str(char).rjust(len(str(ref)), " ")


    def print_tracker(self, count, limit, start_time, force=False, end=None):
        if (count % 1000 == 0) or force:
            if count % 4000 == 0:
                end = "\n"
            elif end is None:
                end = "\t"
            print(f"[{self.adjust_placement(count, limit)}] = {custom_timer.track_time(start_time)}", end=end)


    def removeNewsMark_perDoc(self, doc):
        if len(doc) > 2:
            output = " ".join(doc.split()[2:])
        else:
            output = ""

        return output

    def removeNewsMark(self, corpus, threshold=.001, save_name=str()):
        firstOneWord = pd.Series(corpus).apply(lambda doc: doc.split()[0])
        firstOneWord_vc = firstOneWord.value_counts()
        firstOneWord_vc_perc = firstOneWord_vc/firstOneWord_vc.sum()

        needCorrection_boolArray = np.isin(firstOneWord, firstOneWord_vc.index[(firstOneWord_vc/firstOneWord_vc.sum() > threshold)])
        corpus_clean = corpus.loc[needCorrection_boolArray].apply(self.removeNewsMark_perDoc)

        if save_name != str():
            firstOneWord_df = pd.concat({
                'word': firstOneWord_vc.index.to_series(),
                'count': firstOneWord_vc,
                'percent': firstOneWord_vc_perc*100
            }, axis=1)
            firstOneWord_df.to_csv(f"{save_name}_news_mark.csv", index=False)

        return corpus_clean

    def clean(self, X, save_name=str(), do_print=True, remove_news_mark=False):
        start_time = custom_timer.time_now()
        X = pd.Series(X)

        limit = len(X)
        word_count = X.apply(lambda doc: len(str(doc).split()))
        
        if do_print:
            print(f"{save_name} -> {limit} rows\n(mean: {word_count.mean().round(2)} words; median: {word_count.median().round(2)} words)")

        X_clean_clust = []
        X_clean_class = []


        for count, doc in enumerate(X, start=1):
            doc_clean = self.document_wise_cleaning(doc)
            doc_clean_clust, doc_clean_class = self.word_wise_cleaning(doc_clean)

            X_clean_clust.append(doc_clean_clust)
            X_clean_class.append(doc_clean_class)

            if do_print:
                self.print_tracker(count, limit, start_time)
            

        if do_print:
            print()
            print("end doc-wise cleaning")
            self.print_tracker(limit, limit, start_time, force=True, end="\n\n")
            print("start remove rare words")

        X_clean_clust, rare_words = self.corpus_wise_cleaning(X_clean_clust)
        X_clean_class = pd.Series(X_clean_class)

        if remove_news_mark:
            print("Remove byline & dateline ... ", end="")
            X_clean_clust = self.removeNewsMark(X_clean_clust)
            X_clean_class = self.removeNewsMark(X_clean_class)
            print("success!")

        if do_print:
            print(rare_words)
            print()
            self.print_tracker(limit, limit, start_time, force=True, end="\n")

        return X_clean_clust, X_clean_class


    def clean_save(self, X, y, save_name, do_print=True, remove_news_mark=False):
        X_clean_clust, X_clean_class = self.clean(X, save_name=save_name, do_print=do_print, remove_news_mark=remove_news_mark)
        df_clean = pd.concat({"TEXT_raw": X, "TEXT_clust": X_clean_clust, "TEXT_class": X_clean_class, "LABEL": y}, axis=1)

        df_clean_noDuplicates = df_clean.drop_duplicates(subset=["TEXT_clust", "LABEL"]).dropna()

        print(f"Duplicates removed. Before: {df_clean.shape[0]}. After: {df_clean_noDuplicates.shape[0]}")

        clean_save_name = f"{save_name}_clean.csv"
        df_clean_noDuplicates.to_csv(clean_save_name, index=False)
        print(clean_save_name)

        return df_clean_noDuplicates['TEXT_clust'], df_clean_noDuplicates['TEXT_class']


# END