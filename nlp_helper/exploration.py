import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import ceil
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


def get_row_col(index, col_max):
    # row & col starts from 0
    # row_max & col_max starts from 1
    row = index//col_max
    col = index%col_max
    
    return row, col


def wordCloudStratified(corpus, label, suffix_list=[], color_code=None, name=str(), save=False, show=True, show_title=True, ngram_range=(1, 1), top_adjustment=1.25, random_state=55):
    groups = label.unique()
    num_of_groups = len(groups)
    
    if num_of_groups > 6:
        col_max = 4
        row_max = ceil(num_of_groups/col_max)
    else:
        col_max = num_of_groups
        row_max = 1
    
    height = 80*((row_max-1)//2 + 1)
    width = 100*((col_max - 1)//6 + 1)

    print(f"width, height = {width}, {height}")
    #fig, axes = plt.subplots(row_max, col_max, figsize=(80, 100))
    fig, axes = plt.subplots(row_max, col_max, figsize=(width, height))
    index = 0
    
    if suffix_list == []:
        suffix_list = [""]*num_of_groups
    if color_code is None:
        color_code = ["Greys"]*len(groups)
    
    groups = np.sort(groups)
    
    for group_name, suffix, color_name in zip(groups, suffix_list, color_code):
        corpus_in_group = corpus.loc[label == group_name]
        
        try:
            CountVec = CountVectorizer(ngram_range=ngram_range, lowercase=False)
            CountVecFitted = CountVec.fit_transform(corpus_in_group)
            freqs = pd.DataFrame(CountVecFitted.toarray(), columns=CountVec.get_feature_names_out())
            #print(freqs)
            freqs = freqs.sum().to_dict()
        except ValueError as e:
            freqs = {"": 1}
            print(e)
        except TypeError as e:
            freqs = {"": 1}
            print(e)
        
        wc = WordCloud(
            colormap=color_name, width=400, height=400, mode='RGBA',
            background_color='white', max_words=300,# min_word_length=1,
            prefer_horizontal=1, random_state=random_state,
            collocations=False
        )
        wc = wc.generate_from_frequencies(freqs)
        
        if row_max == 1:
            ax = axes[index]
        else:
            row, col = get_row_col(index, col_max)
            ax = axes[row, col]
        
        if suffix == str():
            ax.set_title(group_name, fontsize=150)
        else:
            ax.set_title(f"{group_name} [{suffix}]", fontsize=150)

        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        ax.margins(x=0, y=0)
        index += 1
    
    
    if name is None:
        name = corpus.name
    else:
        name = f"{name} ({corpus.name})"

    suptitle = f"{name} (n-gram {str(ngram_range)[1:-1].replace(', ', '-')})"

    if show_title:
        fig.suptitle(suptitle, fontsize=150)
        fig.subplots_adjust(top=top_adjustment)
    
    if save:
        save_name = f"{suptitle}.png"
        plt.savefig(save_name)
        print(save_name)

    if show:
        plt.show()


#wordCloudStratified(corpus=df_review['X_raw'], name="df_review", stratifier=df_review['y'],
#                    color_code=["Reds", "Blues"], save=False)


def describe_corpus(corpus):
    if type(corpus) != pd.Series:
        return "Corpus must be of type pd.Series"

    len_chars = corpus.apply(len)
    len_words = corpus.apply(lambda doc: len(doc.split()))
    
    output = pd.DataFrame({
        "CHARACTERS": len_chars.describe(),
        "WORDS": len_words.describe()
    })
    
    output.loc["skew"] = [len_chars.skew(), len_words.skew()]
    output = output.round(2)
    
    return output


def plotCorpus():
    pass


def labelFormatting(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return f"{absolute:d}\n({pct:.1f}%)"


def plotLabel(y, ax=None):
    y = pd.Series(y).value_counts()

    if ax is None:
        plt.pie(y, labels=y.index, autopct=lambda pct: labelFormatting(pct, y))
        plt.show()
    else:
        ax.pie(y, labels=y.index, autopct=lambda pct: labelFormatting(pct, y))


# END