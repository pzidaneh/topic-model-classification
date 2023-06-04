from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statistics import mean

from . import custom_timer
from . import exploration


def show_topics(model, topn=10, prob=False):
    topics_df = pd.DataFrame(model.show_topics(num_topics=-1, num_words=topn, formatted=False))[1]

    if prob:
        output = topics_df.apply(dict).apply(pd.Series).fillna(0)
    else:
        output = topics_df.apply(lambda x: [i[0] for i in x])

    return output


def generate_gensim_object(corpus, corpus_ref=None):
    if corpus_ref is None:
        common_texts = corpus.apply(lambda doc: str(doc).split())
        common_dictionary = Dictionary(common_texts)
    else:
        common_dictionary = Dictionary(corpus_ref.apply(lambda doc: str(doc).split()))
    
    common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    
    #return common_texts, common_dictionary, common_corpus
    return common_dictionary, common_corpus


def compute_coherence_values(corpus, corpus_name=str(), start=2, stop=20, step=1, update_every=1000, coherence_repeat=5, topn=10, random_state=55):
    """Compute c_v coherence for various number of topics
    
    source (with adjustment):
    https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#14computemodelperplexityandcoherencescore    
    """
    common_dictionary, common_corpus = generate_gensim_object(corpus)
    common_texts = corpus.apply(lambda doc: str(doc).split())

    if common_dictionary != str():
        common_dictionary.save(f"{corpus_name}_dictionary.pkl")
    
    coherence_value_individual = []
    coherence_value_per_topic = []
    coherence_values = []
    model_list = []
    topics_num = []
    
    corpus_size = len(corpus)
    update_every = update_every
    chunksize = round(corpus_size/update_every*5)
    
    print(corpus.head(5))
    print(f"corpus_size {corpus_size}, chunksize {chunksize}, update_every {update_every}")
    print(f"Start at {custom_timer.time_now(asc=True)} & will end after {stop}")
    
    coherence_max = 0
    enum = 1
    for num_topics in range(start, stop + 1, step):
        time_start = custom_timer.time_now()
        
        model = LdaModel(
            corpus=common_corpus, id2word=common_dictionary, num_topics=num_topics,
            random_state=random_state, chunksize=chunksize, passes=1,
            update_every=update_every, per_word_topics=False
        )
        
        print(f"({num_topics})".rjust(3), end="")
        
        coherence_value_per_topic_temp = []
        for coherence_calculation in range(coherence_repeat):
            CM = CoherenceModel(
                model=model, texts=common_texts, dictionary=common_dictionary, coherence='c_v',
                topn=topn
            )
            coherence_value_individual.append(CM.get_coherence())
            coherence_value_per_topic_temp.append(CM.get_coherence_per_topic())
            
            print(".", end="")
        

        coherence_value_individual_aggregate = np.mean(coherence_value_individual).item()
        
        if (len(coherence_values) == 0) or (coherence_value_individual_aggregate > coherence_max):
            lda_best = model
            coherence_max = coherence_value_individual_aggregate
            coherence_value_per_topic = np.array(coherence_value_per_topic_temp).mean(axis=0)
            coherence_value_per_topic = list(coherence_value_per_topic)
        else:
            pass
        
        topics_num.append(num_topics)
        coherence_values.append(coherence_value_individual_aggregate)
        coherence_value_individual = []
        
        if enum % 5:
            #end = "\t"
            end = " "
        else:
            end = "\n"
            
        #print("Done.", end=end)
        print(f"{custom_timer.track_time(time_start)}", end=end)
        enum += 1
    
    
    #x = topics_num
    #print(coherence_values)
    ax = plt.figure().gca()
    
    ax.set_title(corpus_name)
    ax.set_xlabel("Number of topics")
    ax.set_ylabel("Coherence score")

    ax.plot(topics_num, coherence_values)
    ax.set_ylim(bottom=0, top=1)
    ax.xaxis.get_major_locator().set_params(integer=True)
    yticks = [i/100 for i in range(0, 100+1, 20)]
    ax.set_yticks(ticks=yticks)

    ax.axvline(x=topics_num[np.argmax(coherence_values)], color="black",
                linestyle="dashed", linewidth=.75)
    ax.axhline(y=mean(coherence_value_per_topic), color="black",
                linestyle="dashed", linewidth=.75)
    
    if corpus_name != str():
        plt.savefig(f'{corpus_name}.png')
    plt.show();
    
    print(f"Best number of topics: {topics_num[np.argmax(coherence_values)]}")
    print(f"per topic: {coherence_value_per_topic}")
    print(f"average: {mean(coherence_value_per_topic)}")

    return topics_num, lda_best, coherence_values, coherence_value_per_topic


def fillna_prob(prob_series):
    needs_filling = prob_series.isna().sum().item()
    
    if needs_filling == 0:
        return prob_series

    sum_not_nan = prob_series.dropna().sum().item()
    fill_value = (1 - sum_not_nan)/needs_filling

    return prob_series.fillna(fill_value)


def infer_topics(lda_model, corpus):
    topics_prob_list = []
    common_corpus = generate_gensim_object(corpus)[1]
    len_corpus = np.array(corpus).flatten().shape[0]
    
    
    for i in range(len_corpus):
        topics_prob = lda_model.get_document_topics(common_corpus[i], minimum_probability=None)
        topics_prob_list.append(dict(topics_prob))
    
    
    topics_prob_df = pd.DataFrame(topics_prob_list)
    topics_prob_df = topics_prob_df.apply(fillna_prob, axis=1)
    topic_series = topics_prob_df.idxmax(axis=1)

    return topics_prob_df, topic_series


def cross_topic_label(topic, label, save_name=str()):
    output = pd.crosstab(label, topic)
    
    topic_summary = topic.value_counts()#.sort_index()
    #topic_summary = pd.DataFrame(topic_summary).transpose()
    
    #height = 5*((len(np.unique(label)) - 1)//2 + 1)
    #width = 20*((len(np.unique(topic)) - 1)//6 + 1)

    width  =  5*len(np.unique(topic))
    height = 20*len(np.unique(label))

    print(f"width, height = {width}, {height}")
    #fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 40))
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(width, height))
    
    # Distribution of topics
    #sns.heatmap(topic_summary, annot=True, cmap="Greys", cbar=False, square=1, linewidth=1., fmt='g', ax=axes[0])
    axes[0].pie(topic_summary, labels=topic_summary.index, autopct=lambda pct: exploration.labelFormatting(pct, topic_summary))

    # Cross tabulation
    sns.heatmap(output, annot=True, cmap="Greys", cbar=False, square=1, linewidth=1., fmt='g', ax=axes[1])
    
    # per clust
    output_norm = output.apply(lambda x: x/x.sum(), axis=0)*100
    sns.heatmap(output_norm, annot=True, fmt = '.2f', cmap="Greys", cbar=False, square=1, linewidth=1., ax=axes[2])
    
    # per class
    output_norm = output.apply(lambda x: x/x.sum(), axis=1)*100
    sns.heatmap(output_norm, annot=True, fmt = '.2f', cmap="Greys", cbar=False, square=1, linewidth=1., ax=axes[3])
    
    axes[0].set_xlabel("Topic")
    axes[0].set_ylabel("")
    
    for i in [1, 2, 3]:
        axes[i].set_xlabel("Topic")
        axes[i].set_ylabel("Class")
    
    for i in [2, 3]:
        for t in axes[i].texts:
            t.set_text(t.get_text() + " %")
    
    if save_name != str():
        plt.savefig(f'{save_name}_crosstab.png')
    plt.show()


def topic_model_fit(X_train, save_name=str(), start=2, stop=20, step=1, update_every=1000, coherence_repeat=5, topn="auto", topn_threshold=10, random_state=55):
    #calculate_coherence()  # save to csv
    #plot_coherence()  # save to png
    #explore_topics()  # save to png

    if topn == "auto":
        topn = int(X_train.apply(lambda doc: len(str(doc).split())).median())
        if topn > topn_threshold:
            topn = topn_threshold

    topics_num, lda_best, coherence_values, coherence_value_per_topic = compute_coherence_values(corpus=X_train, corpus_name=save_name, update_every=update_every, start=start, stop=stop, step=step, coherence_repeat=coherence_repeat, topn=topn, random_state=random_state)
    coherence_value_per_topic = list(np.round(coherence_value_per_topic, 2))

    print(f"Coherence Value: topn = {topn}")
    wordCloudTopic(lda_best, suffix_list=coherence_value_per_topic, topn=topn*2, save_name=f"{save_name}_topic", show_title=False)
    #cross_topic_label(topic=topic_series, label=y, save_name=save_name)
    if save_name != str():
        lda_best.save(f"{save_name}_lda.pkl")
    
    return lda_best


def topic_model_transform(lda_model, X, y, visualize=False, save_name=str()):
    # To be implemented SEPARATELY on train AND test    
    topics_df, topic_series = infer_topics(lda_model, X)
    if visualize:
        cross_topic_label(topic=topic_series, label=y, save_name=save_name)

    return topics_df


def wordCloudTopic(lda_model, suffix_list=[], topn=10, save_name=str(), show_title=True):
    summary_topics = (show_topics(lda_model, topn=topn, prob=True)*1000).round().astype(int)
    summary_topics = summary_topics.apply(lambda col: [[col.name]*value for value in col], axis=0)
    summary_topics = summary_topics.apply(lambda row: [value for value in row], axis=1)
    summary_topics = summary_topics.apply(lambda l: " ".join(list(chain.from_iterable(l)))).reset_index()
    #print(summary_topics)
    #print(summary_topics[0].apply(lambda x: x.split()).explode().value_counts())

    if save_name != str():
        save = True
    else:
        save = False

    exploration.wordCloudStratified(corpus=summary_topics[0], label=summary_topics['index'], suffix_list=suffix_list, name=save_name, save=save, show_title=show_title)


# END