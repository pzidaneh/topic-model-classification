import pandas as pd

from . import custom_timer
from . import preprocess
from . import exploration
from . import topic_model
from . import classification

work_dir="C:/Users/LENOVO/Documents/00 IPB/Tugas Akhir/Coding/"
slang_df = pd.read_csv(f"{work_dir}slang.csv")
emoji_df = pd.read_csv(f"{work_dir}emoji.csv")
#MyCleaner = preprocess.Cleaner(slang=slang_df['slang'], slang_replacement=slang_df['formal'], emoji=emoji_df['emoji'], emoji_replacement=emoji_df['makna'])
MyCleaner = preprocess.Cleaner(slang=slang_df['slang'], slang_replacement=slang_df['formal'])


def runCleaner(X, y, color_code=None, remove_news_mark=False, save_name=str(), output=True):
    # Preprocess
    #X_clust, X_class = MyCleaner.clean(X, save_name=save_name)
    X_clust, X_class = MyCleaner.clean_save(X, y, remove_news_mark=remove_news_mark, save_name=save_name)

    # Explore Before
    print("X_raw")
    print(exploration.describe_corpus(X))
    exploration.wordCloudStratified(X, y, color_code=color_code, name=f"{save_name}_raw", save=True)

    # Explore After
    print("X_clust")
    print(exploration.describe_corpus(X_clust))
    exploration.wordCloudStratified(X_clust, y, color_code=color_code, name=f"{save_name}_clust", save=True)

    print("X_class")
    print(exploration.describe_corpus(X_class))
    exploration.wordCloudStratified(X_class, y, color_code=color_code, name=f"{save_name}_class", save=True)
    exploration.wordCloudStratified(X_class, y, color_code=color_code, name=f"{save_name}_class", ngram_range=(2, 2), save=True)

    exploration.plotLabel(y)

    if output:
        return X_clust, X_class
    else:
        pass


def runPipeline(X_clust, X_class, y, n_splits=5, save_name=str()):
    split_dict = preprocess.cross_split(X, y, n_splits=n_splits)

    if len(split_dict) % 2 == 0:
        n_splits = len(split_dict)//2
        print(f"n_splits = {n_splits}")
    else:
        print(list(split_dict.keys()))
        raise Exception("length of split_dict.keys() is not an even number")

    # Cross Splitting
    for fold_num in range(n_splits):
        train_index = split_dict[f"train_{fold_num}"]
        test_index = split_dict[f"test_{fold_num}"]
        save_name_fold = f"{save_name}_fold_{fold_num}"

        print(save_name_fold)

        X_clust_train = X_clust.loc[train_index].reset_index(drop=True)
        X_class_train = X_class.loc[train_index].reset_index(drop=True)
        y_train = y.loc[train_index].reset_index(drop=True)

        X_clust_test = X_clust.loc[test_index].reset_index(drop=True)
        X_class_test = X_class.loc[test_index].reset_index(drop=True)
        y_test = y.loc[test_index].reset_index(drop=True)

        # Clustering
        lda_model = topic_model.topic_model_fit(X_clust_train, save_name=save_name_fold)
        topics_df_train = topic_model.topic_model_transform(lda_model, X_clust_train, y_train, save_name=save_name_fold)
        topics_df_test = topic_model.topic_model_transform(lda_model, X_clust_test, y_test)

        # TF-IDF
        tfidf_model = classification.feature_extraction_fit(X_class_train, save_name=save_name_fold)
        A_train, B_train, C_train = classification.feature_extraction_transform(tfidf_model, X_class_train, topics_df_train)
        A_test, B_test, C_test = classification.feature_extraction_transform(tfidf_model, X_class_test, topics_df_test)

        pd.concat([C_train, y_train], axis=1).to_csv(f"{save_name_fold}_train.csv", index=False)
        pd.concat([C_test, y_test], axis=1).to_csv(f"{save_name_fold}_test.csv", index=False)

        # Resampling
        A_train_resampled, _ = classification.handle_imbalance(A_train, y_train)
        B_train_resampled, _ = classification.handle_imbalance(B_train, y_train)
        C_train_resampled, y_train_resampled = classification.handle_imbalance(C_train, y_train)

        #### WILL DELETE LATER ON
        #continue
        #### To stop from doing classification

        # Classification
        for (features_name, X_train, X_test) in zip(
            ["A", "B", "C", "A_resampled", "B_resampled", "C_resampled"],
            [A_train, B_train, C_train, A_train_resampled, B_train_resampled, C_train_resampled],
            [A_test, B_test, C_test, A_test, B_test, C_test]
        ):
            save_name_fold_type = f"{save_name_fold}_{features_name}"
            print(f"START {save_name_fold_type}")

            model_list = classification.classification_fit(X_train, y_train, save_name=save_name_fold_type)
            classification.keep_record_fit(model_list, save_name_fold_type)
            #classification.classification_transform(X_test, y_test)
            classification.keep_record_transform(model_list, X_test, y_test, save_name=save_name, features_name=features_name, fold_num=fold_num)

            # save ALL prediction results for each iteration into a csv
            # save BEST score results for each iteration into csv

            print(f"STOP {save_name_fold_type}")

        # Make dataframe for prediction results
        #.to_csv(f"{save_name_fold}_pred.csv", index=False)


def runTopicModel(save_name, n_splits=5, start=2, stop=50):
    df = pd.read_csv(f"{save_name}_clean.csv")

    X_clust = df['TEXT_clust']
    X_class = df['TEXT_class']
    y = df['LABEL']

    split_dict = preprocess.cross_split(X_clust, y, n_splits=n_splits)

    if len(split_dict) % 2 == 0:
        n_splits = len(split_dict)//2
        print(f"n_splits = {n_splits}")
    else:
        print(list(split_dict.keys()))
        raise Exception("length of split_dict.keys() is not an even number")

    # Cross Splitting
    for fold_num in range(n_splits):
        train_index = split_dict[f"train_{fold_num}"]
        test_index = split_dict[f"test_{fold_num}"]
        save_name_fold = f"{save_name}_fold_{fold_num}"

        print(f"START TOPIC MODEL {save_name_fold}")

        X_clust_train = X_clust.loc[train_index].reset_index(drop=True)
        X_class_train = X_class.loc[train_index].reset_index(drop=True)
        y_train = y.loc[train_index].reset_index(drop=True)

        X_clust_test = X_clust.loc[test_index].reset_index(drop=True)
        X_class_test = X_class.loc[test_index].reset_index(drop=True)
        y_test = y.loc[test_index].reset_index(drop=True)

        # Clustering
        lda_model = topic_model.topic_model_fit(X_clust_train, start=start, stop=stop, save_name=save_name_fold)
        topics_df_train = topic_model.topic_model_transform(lda_model, X_clust_train, y_train, visualize=True, save_name=save_name_fold)
        topics_df_test = topic_model.topic_model_transform(lda_model, X_clust_test, y_test)

        pd.concat([X_class_train, topics_df_train, y_train], axis=1).to_csv(f"{save_name_fold}_topic_train.csv", index=False)
        pd.concat([X_class_test, topics_df_test, y_test], axis=1).to_csv(f"{save_name_fold}_topic_test.csv", index=False)

        print(f"STOP TOPIC MODEL {save_name_fold}")


def runClassification(save_name, fold_num=None, n_splits=5, ngram_range=(1, 1), feature_selection=False, random_state=None):
    # Split-wise Execution
    if fold_num is None:
        range_used = [r for r in range(n_splits)]
    else:
        range_used = [fold_num]

    print(f"range_used = {range_used}")

    # Cross Splitting
    for fold_num in range_used:
        save_name_fold = f"{save_name}_fold_{fold_num}"
        save_name_fold_ngram = f"{save_name_fold}_{ngram_range}"

        df_train = pd.read_csv(f"{save_name_fold}_topic_train.csv")
        df_test = pd.read_csv(f"{save_name_fold}_topic_test.csv")

        #print(df_train.columns)

        X_class_train = df_train['TEXT_class']
        topics_df_train =  df_train.drop(['TEXT_class', 'LABEL'], axis=1)
        y_train = df_train['LABEL']

        X_class_test = df_test['TEXT_class']
        topics_df_test =  df_test.drop(['TEXT_class', 'LABEL'], axis=1)
        y_test = df_test['LABEL']

        print(f"START FEATURE EXTRACTION {save_name_fold_ngram} AT {custom_timer.time_now(asc=True)}\n")
        # TF-IDF
        tfidf_model = classification.feature_extraction_fit(X_class_train, y_train, ngram_range=ngram_range, feature_selection=feature_selection, save_name=save_name_fold)
        A_train, B_train, C_train = classification.feature_extraction_transform(tfidf_model, X_class_train, topics_df_train, y_train, feature_selection=feature_selection)
        A_test, B_test, C_test = classification.feature_extraction_transform(tfidf_model, X_class_test, topics_df_test)

        pd.concat([C_train, y_train], axis=1).to_csv(f"{save_name_fold_ngram}_train.csv", index=False)
        pd.concat([C_test, y_test], axis=1).to_csv(f"{save_name_fold_ngram}_test.csv", index=False)

        # Resampling
        A_train_resampled, y_train_A_resampled = classification.handle_imbalance(A_train, y_train)
        B_train_resampled, y_train_B_resampled = classification.handle_imbalance(B_train, y_train)
        C_train_resampled, y_train_C_resampled = classification.handle_imbalance(C_train, y_train)

        print(f"START CLASSIFICATION {save_name_fold_ngram}\n")

    # Classification
        for (features_name, X_train, X_test, y_train_2) in zip(
            ["A", "B", "C", "A_resampled", "B_resampled", "C_resampled"],
            [A_train, B_train, C_train, A_train_resampled, B_train_resampled, C_train_resampled],
            [A_test, B_test, C_test, A_test, B_test, C_test],
            [y_train, y_train, y_train, y_train_A_resampled, y_train_B_resampled, y_train_C_resampled]
        ):
            save_name_fold_type = f"{save_name_fold_ngram}_{features_name}"
            print(f"START {save_name_fold_type} at {custom_timer.time_now(asc=True)}\n")

            model_list = classification.classification_fit(X_train, y_train_2, random_state=random_state, save_name=save_name_fold_type)
            classification.keep_record_fit(model_list, save_name_fold_type)
            #classification.classification_transform(X_test, y_test)
            classification.keep_record_predict(model_list, X_test, y_test, save_name=save_name, features_name=features_name, fold_num=fold_num, ngram_range=ngram_range)

            # save ALL prediction results for each iteration into a csv
            # save BEST score results for each iteration into csv

            print(f"STOP {save_name_fold_type}\n")

        print(f"STOP CLASSIFICATION {save_name_fold} at {custom_timer.time_now(asc=True)}\n")


# END