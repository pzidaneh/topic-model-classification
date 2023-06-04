import pandas as pd
import numpy as np
from joblib import dump

from imblearn.over_sampling import RandomOverSampler

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer


def handle_imbalance(X, y, random_state=55):
    ROS = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ROS.fit_resample(pd.DataFrame(X), pd.Series(y))

    return X_resampled, y_resampled


def feature_extraction_fit(X, y=None, ngram_range=(1, 1), feature_selection=False, mi_threshold=0, save_name=str()):
    # https://stackoverflow.com/questions/57333183/sk-learn-countvectorizer-keeping-emojis-as-words
    #token_pattern = r"[^\s]+"  # To include everything. Needed for emoji, but will also include punctuations
    token_pattern = r"(?u)\b\w\w+\b"  # default
    tfidf_model = TfidfVectorizer(token_pattern=token_pattern, ngram_range=ngram_range, lowercase=False)
    tfidf_model.fit(X.fillna(""))

    if feature_selection:
        if y is None:
            raise Exception("y cannot be None")
        
        X_tfidf = pd.DataFrame(
            tfidf_model.transform(X.fillna("")).A,
            columns = tfidf_model.get_feature_names_out()
        )

        num_of_class = len(np.unique(y))
        if num_of_class == 2:
            minority_label = pd.Series(y).value_counts().idxmin()
            label = np.where(y == minority_label, 1, 0)
        else:
            label = y.copy()

        mi = mutual_info_classif(X_tfidf, label)

        print(f"VOCAB: Original {X_tfidf.shape[1]} vs Selected {(mi > mi_threshold).sum()}")

        tfidf_model = TfidfVectorizer(token_pattern=token_pattern, ngram_range=ngram_range, lowercase=False, vocabulary=X_tfidf.columns[mi > mi_threshold])
        tfidf_model.fit(X.fillna(""))

    if save_name != str():
        dump(tfidf_model, f"{save_name}_{ngram_range}_tfidf.joblib")

    return tfidf_model


def feature_extraction_transform(tfidf_model, X, topics_df, y=None, feature_selection=False, mi_threshold=0):
    # To be implemented SEPARATELY on train AND test
    # A: Text (TF-IDF)
    # B: Topics
    # C: A + B
    A = pd.DataFrame(
        tfidf_model.transform(X.fillna("")).A,
        columns = tfidf_model.get_feature_names_out()
    )

    B = topics_df.copy().add_prefix("TOPIC #")
    B.columns = [str(i).zfill(3) for i in B.columns]
    B = B.reindex(sorted(B.columns), axis=1)

    if feature_selection:
        if y is None:
            raise Exception("y cannot be None")
        else:
            num_of_class = len(np.unique(y))

            if num_of_class == 2:
                minority_label = pd.Series(y).value_counts().idxmin()
                label = np.where(y == minority_label, 1, 0)
            else:
                label = y.copy()

            mi = mutual_info_classif(B, label)
            B = B.loc[:, mi > mi_threshold]
            print(f"TOPIC: Original {B.shape[1]} vs Selected {(mi > mi_threshold).sum()}")

    C = pd.concat([A, B], axis=1)

    return A, B, C


def f1_auto(y_true, y_pred):
    minority_label = pd.Series(y_true).value_counts().idxmin()
    f1 = f1_score(y_true, y_pred, pos_label=minority_label)

    return f1


def classification_fit(X_train, y_train, save_name=str(), cv=5, verbose=10, try_algorithm=["lr", "da", "knn", "rf"], random_state=55):
    model_list = []
    num_of_class = len(np.unique(y_train))
    #print(num_of_class)

    if num_of_class == 2:
        #scoring = "f1"
        scoring = make_scorer(f1_auto, greater_is_better=True)
    elif num_of_class > 2:
        scoring = "f1_macro"
    else:
        raise Exception("y_train only have 1 class (or worse, none at all)")
    
    # Logistic Regression
    if "lr" in try_algorithm:
        lr = GridSearchCV(
            estimator=LogisticRegression(),
            param_grid= {
                #'penalty': ["none", "l1", "l2", "elasticnet"],
                'penalty': ["elasticnet"],
                'random_state': [random_state],
                #'solver': ["lbfgs", "newton-cg", "sag", "saga"],
                'solver': ["saga"],
                #'max_iter': [1000],
                #'multi_class': ['ovr', 'multinomial'],
                'multi_class': ['multinomial'],
                'l1_ratio': [0.25, 0.50, 0.75]
            },
            scoring=scoring, cv=cv, verbose=verbose
        )
        lr.fit(X_train, y_train)
        model_list.append(lr)
    
    # Discriminant Analysis (Linear)
    if "da" in try_algorithm:
        da = GridSearchCV(
            estimator=LinearDiscriminantAnalysis(),
            param_grid={
                #'solver': ["svd", "lsqr", "eigen"],
                'solver': ["lsqr"],
                #'shrinkage': [.25, .50, .75, 1, "auto"],
                'shrinkage': [.25, .50, .75]
            },
            scoring=scoring, cv=cv, verbose=verbose
        )
        da.fit(X_train, y_train)
        model_list.append(da)
    
    # K-Nearest Neighbour
    if "knn" in try_algorithm:
        knn = GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid={
                'n_neighbors': [3, 5, 7, 9],
                #'weights': ["uniform", "distance"],
                #'metric': ['euclidean', 'cosine', 'manhattan']
            },
            scoring=scoring, cv=cv, verbose=verbose
        )
        knn.fit(X_train, y_train)
        model_list.append(knn)
    
    # Random Forest
    if "rf" in try_algorithm:
        rf = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid={
                'n_estimators': [100, 200],
                #'criterion': ["gini", "entropy", "log_loss"],
                'max_features': ["sqrt", "log2"],  # "log2"
                #'oob_score': [True, False],  # have no effect on accuracy (f1) but a bit slower
                'random_state': [random_state],
                #'max_samples': [.25, .50]
            },
            scoring=scoring, cv=cv, verbose=verbose
        )
        rf.fit(X_train, y_train)
        model_list.append(rf)
    
    # Save to csv
    #track_hyperparam()
    #track_runtime()
    
    # Already on the pipeline
    #keep_record_fit(model_list, save_name=save_name)

    return model_list


def keep_record_fit(model_list, save_name):
    if save_name == str():
        return None

    summary = None
    for model in model_list:
        algorithm = str(model.estimator).replace("()", "")

        # Will overload disk space
        #dump(model_list, f"{save_name}_{algorithm}.joblib")

        result_df = pd.DataFrame(model.cv_results_).sort_values(by='rank_test_score')
        result_df.insert(0, 'algorithm', algorithm)

        if summary is not None:
            summary = pd.concat([summary, result_df], axis=0, ignore_index=True)
        else:
            summary = result_df.copy()

    params = summary['params']
    param_ = summary.filter(like='param_')
    score = summary.drop(summary.filter(like='param').columns, axis=1)

    #score.isna()

    summary = pd.concat([score, params, param_], axis=1)

    save_name_csv = f"{save_name}_fit.csv"
    summary.to_csv(save_name_csv, index=False)

    print(f"save_name_csv: {save_name_csv}")
    #return save_name_csv


def classification_predict(model, X, y):
    pred = model.predict(X)
    
    num_of_class = len(np.unique(y))
    if num_of_class == 2:
        #score = f1_score(y, pred, average='binary')
        score = f1_auto(y, pred)
    elif num_of_class > 2:
        score = f1_score(y, pred, average='macro')
    else:
        raise Exception(f"y only have {num_of_class} class, must at least be 2")

    return pred, score


def keep_record_predict(model_list, X_test, y_test, save_name, features_name, fold_num, ngram_range):
    """
    From best model(s), keep records of
    1. y_pred
    2. scoring

    Will overwrite existing csv to add new records
    """
    save_name_pred = f"{save_name}_pred.csv"
    save_name_score = f"{save_name}_score.csv"

    try:
        pred_df = pd.read_csv(save_name_pred).reset_index(drop=True)
    except FileNotFoundError:
        pred_df = pd.DataFrame()

    try:
        score_df = pd.read_csv(save_name_score, index_col=0)
    except FileNotFoundError:
        score_df = pd.DataFrame()

    for model in model_list:
        algorithm = str(model.estimator).replace("()", "")
        #params = str(model.best_params_)
        y_pred, score = classification_predict(model, X_test, y_test)
        #y_pred = y_pred.reset_index(drop=True)  # y_pred is np.array

        pred_df.loc[:, f"{algorithm}_{features_name}_fold_{fold_num}_{ngram_range}"] = pd.Series(y_pred)
        score_df.loc[f"{features_name}_fold_{fold_num}_{ngram_range}", f"{algorithm}"] = score

    pred_df.to_csv(save_name_pred, index=False)
    score_df.to_csv(save_name_score, index=True)

    print(f"save_name_pred: {save_name_pred}")
    print(f"save_name_score: {save_name_score}")


# END