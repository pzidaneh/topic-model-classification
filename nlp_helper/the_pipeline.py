def modelling(X_train, y_train, X_test, y_test, save_name):
    lda_model = topic_model_fit(X_train, max_topics=100)
    
    A_train, B_train, C_train = topic_model_transform(lda_model, X_train)
    A_test, B_test, C_test = topic_model_transform(lda_model, X_test)
    
    lr, da, knn, rf = classification_fit(X_train, y_train, save_name, random_state=55, verbose=10)
    pred, score = classification_transform(model, X, y)


def cross_val_modelling(data):
    # cross_val_split
    # loop to apply modelling()
    return None