import pandas as pd
#Standard ML methods
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score


def NumOneHotEncoder(df_target, n, target):
    try:
        min = df_target.min()
        max = df_target.max()
        step = abs(max-min)/n
        bin_edges = [min+i*step for i in range(n+1)]
        bin_labels = [r'{}_{}'.format(target, i) for i in range(n)]
        #bin_labels = [r'{}_{}'.format(target[0:4], i) for i in range(n)]
        categorical_df_target = pd.cut(df_target, bins=bin_edges, labels=bin_labels)
        return pd.get_dummies(categorical_df_target)
    except Exception as e:
        print(r'Unable to OneHotEncode numerical column: {}'.format(str(e)))


def ClassificationAlgorithms(features, processed_df, target):
    try:
        print("Processed dataframe: ")
        print(processed_df.columns.tolist())
        X = processed_df.drop(target, axis=1)
        #The larger the discretisation, the worse the classification
        Y = NumOneHotEncoder(processed_df[target], 6, target)
        accuracy_scores = {'DecTrs':[], 'RandFor': [], 'kNN':[], 'XGBoost':[]}
        f1_scores = {'DecTrs':[], 'RandFor': [], 'kNN':[], 'XGBoost':[]}
        kf = KFold(n_splits=5, shuffle=False, random_state=None)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
            #Model 1: Decision trees
            try:
                clf_A = DecisionTreeClassifier()
                clf_A.fit(X_train, Y_train)
                Y_pred = clf_A.predict(X_test)
                accuracy_scores['DecTrs'].append(accuracy_score(Y_test, Y_pred))
                f1_scores['DecTrs'].append(f1_score(Y_test, Y_pred, average='micro', zero_division=1))
            except Exception as e:
                print(r'Unable to classify data with decision trees: {}'.format(str(e)))

            ##Model 2: Random forests
            try:
                clf_B = RandomForestClassifier()
                clf_B.fit(X_train, Y_train)
                Y_pred = clf_B.predict(X_test)
                accuracy_scores['RandFor'].append(accuracy_score(Y_test, Y_pred))
                f1_scores['RandFor'].append(f1_score(Y_test, Y_pred, average='micro', zero_division=1))
            except Exception as e:
                print(r'Unable to classify data with random forests: {}'.format(str(e)))

            ##Model 3: kNN
            try:
                clf_C = KNeighborsClassifier(n_neighbors=5)
                clf_C.fit(X_train, Y_train)
                Y_pred = clf_C.predict(X_test)
                accuracy_scores['kNN'].append(accuracy_score(Y_test, Y_pred))
                f1_scores['kNN'].append(f1_score(Y_test, Y_pred, average='micro', zero_division=1))
            except Exception as e:
                print(r'Unable to classify data with kNN: {}'.format(str(e)))

            ##Model 4: Logistic regression
            try:
                #No binarization for logistic regression
                #Y_train, Y_test = processed_df[target].iloc[train_index], processed_df[target].iloc[test_index]
                clf_D = XGBClassifier()
                clf_D.fit(X_train, Y_train)
                Y_pred = clf_D.predict(X_test)
                accuracy_scores['XGBoost'].append(accuracy_score(Y_test, Y_pred))
                f1_scores['XGBoost'].append(f1_score(Y_test, Y_pred, average='micro', zero_division=1))
            except Exception as e:
                print(r'Unable to classify data with XGBoost: {}'.format(str(e)))


        #accuracy_scores = cross_val_score(clf, X_train, Y_train, cv=cv, scoring='accuracy', error_score='raise')
        #f1_scores = cross_val_score(clf, X_train, Y_train, cv=cv, scoring='f1_macro', error_score='raise')
        print(r'Accuracy scores: ')
        print(accuracy_scores)
        for key in accuracy_scores:
            scores_A = accuracy_scores[key]
            print(r'Average accuracy for {}: {}'.format(key, round(sum(scores_A)/len(scores_A), 3)))
        print(r'F1 scores: ')
        print(f1_scores)
        for key in accuracy_scores:
            scores_B = f1_scores[key]
            print(r'Average F1 score for {}: {}'.format(key, round(sum(scores_B)/len(scores_B), 3)))
    except Exception as e:
        print(r'Unable to classify data with classic ML methods: {}'.format(str(e)))
