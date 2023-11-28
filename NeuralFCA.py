#Core libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = (1,1,1,1)
#Neural FCA dependencies
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice
from fcapy.visualizer import LineVizNx
import neural_lib as nl
#Scoring
from sklearn.metrics import accuracy_score, f1_score
#KFold
from sklearn.model_selection import KFold
#Binarization
from MLClassification import NumOneHotEncoder


def OneHotEncodeDF(df,n):
    bin_df = pd.DataFrame()
    columns = df.columns.tolist()
    for column in columns:
        if(len(df[column].value_counts())>2):
            bin_df = pd.concat([bin_df, NumOneHotEncoder(df[column],n,column)], axis=1)
        else:
            bin_df = pd.concat([bin_df, df[column]], axis=1)
    return bin_df


def NeuralFCA_Algorithm(X_train, Y_train, X_test, Y_test):
    K_train = FormalContext.from_pandas(X_train)
    #print(K_train)
    L = ConceptLattice.from_context(K_train, is_monotone=True)
    print(len(L))
    print('Concept lattice done')
    for c in L:
        y_preds = np.zeros(K_train.n_objects)
        y_preds[list(c.extent_i)] = 1
        #Target one hot encoded, we average f1 score
        c.measures['f1_score'] = f1_score(Y_train, y_preds, average='micro', zero_division=1)
    best_concepts = list(L.measures['f1_score'].argsort()[::-1][:7])
    print({g_i for c in L[best_concepts] for g_i in c.extent_i})
    print(K_train.n_objects)
    #assert len({g_i for c in L[best_concepts] for g_i in c.extent_i})==K_train.n_objects, "Selected concepts do not cover all train objects"
    cn = nl.ConceptNetwork.from_lattice(L, best_concepts, sorted(set(Y_train)))
    vis = LineVizNx(node_label_font_size=14, node_label_func=lambda el_i, P: nl.neuron_label_func(el_i, P, set(cn.attributes))+'\n\n')
    vis.init_mover_per_poset(cn.poset)
    mvr = vis.mover
    descr = {''}
    traced = cn.trace_description(descr, include_targets=False)
    fig, ax = plt.subplots(figsize=(15,5))

    vis.draw_poset(
        cn.poset, ax=ax,
        flg_node_indices=False,
        node_label_func=lambda el_i, P: nl.neuron_label_func(el_i, P, set(cn.attributes), only_new_attrs=True)+'\n\n',
        node_color=['darkblue' if el_i in traced else 'lightgray' for el_i in range(len(cn.poset))]
    )
    plt.title(f'NN based on 7 best concepts from monotone concept lattice', loc='left', x=0.05, size=24)
    plt.text(max(vis.mover.posx), min(vis.mover.posy)-0.3, f'*Blue neurons are the ones activated by description {descr}', fontsize=14, ha='right', color='dimgray')
    plt.subplots_adjust()
    plt.tight_layout()
    #plt.show()

    cn.fit(X_train, Y_train)
    return cn




def NeuralFCAClassification(features, processed_df, target):
    #try:
        print('Neural FCA classification for:')
        print(processed_df.head())
        #Binarize df
        #bin_df = OneHotEncodeDF(bin_df,6)
        X = processed_df.drop(target, axis=1)
        #OneHotEncode Data
        X = OneHotEncodeDF(X,5)
        Y = NumOneHotEncoder(processed_df[target], 5, target)
        #Covert to bool dataframes
        X.replace({0: False, 1: True}, inplace=True)
        Y.replace({0: False, 1: True}, inplace=True)
        #Make the index str
        X.set_index([pd.Index(['object_'+str(i) for i in range(len(X))])], inplace=True)
        Y.set_index([pd.Index(['object_'+str(i) for i in range(len(Y))])], inplace=True)
        #Reduce object number
        X = X.iloc[:20]
        Y = Y.iloc[:20]
        accuracy_scores = []
        f1_scores = []
        kf = KFold(n_splits=5, shuffle=False, random_state=None)
        #index = 1
        train_index, test_index = list(kf.split(X))[0]
        print(train_index, test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        #NeuralFCA_Algorithm(X_train, Y_train, X_test, Y_test)
        if True:
        #try:
            #print('Neural FCA algorithm: KFold {}'.format(index))
            #Execute for all binary columns of the OneHotEncoded numerical target attribute
            Y_pred = pd.DataFrame(index=Y_test.index)
            for i in range(Y_train.shape[1]):
                col_name = Y_train.iloc[:,i].name
                #print(Y_train.iloc[:,i])
                cn = NeuralFCA_Algorithm(X_train, Y_train.iloc[:,i], X_test, Y_test.iloc[:,i])
                col_result = pd.DataFrame(cn.predict(X_test).numpy(), columns=[col_name], index=Y_test.index)
                Y_pred = pd.concat([Y_pred, col_result], axis=1)
                print('Class prediction', Y_pred)
                print('Class prediction with probabilities', cn.predict_proba(X_test).detach().numpy())
                print('True class', Y_test)
                #accuracy_scores.append()
                #f1_scores.append()
            print('Accuracy score: ' ,accuracy_score(Y_test, Y_pred))
            print('F1 score: ' ,f1_score(Y_test, Y_pred, average='micro', zero_division=1))
        #except Exception as e:
        #    print(r'Unable to execute Neural FCA algorithm: {}'.format(str(e)))
        #index += 1
        #Print average accuracy and f1

    #except Exception as e:
    #    print(r'Unable to classify data with Neural FCA: {}'.format(str(e)))
