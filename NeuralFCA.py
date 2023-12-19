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
import networkx as nx
#Scoring
from sklearn.metrics import accuracy_score, f1_score
#KFold
from sklearn.model_selection import KFold
#Binarization
from MLClassification import NumOneHotEncoder



#To binarise a DataFrame
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
    L = ConceptLattice.from_context(K_train, algo='Sofia', is_monotone=True)
    for c in L:
        y_preds = np.zeros(K_train.n_objects)
        y_preds[list(c.extent_i)] = 1
        c.measures['f1_score'] = f1_score(Y_train, y_preds, average='micro', zero_division=1)
        #c.measures['f1_score'] = accuracy_score(Y_train, y_preds)
    best_concepts = list(L.measures['f1_score'].argsort()[::-1][:212])
    assert len({g_i for c in L[best_concepts] for g_i in c.extent_i})==K_train.n_objects, "Selected concepts do not cover all train objects"
    cn = nl.ConceptNetwork.from_lattice(L, best_concepts, sorted(set(Y_train)))

    #vis = LineVizNx(node_label_font_size=14, node_label_func=lambda el_i, P: nl.neuron_label_func(el_i, P, set(cn.attributes))+'\n\n')
    #vis.init_mover_per_poset(cn.poset)
    #mvr = vis.mover
    #descr = {'experience_level_SE','company_size_L','employment_type_FT'}
    #traced = cn.trace_description(descr, include_targets=False)
    #fig, ax = plt.subplots(figsize=(15,5))

    #vis_attrs = ['att_'+str(i) for i in range(len(cn.attributes))]
    #vis.draw_poset(
    #    cn.poset, ax=ax,
    #    flg_node_indices=False,
    #    node_label_func=lambda el_i, P: nl.neuron_label_func(el_i, P, set(cn.attributes), only_new_attrs=True)+'\n\n',
    #    node_color=['darkblue' if el_i in traced else 'lightgray' for el_i in range(len(cn.poset))]
    #)
    #plt.title(f'NN based on 7 best concepts from monotone concept lattice', loc='left', x=0.05, size=24)
    #plt.text(max(vis.mover.posx), min(vis.mover.posy)-0.3, f'*Blue neurons are the ones activated by description {descr}', fontsize=14, ha='right', color='dimgray')
    #plt.subplots_adjust()
    #plt.tight_layout()
    #plt.show()

    cn.fit(X_train, Y_train)
    return cn




def VisualiseFCA_NN(cn):
    edge_weights = cn.edge_weights_from_network()
    fig, ax = plt.subplots(figsize=(15,5))
    vis = LineVizNx(node_label_font_size=14, node_label_func=lambda el_i, P: nl.neuron_label_func(el_i, P, set(cn.attributes))+'\n\n')
    vis.init_mover_per_poset(cn.poset)
    vis.draw_poset(
        cn.poset, ax=ax,
        flg_node_indices=False,
        node_label_func=lambda el_i, P: nl.neuron_label_func(el_i, P, set(cn.attributes), only_new_attrs=True)+'\n\n',
        edge_color=[edge_weights[edge] for edge in cn.poset.to_networkx().edges],
        edge_cmap=plt.cm.RdBu,
        )
    nx.draw_networkx_edge_labels(cn.poset.to_networkx(), vis.mover.pos, {k: f"{v:.1f}" for k,v in edge_weights.items()}, label_pos=0.7)
    plt.title('Neural network with fitted edge weights', size=24, x=0.05, loc='center')
    plt.tight_layout()
    plt.subplots_adjust()
    #plt.savefig('fitted_network.png')
    plt.show()




def NeuralFCAClassification(features, processed_df, target):
    try:
        print('Neural FCA classification for:')
        print(processed_df.head())
        #Binarize df
        X = processed_df.drop(target, axis=1)
        #Binarise data
        X = OneHotEncodeDF(X,6)
        Y = NumOneHotEncoder(processed_df[target],6, target)
        #Covert to bool dataframes
        X.replace({0: False, 1: True}, inplace=True)
        Y.replace({0: False, 1: True}, inplace=True)
        #Make the index str
        X.set_index([pd.Index(['object_'+str(i) for i in range(len(X))])], inplace=True)
        Y.set_index([pd.Index(['object_'+str(i) for i in range(len(Y))])], inplace=True)
        #Reduce object number
        #X = X.iloc[:50]
        #Y = Y.iloc[:50]
        accuracy_scores = []
        f1_scores = []
        kf = KFold(n_splits=5, shuffle=False, random_state=None)
        #KFold index
        index = 1
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
            try:
                print('Neural FCA algorithm: KFold {}...'.format(index))
                #Execute for all binary columns of the OneHotEncoded numerical target attribute
                Y_pred = pd.DataFrame(index=Y_test.index)
                for i in range(Y_train.shape[1]):
                    col_name = Y_train.iloc[:,i].name
                    print('Predicting {}...'.format(col_name))
                    cn = NeuralFCA_Algorithm(X_train, Y_train.iloc[:,i], X_test, Y_test.iloc[:,i])
                    col_result = pd.DataFrame(cn.predict(X_test).numpy(), columns=[col_name], index=Y_test.index)
                    #Transform result to binary format like the testing data
                    col_result.replace({0: False, 1: True}, inplace=True)
                    Y_pred = pd.concat([Y_pred, col_result], axis=1)
                    #print('Accuracy score for this column of the target: {}'.format(accuracy_score(col_result,Y_test.iloc[:,i])))
                    #Visualise the NN with weights
                    #VisualiseFCA_NN(cn)
                accuracy_scores.append(accuracy_score(Y_test, Y_pred))
                f1_scores.append(f1_score(Y_test, Y_pred, average='micro', zero_division=1))
                #print('Accuracy score: {}'.format(accuracy_scores[-1]))
                #print('F1 score: {}'.format(f1_scores[-1]))
            except Exception as e:
                print(r'Unable to execute Neural FCA algorithm: {}'.format(str(e)))
            #Increment KFold index
            index += 1
        #Print average accuracy and f1
        print('Accuracy scores:')
        print(accuracy_scores)
        print('Average: {}'.format(sum(accuracy_scores)/len(accuracy_scores)))
        print('F1 scores:')
        print(f1_scores)
        print('Average: {}'.format(sum(f1_scores)/len(f1_scores)))

    except Exception as e:
        print(r'Unable to classify data with Neural FCA: {}'.format(str(e)))
