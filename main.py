#Marwan Bouabidi
#HSE Data Science M1
#Core libs
import pandas as pd
import numpy as np
#Neural FCA
import MLClassification
import NeuralFCA

#Data
datasets = ['ds_salaries', 'world_population', 'US_electricity_2017']
ds_salaries_features = ['job_title', 'experience_level', 'salary_in_usd',\
    'work_year', 'employment_type', 'remote_ratio', 'company_size']
world_population_features = ['Country/Territory', 'Rank', 'Continent',\
    'Growth Rate', '2022 Population', '2020 Population', '2015 Population',\
    '2010 Population', '2000 Population', 'Area (km²)','Density (per km²)']
US_electricity_features = ['Utility.Number', 'Utility.Type',\
    'Demand.Summer Peak', 'Sources.Total', 'Uses.Total',\
    'Retail.Total.Customers']
features = [ds_salaries_features, world_population_features, US_electricity_features]
targets = [ds_salaries_features[2], world_population_features[3], US_electricity_features[2]]


def readData(dataset_name, features):
    try:
        dataPath = r'datasets/{}.csv'.format(dataset_name)
        df = pd.read_csv(dataPath)
        #print(df.head())
        processed_df = pd.DataFrame()
        for feature in features:
            #Detect if there are strings in the feature's column
            non_numeric_columns = df.select_dtypes(exclude=['number']).columns.tolist()
            if(feature in non_numeric_columns):
                #Check if binarization is possible, by checking category number
                binarized_df_column = pd.get_dummies(df[feature], prefix=feature)
                bin_categories = binarized_df_column.columns.tolist()
                if(len(bin_categories) <= 20):
                    #Add to processed_df
                    processed_df = pd.concat([processed_df, binarized_df_column], axis=1)
            else:
                processed_df = pd.concat([processed_df, df[feature]], axis=1)
        return processed_df
    except Exception as e:
        print(r'Unable to read data: {}'.format(str(e)))
        return pd.DataFrame()

#Main
if __name__=="__main__":
    for i in range(len(datasets)):
        processed_df = readData(datasets[i], features[i])
        dataset_features = features[i]
        print('')
        print(r'Treating {} data...'.format(datasets[i]))
        #MLClassification.ClassificationAlgorithms(dataset_features, processed_df, targets[i])
        NeuralFCA.NeuralFCAClassification(dataset_features, processed_df, targets[i])
