from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from gensim.models import Word2Vec
import requests
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# This code is used to train the model that is capable in taking a the tuple (reactive fragment, target protein) and 
# acuratelly predict the competition ratio of 286 compounds (reactive fragments) listed in data/compounds.txt

# The input is a tuple where we have ('reactive fragmant','protein') 
predicting_tuple = ('CL1','GSTO1_HUMAN')


def get_protein_sequence(uniprot_id,get_protein_sequence):
    try:
        sequence = get_protein_sequence[get_protein_sequence['Entry']==uniprot_id]['Sequence'].values[0]
    except:
        # this must be an isoform not part of the retrieved uniprot ids.
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
        response = requests.get(url)
        
        if response.ok:
            # Extract the protein sequence from the response content
            lines = response.text.split("\n")
            sequence = "".join(lines[1:])  # Join all lines except the first (header)
            return sequence
        else:
            print(f"Error retrieving protein sequence for UniProt ID: {uniprot_id}")
            return None
    return sequence


def encode_sequence(peptide,protein_sequence,position_of_site):
    aa = position_of_site[0]
    peptide_pre= peptide
    peptide=peptide_pre
    position = int(position_of_site.replace(aa,''))
    peptide=peptide.replace('.','').replace('*','') #Since typically in the proteomics the peptide fragmnets before . and after indicate the next folowing amino acid we remoe these since it wasnt actuialy found by LC-MS
    # Within protein sequence we locate the full protein sequence.
    # The database was retrieved from Uniprot for all human proteins on 29th of May 2023.
    indices_object = re.finditer(pattern=peptide, string=protein_sequence)
    indices = [index.start() for index in indices_object]
    if not len(indices)>0:
        # Here the sequence that is retrieved from unprot is not the same as the one provided in the excel sheet of publication (unprot may have been updated)
        return []
    else:
        if len(indices)>1:
            if not protein_sequence[position-1]==aa:
                # Here we dont have amino acid indicated in the correct position and have multiple matches of the same peptide within protein sequence, hence we can not figure out which is the real position
                return []
    # If the indicated amino acid is not in the correct position then we need to double check whether the peptide contains multiple occurances of the same amino acid and repeat the positional selection.
    try:
        if not protein_sequence[position-1]==aa:
            return []
    except:
        # Here we have a situation that the protein sequence reported in uniprot is shorter than the one indicated in the manuscript.
        return []
    prefix_sufix='0000000000'
    protein_sequence = prefix_sufix+protein_sequence+prefix_sufix
    position=position+len(prefix_sufix)
    
    peptide = protein_sequence[position-1-10:position-1+10]
    # For peptide encoding we use the Positional Encoding as it is important which amino acid comes after which.
    # Later for predictions we will perform a sliding window approach to screen the entire length of the provided Uniprot_ID sequence.
    amino_acids = '0ACDEFGHIKLMNPQRSTVWY'
    # Positional encodin
    # To unify the lengths of the peptides used for this model we select a ±aa window of search and perform an iteration
    
    encoded_sequence = []
    position_pep = 1
    for amino_acid in peptide:
        if amino_acid == '.':
            encoded_sequence.append(np.zeros(len(amino_acids)))
        elif amino_acid == '*':
            # here there is an indication that the peptide contains modifictaion, these may be important in the particular binding analysis, however this can be investigated further.
            continue
        else:
            encoding = np.eye(len(amino_acids))[amino_acids.index(amino_acid)]
            # We may want to rething this encoding as it utilises the length of the peptide
            encoding = np.concatenate([encoding, np.array([position_pep / len(peptide)])])
            encoded_sequence.append(encoding)
            position_pep += 1
    encoded_sequence = np.concatenate(encoded_sequence)
    return encoded_sequence

def train_model():
    # here we train the model based on the data provided by GSK which is a subset of https://www.nature.com/articles/s41587-020-00778-3#Sec35
    compounds_info = pd.read_csv('data/compounds.txt',sep='\t',index_col=0).T
    competition_ratios = pd.read_csv('data/competition-ratios.tsv',sep='\t')
    sequence_infos = competition_ratios.iloc[:,0:5]
    binding_afinities = competition_ratios.iloc[:,5:]
    all_protein_sequences = pd.read_csv('data/uniprot-compressed_true_download_true_fields_accession_2Cprotein_nam-2023.05.29-17.52.20.88.tsv.gz',compression='gzip',sep='\t')
    competition_ratios = competition_ratios.sort_values(by=['Uniprot ID'])
    
    combined_data = []
    prev_protein = ''
    
    longest_peptide = max(list(sequence_infos['Peptide Sequence']), key=len)
    # for idx,row1 in sequence_infos.iterrows():
    #     print(idx)
    #     if idx==213:
    #         print('ere')
    #     peptide = sequence_infos.iloc[idx]['Peptide Sequence']
    #     uniprot_id = sequence_infos.iloc[idx]['Uniprot ID'].split('|')[1]
        
    #     position_of_site = sequence_infos.iloc[idx]['Gene + Site'].split('_')[1]
    #     # print(f'{prev_protein}=={uniprot_id}')
    #     if not prev_protein==uniprot_id:
    #         protein_sequence = get_protein_sequence(uniprot_id,all_protein_sequences)
    #         prev_protein=uniprot_id
    #     if protein_sequence == None:
    #         continue
        
    #     compounds = pd.DataFrame(binding_afinities.iloc[idx])
        
    #     for i,c1 in compounds.iterrows():
    #         # print(f"{i}:{c1}")
    #         # if i=='CL1':
    #         compound_name = i
    #         compound_competition_ratio_value = float(c1)
    #         if compound_competition_ratio_value !=compound_competition_ratio_value:
    #             # We are checking for nan values since we have missing data here and hence this can not be utilised for the ML training.
    #             continue
    #         ecfp4_fingerprint = compounds_info[compound_name]['ECFP_4']
    #         encoded_peptide_representation = encode_sequence(peptide,protein_sequence,position_of_site)
    #         if (len(encoded_peptide_representation)>0):
    #             combined_representation = np.concatenate([encoded_peptide_representation, np.array(list(ecfp4_fingerprint), dtype=int)])
    #             combined_data.append((combined_representation,compound_competition_ratio_value))
                
    #         else:
    #             # Here we have an exception where we dont have the correct peptide sequence selected.
    #             continue
    
    # with open("combined_representation.pkl", "wb") as fp:   #Pickling
    #     pickle.dump(combined_data, fp)
    with open("combined_representation.pkl", "rb") as fp:   # Unpickling
        combined_data = pickle.load(fp)
    # Shuffle the data to make sure that we pick a random protein representation for training and testing
    np.random.shuffle(combined_data)
    print('test')
    # Split the data into training and testing sets (e.g., 80% for training, 20% for testing)
    train_data = combined_data[:int(0.8 * len(combined_data))]
    test_data = combined_data[int(0.8 * len(combined_data)):]
    del combined_data
    # Extract features (input) and target values (output) for training and testing sets
    X_train = np.array([sample[0] for sample in train_data])
    y_train = np.array([sample[1] for sample in train_data])

    X_test = np.array([sample[0] for sample in test_data])
    y_test = np.array([sample[1] for sample in test_data])    
    # once we have a combined representations of the encoded_peptide_representation of ecfp4 and the peptide encoding for each of the compoundsand their coresponding binding afinity values 
    # we can partition the data and train multiple deep learning/ml methods to estimate which would be the most performant.
    
    # Standardize the input features
    scaler = StandardScaler()
    Y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
    # scaler.inverse_transform(Y_train_scaled.reshape(-1, 1))
    combined_shape = X_train.shape[1]  # Assuming ecfp4_fingerprint is the fixed-length ECFP4 string
    Y_test_scaled = scaler.fit_transform(y_test.reshape(-1, 1))
    # # Reshape input data to match LSTM input shape
    X_train_reshaped = X_train.reshape(-1, 1, combined_shape)
    X_test_reshaped = X_test.reshape(-1, 1, combined_shape)

    # input_shape = X_train.shape[1:]  # Shape of the combined representation
    # X_train_reshaped = X_train.reshape(-1, input_shape[0], input_shape[1])
    # X_test_reshaped = X_test.reshape(-1, input_shape[0], input_shape[1])

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_reshaped, Y_train_scaled, test_size=0.2, random_state=42
    )
    del X_train_reshaped 
    del Y_train_scaled

    # with open("X_train_split.pkl", "wb") as fp:   #Pickling
    #     pickle.dump(X_train_split, fp)
    # with open("X_val_split.pkl", "wb") as fp:   #Pickling
    #     pickle.dump(X_val_split, fp)
    # with open("y_train_split.pkl", "wb") as fp:   #Pickling
    #     pickle.dump(y_train_split, fp)
    # with open("y_val_split.pkl", "wb") as fp:   #Pickling
    #     pickle.dump(y_val_split, fp)
    # with open("X_test_reshaped.pkl", "wb") as fp:   #Pickling
    #     pickle.dump(X_test_reshaped, fp)
    # with open("Y_test_scaled.pkl", "wb") as fp:   #Pickling
    #     pickle.dump(Y_test_scaled, fp)


    # with open("X_train_split.pkl", "rb") as fp:   # Unpickling
    #     X_train_split = pickle.load(fp)
    # with open("X_val_split.pkl", "rb") as fp:   # Unpickling
    #     X_val_split = pickle.load(fp)
    # with open("y_train_split.pkl", "rb") as fp:   # Unpickling
    #     y_train_split = pickle.load(fp)
    # with open("y_val_split.pkl", "rb") as fp:   # Unpickling
    #     y_val_split = pickle.load(fp)
    # with open("X_test_reshaped.pkl", "rb") as fp:   # Unpickling
    #     X_test_reshaped = pickle.load(fp)
    # with open("Y_test_scaled.pkl", "rb") as fp:   # Unpickling
    #     Y_test_scaled = pickle.load(fp)


    # Create the Random Forest model
    # model = RandomForestRegressor(random_state=42)
    # model.fit(X_train_split, y_train_split)
    # model = SVR()

    model = Sequential()
    model.add(Bidirectional(LSTM(64, activation='relu',recurrent_dropout=0.02), input_shape=(1, combined_shape)))
    # model.add(LSTM(64, input_shape=(1, combined_shape)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')


    model.fit(
        X_train_split,
        y_train_split,
        validation_data=(X_val_split, y_val_split),
        epochs=10,
        batch_size=32
    )
    
    mse = model.evaluate(X_test_reshaped, Y_test_scaled)   
    
    print('Mean Squared Error:', mse)
    print('Done')
    predictions = model.predict(X_test_reshaped)
    # y_test.reshape(-1, 1)
    predictions_inverse = scaler.inverse_transform(predictions)

    mse = mean_squared_error(y_test, predictions_inverse)
    mae = mean_absolute_error(y_test, predictions_inverse)
    r2 = r2_score(y_test, predictions_inverse)

    # Print the evaluation metrics
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R2) Score:", r2)



if __name__ == "__main__":
    train_model()
    
print('Done')