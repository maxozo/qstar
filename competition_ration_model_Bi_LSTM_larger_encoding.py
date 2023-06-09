#!/usr/bin/env python

__date__ = '2023-30-05'
__version__ = '0.0.1'
__author__ = 'M.Ozols'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional,Dropout
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import requests
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import matplotlib.pyplot as plt 
from collections import Counter
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from pickle import dump

# This code is used to train the model that is capable in taking a the tuple (reactive fragment, target protein) and 
# acuratelly predict the competition ratio of 286 compounds (reactive fragments) listed in data/compounds.txt
# The input is a tuple where we have ('reactive fragmant','protein'):
# predicting_tuple = ('CL1','GSTO1_HUMAN')


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


def AA_Frequencies(peptide,position_of_site):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    counts=Counter(peptide)
    aa = position_of_site[0]
    position = int(position_of_site.replace(aa,''))
    Freq_encoding = np.array([])
    for amino_acid in amino_acids:
        count1 = counts[amino_acid]
        Freq_encoding = np.concatenate([Freq_encoding, np.array([int(count1)])])
        amino_acids
    Freq_encoding = np.concatenate([Freq_encoding, np.array([int(position)])])
    return Freq_encoding

def encode_sequence(peptide,protein_sequence,position_of_site,fragment_size):
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
    ps_ee=fragment_size+2
    prefix_sufix='0'*ps_ee
    protein_sequence = prefix_sufix+protein_sequence+prefix_sufix
    position=position+len(prefix_sufix)
    
    peptide = protein_sequence[position-1-fragment_size:position-1+fragment_size]
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
            try:
                encoding = np.eye(len(amino_acids))[amino_acids.index(amino_acid)]
            except:
                encoding = np.zeros(len(amino_acids))
            # We may want to rething this encoding as it utilises the length of the peptide
            encoding = np.concatenate([encoding, np.array([position_pep / len(peptide)])])
            
            encoded_sequence.append(encoding)
            position_pep += 1
    encoded_sequence = np.concatenate(encoded_sequence)
    return encoded_sequence

def train_model(dropout,l2_reguliser,lr,batch_size,fragment_size):
    print(f'output_model_One_CNN_epochs10_all_data_{dropout}_{l2_reguliser}_{lr}_{batch_size}')
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
    for idx,row1 in sequence_infos.iterrows():
        print(idx)
        if idx>2500:
            # We skipp the folowing proteins for this chalange, just to indicate approaches that would be taken for building model for this task.
            continue
        peptide = sequence_infos.iloc[idx]['Peptide Sequence']
        uniprot_id = sequence_infos.iloc[idx]['Uniprot ID'].split('|')[1]
        
        position_of_site = sequence_infos.iloc[idx]['Gene + Site'].split('_')[1]
        # print(f'{prev_protein}=={uniprot_id}')
        if not prev_protein==uniprot_id:
            protein_sequence = get_protein_sequence(uniprot_id,all_protein_sequences)
            prev_protein=uniprot_id
        if protein_sequence == None:
            continue
        
        compounds = pd.DataFrame(binding_afinities.iloc[idx])
        
        for i,c1 in compounds.iterrows():
            # print(f"{i}:{c1}")
            # if i not in ['CL1','CL2','CL3','CL4','CL5','CL6','CL7','CL8','CL9','CL10','CL11','CL12','CL13','CL14','CL15','CL16','CL17','CL18','CL19','CL20']:
            #     continue
            compound_name = i
            combined_representation=np.array([])
            compound_competition_ratio_value = float(c1)
            if compound_competition_ratio_value !=compound_competition_ratio_value:
                # We are checking for nan values since we have missing data here and hence this can not be utilised for the ML training.
                continue
            compound_competition_ratio_value =compound_competition_ratio_value *100
            ecfp4_fingerprint = compounds_info[compound_name]['ECFP_4']
            AA_Freq_string = AA_Frequencies(peptide,position_of_site)
            encoded_peptide_representation = encode_sequence(peptide,protein_sequence,position_of_site)
            if (len(encoded_peptide_representation)>0):
                combined_representation = np.concatenate([encoded_peptide_representation,AA_Freq_string,np.array(list(ecfp4_fingerprint), dtype=int)])
                combined_data.append((combined_representation,compound_competition_ratio_value))
                
            else:
                # Here we have an exception where we dont have the correct peptide sequence selected.
                continue
    
    with open(f"combined_representation_LSTM_{coment}__{fragment_size}.pkl", "wb") as fp:   #Pickling
        pickle.dump(combined_data, fp)
    # with open(f"combined_representation_LSTM_{coment}__{fragment_size}.pkl", "rb") as fp:   # Unpickling
    #     combined_data = pickle.load(fp)
    # Shuffle the data to make sure that we pick a random protein representation for training and testing
    np.random.shuffle(combined_data)
    print('Data Loaded, lets train')
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
    Y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

    dump(scaler, open(f'output_model_One_Bi-LSTM_epochs10_all_data_{coment}__{fragment_size}__{dropout}_{l2_reguliser}_{lr}_{batch_size}_scaler.pkl', 'wb'))
    
    # # Reshape input data to match LSTM input shape
    X_train_reshaped = X_train.reshape(-1, 1, combined_shape)
    X_test_reshaped = X_test.reshape(-1, 1, combined_shape)
    
    # Erly stopping is defined to monitor the validation loss while training loss is improving, to avoid overtaining the model.
    early_stopping = EarlyStopping(
        monitor='val_loss',
        verbose=1,
        patience=5,
        mode='min',
        restore_best_weights=True)
    
    model = Sequential()
    model.add(Bidirectional(LSTM(254, activation='relu',recurrent_dropout=dropout), input_shape=(1, combined_shape)))
    model.add(Dense(56))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(
        X_train_reshaped,
        Y_train_scaled,
        validation_split=0.2,
        epochs=70,
        batch_size=batch_size,
        callbacks=early_stopping
    )
    
    model.save(f'output_model_One_Bi-LSTM_epochs10_all_data_{coment}__{fragment_size}__{dropout}_{l2_reguliser}_{lr}_{batch_size}')
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
    

    # Extract training and validation loss and plot it over time to se the training performance.
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'go-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'output_model_One_Bi-LSTM_epochs10_all_data_{coment}__{fragment_size}__{dropout}_{l2_reguliser}_{lr}_{batch_size}.png')
    plt.close()



if __name__ == "__main__":
    import argparse
    # Defaults
    dropout=0.2
    l2_reguliser=0.001
    lr=0.001
    batch_size=32
    
    # Take different params for training
    """Run CLI."""
    parser = argparse.ArgumentParser(
        description="""
            Performs grid search on HPC
            """
    )

    parser.add_argument(
        '-v', '--version',
        action='version',
        version='%(prog)s {version}'.format(version=__version__)
    )
    
    parser.add_argument(
        '--dropout',
        action='store',
        dest='dropout',
        default=0.5,
        type=float
    )

    parser.add_argument(
        '--l2_reguliser',
        action='store',
        dest='l2_reguliser',
        default=0.001,
        type=float
    )
            
    parser.add_argument(
        '--lr',
        action='store',
        dest='lr',
        default=0.001,
        type=float
    )
         
    parser.add_argument(
        '--batch_size',
        action='store',
        dest='batch_size',
        default=32,
        type=int
    )
    
    parser.add_argument(
        '--fragment_size',
        action='store',
        dest='fragment_size',
        default=20,
        type=int
    )
    
    parser.add_argument(
        '--coment',
        action='store',
        dest='coment',
        default='no_com',
        type=str
    )
    
    options = parser.parse_args()     
    dropout = options.dropout
    fragment_size=options.fragment_size
    l2_reguliser = options.l2_reguliser
    lr = options.lr
    batch_size = options.batch_size
    coment = options.coment
    train_model(dropout,l2_reguliser,lr,batch_size,fragment_size)
    
print('Done')