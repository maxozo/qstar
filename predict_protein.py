#!/usr/bin/env python
__date__ = '2023-30-05'
__version__ = '0.0.1'
__author__ = 'M.Ozols'

# This code will take the model and a tuple of protein name and compound and predict each of the C aa competition ratios.
import tensorflow as tf
from functions import get_protein_sequence
import re
import pandas as pd
import numpy as np
from competition_ration_model_Bi_LSTM_larger_encoding import encode_sequence, AA_Frequencies
from sklearn.preprocessing import StandardScaler
from pickle import load
import os
cwd = os.path.dirname(os.path.realpath(__file__))



def predict_protein(predicting_tuple,model):
    
    scaler = load(open(f'{cwd}/models/output_model_One_Bi-LSTM_epochs10_all_data_all_compounds_first2000__10__0.1_0.0001_0.001_32_scaler.pkl', 'rb'))
    compounds_info = pd.read_csv('data/compounds.txt',sep='\t',index_col=0).T
    fragment_size = int(model.split('__')[1])
    model = tf.keras.models.load_model(model)
    ps_ee=fragment_size+2
    prefix_sufix='0'*ps_ee

    protein_sequence = get_protein_sequence(predicting_tuple[1])
    protein_sequence = prefix_sufix+protein_sequence+prefix_sufix
    
    # locate all the C residues
    indices_object = re.finditer(pattern='C', string=protein_sequence)
    indices = [index.start() for index in indices_object]
    
    indices_objectR = re.finditer(pattern='R', string=protein_sequence)
    indicesR = [index.start() for index in indices_objectR]
    indices_objectK = re.finditer(pattern='K', string=protein_sequence)
    indicesK = [index.start() for index in indices_objectK]
    indicesR.extend(indicesK)
    indicesR.sort()
    predict_data = []
    positions=[]
    for idx1 in indices:
        # Now we encode each of these and return the value for each based on the model trained.
        position=idx1+1
        
        peptide = protein_sequence[position-1-10:position-1+10]
        
        ecfp4_fingerprint = compounds_info[predicting_tuple[0]]['ECFP_4']
        position_of_site=f"{protein_sequence[position-1]}{position}"
        positions.append(position_of_site)
        AA_Freq_string = AA_Frequencies(peptide,position_of_site)
        encoded_peptide_representation = encode_sequence(peptide,protein_sequence,position_of_site,fragment_size)
        combined_representation = np.concatenate([encoded_peptide_representation,AA_Freq_string,np.array(list(ecfp4_fingerprint), dtype=int)])
        predict_data.append(combined_representation)
    
    pr = np.array(predict_data)
    combined_shape = pr.shape[1]
    predict_data_reshaped = pr.reshape(-1, 1, combined_shape)
    predictions = model.predict(predict_data_reshaped)
    predictions_inverse = scaler.inverse_transform(predictions)
    predicted_values = predictions_inverse/100
    predicted_values = pd.DataFrame(predicted_values,index=positions,columns=[predicting_tuple[0]])
    return predicted_values

if __name__ == "__main__":
    
    import argparse

    # This is a software for predicting compound binding ratios.
    # Currently version 1 - better feature selection should be incorporated.
    
    """Run CLI."""
    parser = argparse.ArgumentParser(
        description="""
            Performs predictions of compound and protein.
            """
    )

    parser.add_argument(
        '-v', '--version',
        action='version',
        version='%(prog)s {version}'.format(version=__version__)
    )
    
    parser.add_argument(
        '--compound',
        action='store',
        dest='compound',
        default='CL1',
        type=str
    )    

    parser.add_argument(
        '--protein',
        action='store',
        dest='protein',
        default='GSTO1_HUMAN',
        type=str
    )    
    
    options = parser.parse_args()     
    protein = options.protein
    compound=options.compound
    
    model = f'{cwd}/models/output_model_One_Bi-LSTM_epochs10_all_data_all_compounds_first2000__10__0.1_0.0001_0.001_32'
    predicting_tuple = (compound,protein)
    predicted_values = predict_protein(predicting_tuple,model)
    predicted_values.to_csv(f'Predicted_Values_{predicting_tuple[0]}__{predicting_tuple[1]}.tsv',sep='\t')