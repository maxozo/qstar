# This code will take the model and a tuple of protein name and compound and predict each of the C aa competition ratios.
import tensorflow as tf
from functions import get_protein_sequence
import re
import pandas as pd
import numpy as np
from competition_ration_model_Bi_LSTM import encode_sequence
from sklearn.preprocessing import StandardScaler
from pickle import load

def predict_protein(predicting_tuple,model):
    
    scaler = load(open('/lustre/scratch123/hgi/projects/huvec/analysis/ml/qstar/output_model_One_Bi-LSTM_epochs10_all_data_0.5_0.005_0.001_32_scaler.pkl', 'rb'))
    compounds_info = pd.read_csv('/lustre/scratch123/hgi/projects/huvec/analysis/ml/qstar/data/compounds.txt',sep='\t',index_col=0).T
    model = tf.keras.models.load_model(model)
    prefix_sufix='0000000000'
    protein_sequence = get_protein_sequence(predicting_tuple[1])
    protein_sequence = prefix_sufix+protein_sequence+prefix_sufix
    
    # locate all the C residues
    indices_object = re.finditer(pattern='C', string=protein_sequence)
    indices = [index.start() for index in indices_object]
    predict_data = []
    for idx1 in indices:
        # Now we encode each of these and return the value for each based on the model trained.
        position=idx1+1
        peptide = protein_sequence[position-1-10:position-1+10]
        ecfp4_fingerprint = compounds_info[predicting_tuple[0]]['ECFP_4']
        position_of_site=f"{protein_sequence[position-1]}{position}"
        encoded_peptide_representation = encode_sequence(peptide,protein_sequence,position_of_site)
        combined_representation = np.concatenate([encoded_peptide_representation, np.array(list(ecfp4_fingerprint), dtype=int)])
        predict_data.append(combined_representation)
    
    pr = np.array(predict_data)
    combined_shape = pr.shape[1]
    predict_data_reshaped = pr.reshape(-1, 1, combined_shape)
    predictions = model.predict(predict_data_reshaped)
    predictions_inverse = scaler.inverse_transform(predictions)
    print('Done')

if __name__ == "__main__":
    model = '/lustre/scratch123/hgi/projects/huvec/analysis/ml/qstar/output_model_One_Bi-LSTM_epochs10_all_data_0.2_0.001_0.001_32'
    predicting_tuple = ('CL1','GSTO1_HUMAN')
    
    predict_protein(predicting_tuple,model)