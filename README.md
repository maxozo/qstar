
# QSAR ML Challenge
QSAR machine learning challenge

## Objective
Develop a machine learning model to predict the binding affinities of reactive fragments and protein targets.


## Description:

The ML challenge involves predicting the binding affinity for peptide sequences obtained through LC-MS tryptic digest, where lysine (K) and arginine (R) residues are present at the N and C termini. Two ML methods, namely RNN and CNN, were considered suitable for this challenge. RNN is well-suited for sequential data analysis, while CNN has previously demonstrated effectiveness in predicting protein cleavage affinities in enzymatic digests.

To address the issue of peptide length not necessarily being representative of binding affinities, the peptides were standardized to a fixed length defined as feature_length*2. This was achieved by querying the Uniprot sequence database to obtain all reviewed protein sequences available on 29/05/2013. The code iterates through each protein provided in competition-ratios.tsv and each compound in compounds.txt, extracting the peptides centered around the amino acid of interest indicated in competition-ratios.tsv.

For feature engineering, the amino acid sequences were encoded using one-hot encoding, with an additional positional argument for each amino acid. In the CNN model, one-hot encoding was represented as a matrix where each row corresponds to the one-hot encoding of a single amino acid.

Considering the potential importance of the peptides emitted by LC-MS, the peptide sequences listed in competition-ratios.tsv were also encoded using amino acid frequency encoding, along with the positional index within the protein sequence. Although the positional index may not be highly informative, it captures different functional domains present in proteins that may not be represented solely by the position within the sequence.

In addition to the sequence and amino acid frequency encoding, the ECFP_4 string was included as a feature, as specific side chains may chemically interact with the protein to facilitate binding.

The code was designed to allow for the evaluation of different learning rates, dropout rates, feature lengths, and L2 regularization methods.

The protein datasets can be predicted using the predict_protein.py code by providing the compound ID and protein ID as a tuple. This will encode the sequence accordingly and predict the binding affinities for the model.


## Discussion:

While the current model achieved an R2 score of approximately 0.24, indicating that only 24% of the data is explained by the model, there are several avenues to further improve its performance.

Firstly, leveraging domain knowledge about different compounds and their representative classes can be valuable. By training separate models for each class and utilizing transfer learning, it is possible to fine-tune the models specifically for a particular compound within a broader binding affinity class.

If domain knowledge is lacking, exploring clustering analysis of ECFP_4 signatures can be beneficial. This analysis can reveal compounds that exhibit similarity in terms of their chemical interactions, which can inform the creation of subgroups or allow for the development of compound-specific models.

In addition to ECFP_4 signatures, other chemical features or molecular descriptors can be incorporated into the model. These features could capture important characteristics of the compounds that influence their binding affinity, providing additional information to enhance the model's predictive capabilities.

Furthermore, reducing the dimensionality of the ECFP_4 signatures using techniques such as principal component analysis (PCA) or t-distributed stochastic neighbor embedding (t-SNE) may be effective. This reduction can help retain the most informative features while discarding redundant or noise-inducing components.

Consideration should also be given to incorporating additional biological information, such as secondary structure annotations, solvent accessibility, or physicochemical properties of amino acids. These features can provide valuable insights into the structural and functional characteristics of the peptides, potentially improving the model's ability to capture the underlying patterns and predict binding affinities more accurately.

Moreover, experimenting with different model architectures, such as stacked RNNs, attention mechanisms, or transformer models, may yield better results. These advanced architectures have demonstrated success in various sequence-related tasks and may capture more complex dependencies within the peptide sequences.

In conclusion, to improve the model's performance, a combination of domain knowledge, advanced feature engineering, incorporation of chemical and biological information, dimensionality reduction techniques, and exploration of different model architectures should be considered.



## competition_ration_model_Bi_LSTM_larger_encoding.py
    The script starts with some import statements to import necessary libraries and modules such as TensorFlow, scikit-learn, NumPy, pandas, etc.

    The script defines a function called get_protein_sequence that takes a UniProt ID as input and retrieves the corresponding protein sequence using either a preloaded dataset or by making a web request to the UniProt website.

    Another function called AA_Frequencies is defined, which calculates the amino acid frequencies for a given peptide sequence and position of the site.

    The encode_sequence function encodes a peptide sequence, protein sequence, and position of the site into a numerical representation using one-hot encoding and positional encoding.

    The main function train_model is defined, which performs the training of the model. It loads data from various files including compound information, competition ratios, protein sequences, etc. Then it iterates over the data to prepare the combined representations of the encoded peptide sequences, ECFP4 fingerprints, and other features. These combined representations and corresponding competition ratios are stored in a list called combined_data.

    The combined_data list is shuffled and split into training and testing sets. The input features (combined representations) and target values (competition ratios) are extracted from the split data.

    The script applies standardization to the target values using scikit-learn's StandardScaler to normalize the data.

    The input features are reshaped to match the LSTM input shape.

    The model architecture is defined using TensorFlow's Keras API. It consists of a bidirectional LSTM layer followed by dense layers and dropout layers.

    The model is compiled with the Adam optimizer and mean squared error loss function.

    Finally, the model is trained using the training data and the training history is stored in the history variable.



## Dataset
The dataset is a subset of [Kuljanin et al.](https://www.nature.com/articles/s41587-020-00778-3) and consists of a set of reactive fragments and their corresponding binding affinities to specific protein targets (Dataset9):
- For each reactive fragment, we provide SMILES string and ECFP4 feature
- For each protein target, we provide Uniprot ID, gene symbol, and the binding site position
- Binding affinity is provided as competition ratios

### Dataset file description
`competition-ratios.txt`
| Column | Description |
| --- | --- |
| Uniprot ID | Uniprot ID corresponding to each proteins |
| Gene Symbol | Official Gene Symbol corresponding to each protein |
| Site Position | Position of the cysteine modification (DBIA) found within each proteinPosition of the cysteine modification (DBIA) found within each protein |
| Peptide Seqence | Peptide sequence used to determine position of cysteine residue |
| Gene + Site | Official Gene Symbol corresponding to each protein plus position of modified cysteine residue |
| CL1 - AC157 | Competition ratio of reactive fragments CL1 - AC157 (S/N DMSO divided by 25 μM electrophile fragment CL1 to AC160 treated) |

`compounds.txt`
| Column | Description |
| --- | --- |
| Compound | Reactive fragment ID same as column CL1 - AC157 in `competition-ratios.txt` |
| Name | Name of reactive fragment following IUPAC nomenclature |
| SMILES | Correspondinng SMILES of reactive fragment |
| ECFP4 | Corresponding ECFP4 features |


# qstar
