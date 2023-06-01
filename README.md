
# QSAR ML Challenge
QSAR machine learning challenge

## Objective
Develop a machine learning model to predict the binding affinities of reactive fragments and protein targets.


## Description
The particular ML chalange contains peptide sequnces acquired by LC-MS tryptic digest (clear as K and R residues are present at N and C termini). 
Two ML methods were considered as suitable for the particular challange - RNN method since we are dealing with sequencial data, and CNN which has been demonstrated previously to be capable in efectivelly pedict protein cleavage affinities enzymatic digest.

Since peptide lenghts are not necassarally a feature that is representative of the binding affinities peptides were standardised to a set lenght (defined as a feature_length*2). This was achieved by querying uniprot sequence database. To avoid data updates all reviewed protein sequences were retrieved (on 29/05/2013). Code iterates through each of the proteins provided in competition-ratios.tsv and each of the compounds in compounds.txt and extracts the peptides centered around amino acid of interest indicated in competition-ratios.tsv.

For feature engineering these amiono acid sequences were then encoded sequentially utilising one hot encoding of a sequence providing aditional positional argument for each amino acid sequence. For CNN model One-hot encoding was provided as a matrix where each row represents a single amino acid one hot encoding.

Since there is a possibility that the peptides emited by LC-MS may be important too, the peptide sequences listed in competition-ratios.tsv were also encoded in amino acid frequency encoding alongside the positional index within protein sequence (while positional index may not be important, since proteins in general has a lot of different functional domains that are not necessarally represented by the position within sequence). 

Since these competition ratios are dependant on the compound, alongside sequence and the amino acid frequency encoding we also provided ECFP_4 string, since there may be chemical interactions of specific side chains interacting with the protein to have a binding.

Code was writen in a way that different learning rates, dropout rates, feature_lengths, l2_regularisation methods can be evaluated.


Now protein datasets can be predicted using the predict_protein.py code, where users provide the compound id and protein id as a tuple. This will encode the sequence accordingly and predict the binding affinities for the model. 

## Discussion
Unfortunatelly further feature engineering and testing is necessary since the best performing model only had an R2~0.24, meaning that only 24% of data is explained. 

The steps to further take would be to either use domain knowlage about different compounds and whether there are any representative classes and train seperate models for these, and utilise transfer learning to fine tune the model towards a specific compound within wider binding affinity class. Ifd no domain knowlage is awailable, a clustering analysis of ECFP_4 signatures may be utilised to see which compounds exibit similarity.

Subsequently also the ECFP_4 signatures may be reduced to lover dimensions using dimensionality reduction techniques such as PCS, TSNE ...
Additional features such as secondary structure information, domain exposure and protein disordered region information may be provided as an aditional features in training model to improve the accuracy.  



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
