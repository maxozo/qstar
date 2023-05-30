
# QSAR ML Challenge
QSAR machine learning challenge

## Objective
Develop a machine learning model to predict the binding affinities of reactive fragments and protein targets.

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

## Task
Your task is to develop a machine learning model that can accurately predict the competition ratio of the tuple (reactive fragment, target protein). You can use any machine learning algorithm or approach that you prefer.

## Deliverables
- A Python package contains your data preprocessing, feature engineering, and machine learning model implementation.
- A short write up that explains your general approach, discusses any insights on data splitting, and presents the results of your model evaluation
# qstar
