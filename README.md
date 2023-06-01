
# QSAR ML Challenge
QSAR machine learning challenge

## Objective
Develop a machine learning model to predict the binding affinities of reactive fragments and protein targets.

## Discussion:

While the current model achieved an R2 score of approximately 0.24, indicating that only 24% of the data is explained by the model, there are several avenues to further improve its performance.

Firstly, leveraging domain knowledge about different compounds and their representative classes can be valuable. By training separate models for each class and utilizing transfer learning, it is possible to fine-tune the models specifically for a particular compound within a broader binding affinity class.

If domain knowledge is lacking, exploring clustering analysis of ECFP_4 signatures can be beneficial. This analysis can reveal compounds that exhibit similarity in terms of their chemical interactions, which can inform the creation of subgroups or allow for the development of compound-specific models.

In addition to ECFP_4 signatures, other chemical features or molecular descriptors can be incorporated into the model. These features could capture important characteristics of the compounds that influence their binding affinity, providing additional information to enhance the model's predictive capabilities.

Furthermore, reducing the dimensionality of the ECFP_4 signatures using techniques such as principal component analysis (PCA) or t-distributed stochastic neighbor embedding (t-SNE) may be effective. This reduction can help retain the most informative features while discarding redundant or noise-inducing components.

Consideration should also be given to incorporating additional biological information, such as secondary structure annotations, solvent accessibility, or physicochemical properties of amino acids. These features can provide valuable insights into the structural and functional characteristics of the peptides, potentially improving the model's ability to capture the underlying patterns and predict binding affinities more accurately.

Moreover, experimenting with different model architectures, such as stacked RNNs, attention mechanisms, or transformer models, may yield better results. These advanced architectures have demonstrated success in various sequence-related tasks and may capture more complex dependencies within the peptide sequences.

In conclusion, to improve the model's performance, a combination of domain knowledge, advanced feature engineering, incorporation of chemical and biological information, dimensionality reduction techniques, and exploration of different model architectures should be considered.

## Description:

The ML challenge involves predicting the binding affinity for peptide sequences obtained through LC-MS tryptic digest, where lysine (K) and arginine (R) residues are present at the N and C termini. Two ML methods, namely RNN and CNN, were considered suitable for this challenge. RNN is well-suited for sequential data analysis, while both CNN and RNN has previously demonstrated effectiveness in predicting protein cleavage affinities in enzymatic digests, assumtion was made that similar approach can be used for compound competition ratio predictions.

Two arhitectures that were considered are: 
### RNN
The model described here is a sequential neural network architecture used for regression tasks. It consists of multiple layers, including a bidirectional LSTM layer, dense layers, and dropout layers.

Here's a breakdown of the model architecture and its components:

The model starts with a Sequential() function call, indicating that the layers will be added sequentially.

The first layer added to the model is a bidirectional LSTM layer. The LSTM layer is a type of recurrent neural network (RNN) that can process sequential data. The bidirectional aspect allows the LSTM to consider both past and future context when making predictions.

The LSTM layer has 254 units and uses the ReLU activation function. The recurrent dropout parameter is applied to the LSTM layer, which helps regularize the model and prevent overfitting.

A dense layer with 56 units follows the LSTM layer. Dense layers are fully connected layers where each neuron is connected to every neuron in the previous layer. The ReLU activation function is used for this dense layer.

A dropout layer is added after the dense layer. Dropout randomly sets a fraction of input units to 0 at each update during training, which helps prevent overfitting by introducing redundancy and reducing interdependencies between neurons.

Another dense layer with a single unit is added, representing the output layer of the model.

The model is compiled using the mean squared error (MSE) loss function and the Adam optimizer. MSE is commonly used for regression tasks as it measures the average squared difference between the predicted and true values.

The model is then trained using the fit() function. The training data X_train_reshaped and target values Y_train_scaled are provided. The training is performed for 70 epochs with a specified batch size. Additionally, a validation split of 0.2 is used, meaning 20% of the training data is reserved for validation during training.

The early_stopping callback is applied to monitor the validation loss and stop training early if there is no improvement.

In the field of proteomics, similar approaches involving deep learning and neural networks have been used for various tasks. For example, recurrent neural networks (RNNs), including LSTM layers, have been utilized for protein sequence analysis, such as protein classification, prediction of protein properties, and protein-protein interaction prediction. The bidirectional aspect of LSTMs helps capture dependencies in both directions and improves the modeling of sequential data. Additionally, the use of dense layers and dropout regularization aids in learning complex patterns and preventing overfitting in proteomic data analysis. Overall, deep learning models like the one described have shown promising results in proteomics research by leveraging the inherent sequential nature of protein data.

### CNN
Early Stopping: The model uses early stopping as a regularization technique to prevent overfitting. It monitors the validation loss during training and stops the training process if the loss does not improve for a certain number of epochs (defined by the patience parameter).

Model Definition: The model architecture is defined using Keras, a high-level neural networks API. The model has two types of input data: image data and auxiliary data. The image input shape is (20, 22, 1), indicating an input image with dimensions of 20x22 pixels and a single channel. The auxiliary input shape is determined by the shape of aux_data_test.

Convolutional Neural Network (CNN): The model starts with a convolutional layer (Conv2D) with 32 filters, a filter size of (3, 3), and a 'relu' activation function. This layer performs feature extraction on the image input. Subsequently, a max pooling layer (MaxPooling2D) with a pool size of (2, 2) reduces the spatial dimensions. The output is then flattened (Flatten) to be concatenated with the auxiliary input.

Concatenation: The flattened output from the CNN and the auxiliary input are concatenated (Concatenate) to combine the extracted image features with the auxiliary data.

Dense Layers: The concatenated output is passed through a dense layer (Dense) with 64 units and a 'relu' activation function. Regularization is applied using the kernel_regularizer parameter with an L2 regularization term (l2). Dropout regularization is also applied after the dense layer with a specified dropout rate.

Output Layer: The output of the dropout layer is fed into a final dense layer with a single unit and a linear activation function. This layer produces the regression output.

Model Compilation: The model is compiled with the Adam optimizer, which uses adaptive learning rates, and mean squared error (mse) is used as the loss function. Mean absolute error (mae) is used as an additional metric to evaluate the model's performance.

Model Training: The model is trained (model.fit) using the training data (X_train and aux_data_train) and corresponding scaled target values (Y_train_scaled). The training is performed for a specified number of epochs (100) with a defined batch size. A validation split of 0.2 is used for monitoring the validation loss during training. The early_stopping callback is used to stop training early if the validation loss does not improve.

Overall, this model architecture combines a CNN for image feature extraction with auxiliary data to perform regression. It leverages convolutional and dense layers to learn representations from the image input and auxiliary input, respectively, and produces a regression output.

Similar approaches combining CNNs and auxiliary data have been used in the field of proteomics for various tasks. For example, in protein structure prediction, CNNs have been employed to analyze protein sequences or structural motifs, while auxiliary data such as evolutionary conservation scores or physicochemical properties have been used to enrich the model's understanding of protein structure. The combination of image-based features and auxiliary data allows for a more comprehensive representation of the proteins and can improve predictive performance in various proteomic applications.


To address the issue of peptide length not necessarily being representative of binding affinities, the peptides were standardized to a fixed length defined as feature_length*2. This was achieved by querying the Uniprot sequence database to obtain all reviewed protein sequences available on 29/05/2013. The code iterates through each protein provided in competition-ratios.tsv and each compound in compounds.txt, extracting the peptides centered around the amino acid of interest indicated in competition-ratios.tsv.

For feature engineering, the amino acid sequences were encoded using one-hot encoding, with an additional positional argument for each amino acid. In the CNN model, one-hot encoding was represented as a matrix where each row corresponds to the one-hot encoding of a single amino acid.

Considering the potential importance of the peptides emitted by LC-MS, the peptide sequences listed in competition-ratios.tsv were also encoded using amino acid frequency encoding, along with the positional index within the protein sequence. Although the positional index may not be highly informative, it captures different functional domains present in proteins that may not be represented solely by the position within the sequence.

In addition to the sequence and amino acid frequency encoding, the ECFP_4 string was included as a feature, as specific side chains may chemically interact with the protein to facilitate binding.

The code was designed to allow for the evaluation of different learning rates, dropout rates, feature lengths, and L2 regularization methods.

The protein datasets can be predicted using the predict_protein.py code by providing the compound ID and protein ID as a tuple. This will encode the sequence accordingly and predict the binding affinities for the model.

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
