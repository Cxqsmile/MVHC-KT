# (MVHC-KT) Multi-View Hypergraph-based Contrastive Learning for Knowledge Tracing

As a fundamental educational data mining task, knowledge tracing (KT) dynamically models learners' knowledge states through historical interaction data to predict 
future performance. Existing KT models that explore the relationship between questions and knowledge concepts face three key challenges:
(i) they typically explore question-concept relationships from a single perspective without modeling deeper question-question or concept-concept relationships independently; 
(ii) they rely on expert-annotated concept relationship graphs to explore the associations between concepts;
(iii) they lack comprehensive multi-perspective relationship modeling. 
To address these issues, this paper proposes Multi-View Hypergraph-based Contrastive Learning for Knowledge Tracing (MVHC-KT) by constructing three complementary views,
which enable the model to globally mine question-concept relationships from the dataset rather than relying on limited local associations: a question-concept hypergraph (q2c) 
to capture explicit associations, a dynamically learned concept co-occurrence graph (c2c) to remove dependency on expert annotations, and a question similarity hypergraph (q2q) 
to model implicit relationships among questions. We further employ cross-view contrastive learning to integrate complementary information from these views, enhancing the representations 
of questions and concepts and improving prediction accuracy. Extensive experiments on four benchmark datasets show that MVHC-KT significantly outperforms existing KT methods.

## 📁 Project Structure Description

- `train.py`: The main script for model training. 
- `test.py`: The script for model evaluation.  
- `model_cl.py`: The core model implementation, including the three-channel training networks, fusion module, contrastive learning components, and loss computation.
- `HG_dataset.py`: Data loading and data preparation before being fed into the model.  
- `early_stopping.py`: Implements the early stopping strategy during training.  
- `utils.py`: Contains various utility functions.  
- `data/`: Directory for data files. (Due to the large combined size of all four datasets, the submitted version includes only the raw Algebra 2005 dataset, 
the corresponding preprocessing code, and all generated input files required for the three channels—such as hypergraph incidence matrices and adjacency matrices. 
If you need to process the other three datasets, please download them from the link provided in the paper and run the preprocessing scripts to generate the necessary files.)  
- `logs/`: Directory for training logs and model checkpoints.  

## 🔧 Environment dependencies

- Python 3.12+
- PyTorch ≥ 2.4.0+cu124
- numpy 1.26.4
- scikit-learn 1.4.2
- tqdm 4.66.4
- pandas 2.2.2
- matplotlib 3.8.4

## 📌 Parameter/Hyperparameter Settings
emb_dim = 128              # Dimensions of problem and conceptual representation
correct_emb_dim = 16       # The representation dimension of the answer
ht_dim = 32                # The dimension of the out output by LSTM, which is the student feature representation vector
num_mv_layers = 1          # Number of layers in the first channel
num_gcn_layers = 3         # Number of layers in the second channel
num_question_layers = 6    # Number of layers in the third channel
num_lstm_layers = 3        # lstm layer
temperature = 0.5
dropout = 0.1
lambda_cl = 0.1            # Proportion and weight of loss in contrastive learning

num_epochs = 200
batch_size = 256

## 🚀 Model Training
run train.py  #To train on different datasets, please modify the values of 'num_question', 'num_skill', and 'num_student' according to the specific dataset. These values can be obtained from the output of the preprocessing script.
run test.py  ##To load the trained model, set model_filename = 'trained_model_algebra2005.pt'.

Note: You can also train on your dataset. Refer to the preprocessing code in 'data/algebra2005/algebra2005_dataprocess.ipynb' for guidance. By following a similar procedure, you can generate the required input files and use them directly for model training.

