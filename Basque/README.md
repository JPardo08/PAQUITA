# PAQUITA – BASQUE: Entity and Relation Extraction from Electronic Health Records using SpaCy.

## PREPROCESSING DATA FOR RE y NER
./preprocessing.ipynb


## CREATING DATA FILES FOR RE
./parse_data.py


## RAW DATA
Basque Testlink:
- Tokenized: ./Basque/data/testlink/training/training_tokenized
- Entities: ./Basque/data/testlink/training/training.txt


## PREPROCESSED DATA
./preprocessed


## MODEL CONFIGS
./configs


## TRANSFORMERS MODELS 
Download from the right path and put 'model' file into the left path:

- ./results_basque_ner/model-best/transformer/model -> https://drive.upm.es/s/NWwKJCu7idGqn1k
- ./results_basque_ner/model-last/transformer/model -> https://drive.upm.es/s/BxI9SPTKH5z4CGJ

- ./results_basque_re/model-best/transformer/model -> https://drive.upm.es/s/gkax2btzUzeElsI
- ./results_basque_re/model-last/transformer/model -> https://drive.upm.es/s/C8aP21dBpfSZwxN


## QUICKSTART:
### B1: PREPROCESSING

1. Ensure all data is stored in the **data** folder.
  
2. Execute `preprocessing.ipynb`, the preprocessing notebook for data preparation. It handles both NER and RE tasks, outputting `.tsv` files for NER and `.json` files for RE.
  
3. Execute `parse_data.py` to convert `.json` files into `.spacy` files for RE.

### B2: PREPARATION

1. Upload the NER `.tsv` files and RE `.spacy` files from the **preprocessed** folder to Google Drive. Then, open the `paquita_exec.ipynb` notebook with Google Colab.
  
2. Transfer the NER `.cfg` and RE `.cfg` files from the **configs** folder to Google Drive. Also, upload the `project.yml` that contains the specified metadata for training.
  
3. Run `paquita_exec.ipynb`. This notebook handles the training and deployment of both NER and RE tasks. Ensure the corresponding `.cfg` files for RE and NER are available. The output comprises two models for each task: 
   - **model_best**: the model showcasing the optimal performance throughout the training. 
   - **model_last**: the model reflecting the final trained weight.

> **NOTE:** Each model has two components: the model definition stored in the repository and the configuration that outlines the training process.


### B3: DEPLOYMENT

1. Retrieve the models from Google Drive and place them in the directory: `results_basque_{ner/re}`.

2. Navigate to the project directory (**PAQUITA**) using a terminal. Execute the command `streamlit run ./interface.py`. If you encounter issues, adjust the path to accurately point to `interface.py`. This command launches the project's interface with Streamlit, where you can review experimental results based on the input provided.

> **NOTE:** The `interface.py` file relies directly on `utils_paquita.py`. Both `rel_model.py` and `rel_pipe.py` are reference files. When `paquita_exec.ipynb` is executed, additional projects will be fetched into the local virtual environment.



## LICENSE
PAQUITA is released under the MIT License.
