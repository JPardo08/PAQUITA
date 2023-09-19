# PAQUITA – BASQUE: Medical NER and RE Pipeline with SpaCy and Hugging Face

## PREPROCESSING DATA FOR RE y NER
./preprocessing.ipynb
./preprocessing.py

## CREATING DATA FILES FOR RE
./parse_data.py


## RAW DATA
Basque Testlink:
- Tokenizados: ./Basque/data/testlink/training/training_tokenized
- Entidades: ./Basque/data/testlink/training/training.txt


## PREPROCESSED DATA
./preprocessed


## MODEL CONFIGS
./configs


## QUICKSTART:
1. Run preprocessing.ipynb. This is the preprocessing notebook for the data. It works for NER and RE. It outputs '.tsv' files for NER y '.txt' for RE.

2. Run parse_data.py to extract the '.spacy' data for RE.

3. Upload the NER '.tsv' files and RE '.spacy' files to Drive, located in the preprocessed folder.

4. Upload the NER '.cfg' files and RE '.cfg' files to Drive, located in the configs folder.

5. Execute paquita_exec.ipynb, which performs the training and deployment of NER and RE. Here is important to have the '.cfg' file for RE and NER. 



## LICENSE
PAQUITA is released under the MIT License.
