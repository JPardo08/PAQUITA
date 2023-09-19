# PAQUITA – SPANISH: Medical NER and RE Pipeline with SpaCy and Hugging Face

## PREPROCESSING DATA FOR RE y NER
./preprocessing.ipynb
./preprocessing.py

## CREATING DATA FILES FOR RE
./parse_data.py


## RAW DATA
Spanish Testlink:
- Tokenizados: ./Spanish/data/testlink/training/TESTLINK_ES_training_data_V2/training_tokenized
- Entidades: ./Spanish/data/testlink/training/TESTLINK_ES_training_data_V2/TESTLINK_training_data/training.txt


## PREPROCESSED DATA
./preprocessed


## MODEL CONFIGS
./configs

## TRANSFORMERS MODELS 
Download from the right path and put 'model' file into the left path:

./results_e3_ner/model-best/transformer/model -> https://drive.upm.es/s/xqcWvdFmW0TLKU1
./results_e3_ner/model-last/transformer/model -> https://drive.upm.es/s/sP9eMiWxktDgJlz

./results_e3_re/model-best/transformer/model -> https://drive.upm.es/s/V4g6bHfMG48T0J0
./results_e3_re/model-last/transformer/model -> https://drive.upm.es/s/v5mrEHH1BJxjnzi


## QUICKSTART:
1. Run preprocessing.ipynb. This is the preprocessing notebook for the data. It works for NER and RE. It outputs '.tsv' files for NER y '.txt' for RE.

2. Run parse_data.py to extract the '.spacy' data for RE.

3. Upload the NER '.tsv' files and RE '.spacy' files to Drive, located in the preprocessed folder.

4. Upload the NER '.cfg' files and RE '.cfg' files to Drive, located in the configs folder.

5. Execute paquita_exec.ipynb, which performs the training and deployment of NER and RE. Here is important to have the '.cfg' file for RE and NER. 



## LICENSE
PAQUITA is released under the MIT License.
