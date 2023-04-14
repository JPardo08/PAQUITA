# PAQUITA: Medical NER and RE Pipeline with SpaCy and Hugging Face
PAQUITA is an open-source project that provides a powerful and efficient pipeline for Named Entity Recognition (NER) and Relation Extraction (RE) in the medical domain. Built using the SpaCy and Hugging Face libraries, PAQUITA is designed to process and extract valuable information from medical text data such as electronic health records (EHRs) and clinical notes. This enables advanced data analysis, knowledge discovery, and improvements in patient care.

## FEATURES
- *Named Entity Recognition (NER):* Identifies medical entities such as diseases, medications, symptoms, and procedures within the text.
- *Relation Extraction (RE):* Detects and classifies the relationships between the identified medical entities, enabling the understanding of complex medical scenarios.
- *SpaCy Integration:* Leverages SpaCy's powerful NLP capabilities for fast and efficient text processing and entity recognition.
- *Hugging Face Transformers:* Utilizes state-of-the-art transformer models for relation extraction, ensuring high-performance and accurate results.

## PREPROCESSING DATA FOR RE y NER
./preprocessing.ipynb

## CREATING DATA FILES FOR RE
./parse_data.py


## RAW DATA
Spanish Testlink:
- Tokenizados: ./testlink/TESTLINK_ES_training_data_V2/training_tokenized
- Entidades: ./data/testlink/TESTLINK_ES_training_data_V2/TESTLINK_training_data/training.txt


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
