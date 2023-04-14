00_README

2023-03-06

training.txt

Distribution Licence
All the data in this folder are available under the CC-BY-NC-4.0 licence.

The training.txt file contains the 81 documents of the Spanish training set for the TESTLINK task at IberLEF 2023. 
The  data are in the PubTator format, which consists of tab-delimited text file structured as follows:
<DOCID>|t|<TEXT>
<DOCID>  REL  <RML_START>-<RML_END>  <EVENT_START>-<EVENT_END>  <RML_TEXT>  <EVENT_TEXT>
<DOCID>  REL  <RML_START>-<RML_END>  <EVENT_START>-<EVENT_END>  <RML_TEXT>  <EVENT_TEXT>

<DOCID>|t|<TEXT>
<DOCID>  REL  <RML_START>-<RML_END>  <EVENT_START>-<EVENT_END>  <RML_TEXT>  <EVENT_TEXT>

Where:
Every document in the dataset is in a new line and a space line is used as a document separator.
- DOCID: document id
- t: marker to identify the lines that contain the text of the documents
- TEXT: text of the document
- Every annotated relation is in a separate line and is represented as an ordered pair of textual mentions (i.e. RML,EVENT). Each mention in the relationship is expressed by its start and end character offsets.  The text span can be set but is not mandatory.
- DOCID: document id
- REL: marker to identify the lines that contain the relations of the given  document
- RML_START:  start character offset of the RML entity mention in the document
- RML_END:  end character offset of the RML entity mention in the document
- EVENT_START:  start character offset of the EVENT entity mention in the document
- EVENT_END:  end character offset of the EVENT entity mention in the document
- RML_TEXT [optional]: text span of the RML entity mention
- EVENT_TEXT [optional]: text span of the EVENT entity mention

For example:
100001|t|Paciente de 65 a. de edad, que presentaba una elevación progresiva de las cifras de PSA desde 6 ng/ml a 12 ng/ml en el último año. Dicho paciente había sido sometido un año antes a una biopsia transrectal de próstata ecodirigida por sextantes que fue negativa.  Se decide, ante la elevación del PSA, realizar una E-RME previa a la 2ª biopsia transrectal, en la que se objetiva una lesión hipointensa que abarca zona central i periférica del ápex del lóbulo D prostático. El estudio espectroscópico de ésta lesión mostró una curva de colina discretamente más elevada que la curva de citrato, con un índice de Ch-Cr/Ci > 0,80, que sugería la presencia de lesión neoplásica, por lo que se biopsia dicha zona por ecografía transrectal. La AP de la biopsia confirmó la presencia de un ADK próstata Gleason 6.
100001	REL	94-101		84-87		6 ng/ml	PSA
100001	REL	104-112	84-87		12 ng/ml	PSA
100001	REL	251-259	185-192	negativa	biopsia
100001	REL	619-623	598-604	0,80		índice

Additional information for participants:

The annotation of pertains-to relations is strictly based on tokens (the target, in particular, must necessarily consist of one token, while the source consists of one or more token) and on sentences (source and target of a relation must belong to the same sentence).

In order to reduce the noise that might be introduced by mismatching pre-processing, participants to the TESTLINK task are strongly encouraged to use the same (automatic) tokenization and sentence splitting that has been used when the data have been (manually) annotated. This is provided in the file named training_tokenized.zip

