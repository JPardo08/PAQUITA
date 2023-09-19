00_README

2023-03-06

training.txt

Distribution Licence
All the data in this folder are available under the CC-BY-NC-4.0 licence.

The training.txt file contains the 91 documents of the Basque training set for the TESTLINK task at IberLEF 2023. 
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
100001|t|62 urteko gizona, 2 hilabetetako astenia, anemia, ikterizia eta hantura parametroen igoerarekin debutatu zuena. Egoera honekin, OTA eta gastro-kolonoskopia burutu zitzaizkion. OTAko emaitzak gibel abzesua eta pankreako minbizia iradoki zuen, masa irudi bat agertzen zelako. Horrela, kirurgia orokorrera bideratu zen. Kirurgia orokorrean egindako analitikan ez zen infekzio daturik objektibatu, baina baibilirrubinaren eta Ca 19.9-ren igoera, minbizia susmoa handiagotuz. Ondoren RMN, ekografia abdominala eta gibeleko lesioaren biopsia egin ziren. Biopsian, gibel-parenkima hanturazko infiltratu linfoplasmatikoz ordezkaturiko ehuna antzeman zen. Hantura infiltratuaz gain, gibel-parenkimaren fibrosia nabarmendu zen. Aztertutako zelula plasmatiko gehienak IgG positiboak zireneta IgG4 positiboko 30 zelula baino gehiago kontatu ziren kanpoko.    Ondoren errepikatutako analitikan IgG4 mailaren igoera ikusi zen: 459mg/Dl (N: 14-126).    Horrela, IgG4-EG diagnostikatu zitzaion, 1 motako pankreatitis autoinmune, kolangitis esklerosatzailea eta gibel masaren agerpenarekin. Glukokortikoideekin tratatu zen eta berehala hobekuntza klinikoa, zein irudi frogatakoa eman ziren.
100001	REL	761-771	757-760	positiboak	IgG
100001	REL	913-921	881-885	459mg/Dl	IgG4

Additional information for participants:

The annotation of pertains-to relations is strictly based on tokens (the target, in particular, must necessarily consist of one token, while the source consists of one or more token) and on sentences (source and target of a relation must belong to the same sentence).

In order to reduce the noise that might be introduced by mismatching pre-processing, participants to the Clinkart task are strongly encouraged to use the same (automatic) tokenization and sentence splitting that has been used when the data have been (manually) annotated. This is provided in the file named training_tokenized.zip

