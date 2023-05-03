# -*- coding: utf-8 -*-
"""preprocessing.ipynb

# PREPROCESSING CHALLENGE - ITALIAN
"""


import os
import pandas as pd
import re
import string
from unidecode import unidecode
from sklearn.model_selection import train_test_split

# MENU
print('Processing training documents...')

# Introduce paths in order to process training or testing.
path = "./Basque/data/testlink/training/training.txt"
path_token = "./Basque/data/testlink/training/training_tokenized"





# Load data
with open(path, 'r') as f:
    data = f.readlines()
# Note: In train we have text join with relations. In test we only have texts


# List '.tsv' files
archivos = os.listdir(path_token)



c = 0
for i in data:
    a = '|t|' in i # Separador
    if a:
        c += 1


print('There are', c, 'documents for training.') # Numero de documentos




"""## EXTRACCION DE DOCUMENTOS, TOKENIZADOS Y REGISTROS"""

text_flag = []
data_raw = []
regis_raw = []

# patron = r"\d{4}-\d{4}"

for i in data: # text lines

    if '|t|' in i: # Plain text
        text_flag.append(True)
        data_raw.append(i)

    else: # Individual line
        text_flag.append(False)
        
        sep = i.split(' ') 
        sep1 = [j.split('\t') for j in sep]
        # print(sep1)

        
        
        sep2 = []
        for v in sep1:
            sep2.extend(v)

        regis_raw.append(sep2)

tokenizados = pd.DataFrame()
# tokenizados['DOCID'] = []


for i in archivos:
    docid = i.split('.')[0]

    ruta = path_token+'/'+i
    a = pd.read_csv(ruta, sep="\t", header=None)#, engine='python', encoding='utf-8', error_bad_lines=False)
    a['DOCID'] = docid
    # display(a)
    tokenizados = pd.concat([tokenizados,a])

tokenizados.columns = ['0', '1', '2', 'DOCID']

# display(tokenizados)

# len(tokenizados['DOCID'].unique().tolist())

tokenizados[['phrase', 'id']] = tokenizados['0'].str.split('-', expand=True)

tokenizados[['start', 'end']] = tokenizados['1'].str.split('-', expand=True)

tokenizados['text'] = tokenizados['2']
tokenizados = tokenizados.drop(['0', '1', '2'], axis=1)
# display(tokenizados)

"""### TEXTO"""

data_raw

# print(len(data_raw))

DOCID_txt = []
TEXT = []

for i in data_raw:
    sep = i.split('|t|')
    DOCID_txt.append(sep[0].replace('\n', ''))
    TEXT.append(sep[1].replace('\n', ''))

# print(len(DOCID_txt))
# print(len(TEXT))

df_text = pd.DataFrame()

df_text['DOCID'] = DOCID_txt
df_text['TEXT'] = TEXT

# display(df_text)

"""### REGISTRO"""

# regis_raw

DOCID = []
REL = []
RML = []
EVENT = []
RML_TEXT = []
EVENT_TEXT = []

for r in regis_raw:

    long = len(r)

    c = 0
    for h in r:
        if c == 0:
            if h != '\n':
                DOCID.append(h.replace('\n', ''))
            # print(c, h)
        if c == 1:
            REL.append(h)
            # print(c, h)
        if c == 2:
            RML.append(h)
            # print(c, h)
        if c == 3:
            EVENT.append(h)
            # print(c, h)
        if c == 4:
            RML_TEXT.append(' '.join(r[4:long-1]))
            # print(c, ' '.join(r[4:long-1]))
        if c == 5:
            EVENT_TEXT.append(r[long-1:][0].replace('\n', ''))
            # print(c, r[long-1:][0])
    
        c+=1

# print(len(DOCID))
# print(len(REL))
# print(len(RML))
# print(len(EVENT))
# print(len(RML_TEXT))
# print(len(EVENT_TEXT))

df = pd.DataFrame()

df['DOCID']=DOCID
df['REL']=REL
df['RML']=RML
df['EVENT']=EVENT
df['RML_TEXT']=RML_TEXT
df['EVENT_TEXT']=EVENT_TEXT

# display(df)

"""### UNION DE LOS COJUNTOS DE LOS REGISTROS TEXTO Y REGISTRO"""

DOCID_uni = []
[DOCID_uni.append(a) for a in DOCID if a not in DOCID_uni]

DOCID2 = tokenizados['DOCID'].tolist()

DOCID_uni2 = []
[DOCID_uni2.append(a) for a in DOCID2 if a not in DOCID_uni2]

DOCID_exc = []
[DOCID_exc.append(a) for a in DOCID_uni2 if a not in DOCID_uni]






# print(DOCID_uni)
# print(len(DOCID_uni))

# print(DOCID_uni2)
# print(len(DOCID_uni2))

# print(DOCID_exc)
# print(len(DOCID_exc))

df_general = pd.merge(df, df_text, 
         how='inner', on='DOCID')
# display(df_general)

df_general[['RML_start', 'RML_end']] = df_general['RML'].str.split('-', expand=True)
df_general[['EVENT_start', 'EVENT_end']] = df_general['EVENT'].str.split('-', expand=True)

df_general = df_general.drop(['RML', 'EVENT'], axis=1)

# display(df_general)

DOCID_uni = df['DOCID'].unique()

# print(DOCID_uni)
# print(len(DOCID_uni))

train, test = train_test_split(DOCID_uni, test_size=0.33, random_state=42, shuffle=True)
# print("Ejemplos usados para entrenar y validar: ", len(train))
# print("Ejemplos usados para test: ", len(test))

train, dev = train_test_split(train, test_size=0.2, random_state=42, shuffle=True)
# print("Ejemplos usados para entrenar: ", len(train))
# print("Ejemplos usados para validar: ", len(dev))

train=train.tolist()
test=test.tolist()
dev=dev.tolist()

# print(train)
# print(test)
# print(dev)

df_prueba = pd.DataFrame()

df_prueba['DOCID'] = train+test+dev

train = [a+'.txt' for a in train]
test = [a+'-3.txt' for a in test]
dev = [a+'-4.txt' for a in dev]

# print(train)
# print(test)
# print(dev)

df_prueba['SCRIPTS'] = train+test+dev
df_prueba['LABEL'] = ['train']*len(train)+['test']*len(test)+['dev']*len(dev)

# display(df_prueba)

# df_prueba['SCRIPTS'].tolist()

df_general = pd.merge(df_general, df_prueba,
         how='inner', on='DOCID')
# display(df_general)

# df_general['SCRIPTS'].tolist()

"""## FORMAT RELCOM_COMPONENT"""

# Separamos los cojuntos de train, test, dev

train_df = df_general[df_general['LABEL']=='train']
test_df = df_general[df_general['LABEL']=='test']
dev_df = df_general[df_general['LABEL']=='dev']

# display(train_df)
# display(test_df)
# display(dev_df)

# !pip3 install spacy
# !python -m spacy download es_core_news_lg

# import spacy
# import re
# from spacy.tokens import Doc

# def custom_tokenizer(nlp):
#     # Crea una expresión regular para números con comas decimales
#     re_number = re.compile(r"\d+(,\d+)?")
    
#     # Obtiene el tokenizador predeterminado del modelo de lenguaje
#     tokenizer = nlp.tokenizer
    
#     # Define una función para manejar los casos personalizados
#     def custom_cases(text):
#         # Tokeniza el texto utilizando el tokenizador predeterminado
#         doc = tokenizer(text)
        
#         # Crea una lista vacía para almacenar los nuevos tokens
#         new_tokens = []
        
#         index = 0
#         while index < len(doc):
#             token = doc[index]
#             if re_number.match(token.text):
#                 # Si el token coincide con la expresión regular,
#                 # agrega el token a la lista de nuevos tokens sin dividirlo en subtokens
#                 new_tokens.append(token)
#             else:
#                 # Si el token no coincide con la expresión regular,
#                 # divide su texto en subtokens utilizando el carácter de espacio como separador y crea un nuevo objeto Token para cada subtoken
#                 for subtoken in token.text.split():
#                     new_token = doc.vocab[subtoken]
#                     new_tokens.append(new_token)
#             index += 1
        
#         # Crea un nuevo objeto Doc con los nuevos tokens y devuelve el resultado
#         return Doc(doc.vocab, words=[t.text for t in new_tokens])
    
#     return custom_cases

# # Carga el modelo de lenguaje y establece la función personalizada como tokenizador
# nlp = spacy.load('es_core_news_lg')
# nlp.tokenizer = custom_tokenizer(nlp)

"""Seleccionamos los tokenizados que estan dentro del cojunto de documentos de entrenamiento"""

tokenizados_df = tokenizados[tokenizados['DOCID'].isin(DOCID_uni)].sort_values(by=['DOCID', 'phrase', 'id'])
tokenizados_df['id'] = tokenizados_df['id'].astype(int) - 1

# display(tokenizados_df)

"""Combinamos los tokenizados seleccionados con los spans

train
"""

rml_df0 = train_df[['DOCID', 'RML_TEXT','RML_start', 'RML_end','EVENT_start', 'EVENT_end']]

rml_df0.columns=['DOCID', 'text','start', 'end','start_rel', 'end_rel']
# display(rml_df0)



event_df0 = train_df[['DOCID', 'EVENT_TEXT','EVENT_start', 'EVENT_end']]

event_df0.columns=['DOCID', 'text','start', 'end']
# display(event_df0)

"""test"""

rml_df1 = test_df[['DOCID', 'RML_TEXT','RML_start', 'RML_end','EVENT_start', 'EVENT_end']]

rml_df1.columns=['DOCID', 'text','start', 'end','start_rel', 'end_rel']
# display(rml_df1)



event_df1 = test_df[['DOCID', 'EVENT_TEXT','EVENT_start', 'EVENT_end']]

event_df1.columns=['DOCID', 'text','start', 'end']
# display(event_df1)

"""dev"""

rml_df2 = dev_df[['DOCID', 'RML_TEXT','RML_start', 'RML_end','EVENT_start', 'EVENT_end']]

rml_df2.columns=['DOCID', 'text','start', 'end','start_rel', 'end_rel']
# display(rml_df2)



event_df2 = dev_df[['DOCID', 'EVENT_TEXT','EVENT_start', 'EVENT_end']]

event_df2.columns=['DOCID', 'text','start', 'end']
# display(event_df2)

"""Funcion para ver sobre los tokenizados, quien es una entidad"""

df_frases = tokenizados_df[tokenizados_df['DOCID']=='100042'].copy()
df_spans = rml_df0[rml_df0['DOCID']=='100042'].copy()
df_spans1= event_df0[event_df0['DOCID']=='100042'].copy()


df_frases['start'] = pd.to_numeric(df_frases['start'])
df_frases['end'] = pd.to_numeric(df_frases['end'])


df_spans['start'] = pd.to_numeric(df_spans['start'])
df_spans['end'] = pd.to_numeric(df_spans['end'])

df_spans['start_rel'] = pd.to_numeric(df_spans['start_rel'])
df_spans['end_rel'] = pd.to_numeric(df_spans['end_rel'])


df_spans1['start'] = pd.to_numeric(df_spans1['start'])
df_spans1['end'] = pd.to_numeric(df_spans1['end'])


# df_frases['disabled'] = False
df_frases['type'] = 'PLAIN'
df_frases['child'] = None
df_frases['start_rel'] = None
df_frases['end_rel'] = None


for index, row in df_spans.iterrows():
    start = row['start']
    end = row['end']

    start_rel = row['start_rel']
    end_rel = row['end_rel']

    sel = df_frases[(df_frases['start'] >= start) & (df_frases['start'] <= end)]
    
    valor_id = df_frases.loc[(df_frases['start'] >= start_rel) & (df_frases['end'] <= end_rel), 'id']#.values[0]
    # print(valor_id)
    valor_start = df_frases.loc[(df_frases['start'] >= start_rel) & (df_frases['end'] <= end_rel), 'start']#.values[0]
    # print(valor_start)
    valor_end = df_frases.loc[(df_frases['start'] >= start_rel) & (df_frases['end'] <= end_rel), 'end']#.values[0]
    # print(valor_end)
    
    
    
    # Crear una máscara booleana para identificar las filas donde la columna 'A' contiene un signo de puntuación
    mask = sel['text'].isin(list(['.',',']))

    # Eliminar las filas donde la columna 'A' contiene un signo de puntuación
    sel = sel[~mask]
    # display(sel)
    

    # df_frases.loc[sel.index, 'disabled'] = True
    df_frases.loc[sel.index, 'type'] = 'RML'
    df_frases.loc[sel.index, 'child'] = int(valor_id)
    df_frases.loc[sel.index, 'start_rel'] = int(valor_start)
    df_frases.loc[sel.index, 'end_rel'] = int(valor_end)
    
    # display(df_frases.loc[sel.index, ])


for index, row in df_spans1.iterrows():
    start = row['start']
    end = row['end']
    
    sel = df_frases[(df_frases['start'] >= start) & (df_frases['start'] <= end)]
    
    
    
    # Crear una máscara booleana para identificar las filas donde la columna 'A' contiene un signo de puntuación
    mask = sel['text'].isin(list(['.',',']))

    # Eliminar las filas donde la columna 'A' contiene un signo de puntuación
    sel = sel[~mask]
    # display(sel)

    # df_frases.loc[sel.index, 'disabled'] = True
    df_frases.loc[sel.index, 'type'] = 'EVENT'
    # display(df_frases.loc[sel.index, ])
    


# # print(df_frases['disabled'].value_counts())
# print(df_frases['type'].value_counts())
# print(df_frases['child'].value_counts())


# display(df_frases)

def trf(rml_df, tokenizados_dfa, event_df):
    
    # Crear dataframes de ejemplo
    docc = rml_df['DOCID'].unique().tolist()
    tokenizados_df1 = pd.DataFrame()

    for i in docc:
        df_frases = tokenizados_dfa[tokenizados_dfa['DOCID']==i].copy()
        df_spans = rml_df[rml_df['DOCID']==i].copy()
        df_spans1= event_df[event_df['DOCID']==i].copy()


        df_frases['start'] = pd.to_numeric(df_frases['start'])
        df_frases['end'] = pd.to_numeric(df_frases['end'])

        df_spans['start'] = pd.to_numeric(df_spans['start'])
        df_spans['end'] = pd.to_numeric(df_spans['end'])

        df_spans['start_rel'] = pd.to_numeric(df_spans['start_rel'])
        df_spans['end_rel'] = pd.to_numeric(df_spans['end_rel'])    

        df_spans1['start'] = pd.to_numeric(df_spans1['start'])
        df_spans1['end'] = pd.to_numeric(df_spans1['end'])


        # df_frases['disabled'] = False
        df_frases['type'] = 'PLAIN'
        df_frases['child'] = None
        df_frases['start_rel'] = None
        df_frases['end_rel'] = None

        for index, row in df_spans.iterrows():
            
            start = row['start']
            end = row['end']

            start_rel = row['start_rel']
            end_rel = row['end_rel']

            try:
                sel = df_frases[(df_frases['start'] >= start) & (df_frases['start'] <= end)]
                valor_id = df_frases.loc[(df_frases['start'] >= start_rel) & (df_frases['end'] <= end_rel), 'id'].values[0]
                valor_start = df_frases.loc[(df_frases['start'] >= start_rel) & (df_frases['end'] <= end_rel), 'start'].values[0]
                valor_end = df_frases.loc[(df_frases['start'] >= start_rel) & (df_frases['end'] <= end_rel), 'end'].values[0]
                
            except:
                pass
            
            # Crear una máscara booleana para identificar las filas donde la columna 'A' contiene un signo de puntuación
            mask = sel['text'].isin(list(['.',',']))

            # Eliminar las filas donde la columna 'A' contiene un signo de puntuación
            sel = sel[~mask]
            

            # df_frases.loc[sel.index, 'disabled'] = True
            df_frases.loc[sel.index, 'type'] = 'RML'
            df_frases.loc[sel.index, 'child'] = int(valor_id)
            df_frases.loc[sel.index, 'start_rel'] = int(valor_start)
            df_frases.loc[sel.index, 'end_rel'] = int(valor_end)



        for index, row in df_spans1.iterrows():
            start = row['start']
            end = row['end']
            # text= row['text']
            
            sel = df_frases[(df_frases['start'] >= start) & (df_frases['start'] <= end)]
            
            
            # Crear una máscara booleana para identificar las filas donde la columna 'A' contiene un signo de puntuación
            mask = sel['text'].isin(list(['.',',']))

            # Eliminar las filas donde la columna 'A' contiene un signo de puntuación
            sel = sel[~mask]

            # df_frases.loc[sel.index, 'disabled'] = True
            df_frases.loc[sel.index, 'type'] = 'EVENT'



        
        tokenizados_df1 = pd.concat([tokenizados_df1, df_frases])


    

    # display(tokenizados_df1)

    return tokenizados_df1

# for index, row in tokenizados_df1.iterrows():
#     print((row['start'],row['end'],row['text'], row['disabled']))

tokenizados_df_train = trf(rml_df0, tokenizados_df, event_df0)
tokenizados_df_test = trf(rml_df1, tokenizados_df, event_df1)
tokenizados_df_dev = trf(rml_df2, tokenizados_df, event_df2)

# display(tokenizados_df_train)
# display(tokenizados_df_test)
# display(tokenizados_df_dev)

# print(tokenizados_df_train['disabled'].value_counts())
# print(tokenizados_df_train['type'].value_counts())
# print(tokenizados_df_train['child'].value_counts())

# print(tokenizados_df_test['disabled'].value_counts())
# print(tokenizados_df_test['type'].value_counts())
# print(tokenizados_df_test['child'].value_counts())

# print(tokenizados_df_dev['disabled'].value_counts())
# print(tokenizados_df_dev['type'].value_counts())
# print(tokenizados_df_dev['child'].value_counts())

tokenizados_df_train['phrase'] = pd.to_numeric(tokenizados_df_train['phrase'])
tokenizados_df_train['id'] = pd.to_numeric(tokenizados_df_train['id'])
tokenizados_df_train=tokenizados_df_train.sort_values(by=['DOCID', 'start','phrase', 'id'], ascending=True)

tokenizados_df_test['phrase'] = pd.to_numeric(tokenizados_df_test['phrase'])
tokenizados_df_test['id'] = pd.to_numeric(tokenizados_df_test['id'])
tokenizados_df_test=tokenizados_df_test.sort_values(by=['DOCID', 'start','phrase', 'id'], ascending=True)

tokenizados_df_dev['phrase'] = pd.to_numeric(tokenizados_df_dev['phrase'])
tokenizados_df_dev['id'] = pd.to_numeric(tokenizados_df_dev['id'])
tokenizados_df_dev=tokenizados_df_dev.sort_values(by=['DOCID', 'start','phrase', 'id'], ascending=True)

# tokenizados_df_train=ordenar(tokenizados_df_train)
# tokenizados_df_test=ordenar(tokenizados_df_test)
# tokenizados_df_dev=ordenar(tokenizados_df_dev)


# display(tokenizados_df_train)
# display(tokenizados_df_test)
# display(tokenizados_df_dev)

"""Funcion para ver que los indices"""

def tarraco(tokenizados_df1):
    docid_unicos = tokenizados_df1['DOCID'].unique()
        # print(docid_unicos)

    docd_t =[]
    prh_t=[]
    typ_t=[]
    indices_inicio_t=[]
    indices_fin_t=[]
    indices_inicio_rel_t=[]
    indices_fin_rel_t=[]
    palabras_t=[]
    ids_t = []

    for i in docid_unicos:
        sel = tokenizados_df1[tokenizados_df1['DOCID'] == i]
        # display(sel)
        
        sel['phrase'] = pd.to_numeric(sel['phrase'])
        sel['start'] = pd.to_numeric(sel['start'])
        sel['end'] = pd.to_numeric(sel['end'])
        sel['id'] = pd.to_numeric(sel['id'])
        
        docd_t.append(i)

        # frases_uni = sorted(sel['phrase'].unique().tolist())
        frases_uni = sel['phrase'].unique()
        # print(frases_uni)

        
        prh_p=[]
        typ_p=[]
        indices_inicio_p=[]
        indices_fin_p=[]
        indices_inicio_rel_p=[]
        indices_fin_rel_p=[]
        ids_p = []
        palabras_p=[]
        
        for j in frases_uni:
            selec = sel[sel['phrase'] == j]
            
            typo=[]
            indices_inicio=[]
            indices_fin=[]
            indices_inicio_rel=[]
            indices_fin_rel=[]
            ids = []
            palabras=[]

            prh_p.append(j)
        
            for index, row in selec.iterrows():
                typ = row['type']
                start = row['start']
                end = row['end']
                start_re = row['start_rel']
                end_re = row['end_rel']
                id = row['id']
                palabra = row['text']

                if typ != 'PLAIN':
                    
                    typo.append(typ)
                    indices_inicio.append(start)
                    indices_fin.append(end)
                    indices_inicio_rel.append(start_re)
                    indices_fin_rel.append(end_re)
                    ids.append(id)
                    palabras.append(palabra)
                    
            typ_p.append(typo) 
            indices_inicio_p.append(indices_inicio)
            indices_fin_p.append(indices_fin)
            indices_inicio_rel_p.append(indices_inicio_rel)
            indices_fin_rel_p.append(indices_fin_rel)
            ids_p.append(ids)
            palabras_p.append(palabras)

        
        prh_t.append(prh_p)
        typ_t.append(typ_p)
        indices_inicio_t.append(indices_inicio_p)
        indices_fin_t.append(indices_fin_p)
        indices_inicio_rel_t.append(indices_inicio_p)
        indices_fin_rel_t.append(indices_fin_p)
        ids_t.append(ids_p)
        palabras_t.append(palabras_p)

    # print(docd_t)
    # print(indices_inicio_t)
    # print(indices_fin_t)
    # print(ids_t)
    # print(palabras_t)
    return docd_t, prh_t, typ_t, indices_inicio_t, indices_fin_t, indices_inicio_rel_t, indices_fin_rel_t, ids_t, palabras_t

docd_t, prh_t, typ_t, indices_inicio_t, indices_fin_t, indices_inicio_rel_t, indices_fin_rel_t, ids_t, palabras_t = tarraco(tokenizados_df_train)
docd_t1, prh_t1, typ_t1, indices_inicio_t1, indices_fin_t1, indices_inicio_rel_t1, indices_fin_rel_t1, ids_t1, palabras_t1 = tarraco(tokenizados_df_test)
docd_t2, prh_t2, typ_t2, indices_inicio_t2, indices_fin_t2, indices_inicio_rel_t2, indices_fin_rel_t2, ids_t2, palabras_t2 = tarraco(tokenizados_df_dev)

# print(docd_t)
# print(prh_t)
# print(typ_t)
# print(indices_inicio_t)
# print(indices_fin_t)
# print(indices_inicio_rel_t)
# print(indices_fin_rel_t)
# print(ids_t)
# print(palabras_t)

"""Funcion para crear frases mas cortas"""

# sel = train_df[train_df['DOCID']=='100042']
# # display(sel)

# texto = sel['TEXT'].unique()[0]
# print(texto)

# sel2 = tokenizados_df_train[tokenizados_df_train['DOCID']=='100042']
# # display(sel2)


# #inicios
# sel3 = sel2[sel2['id']==0]
# display(sel3)

# #finales
# maximo_frase = sel2['phrase'].max()

# aux = sel2[sel2['phrase']==maximo_frase]
# display(aux)

# maximo_id = aux['id'].max()
# sel4 = aux[aux['id']==maximo_id]
# display(sel4)




# frases = []
# inicios = []

# c = 0
# for index, row in sel3.iterrows():
#     inicio = int(row['start'])


#     if len(inicios) != 0:
#         print(inicios[c-1],inicio)
#         # print(texto[inicios[c-1]:inicio])

#         frase = texto[inicios[c-1]:inicio]
#         print(frase)
#         frases.append(frase)
    
#     inicios.append(inicio)
#     c += 1


# inicio = int(sel4['end'].tolist()[0])
# print(inicios[c-1],inicio)
# # print(texto[inicios[c-1]:inicio])

# frase = texto[inicios[c-1]:inicio]
# print(frase)
# frases.append(frase)

def separacion(df1, df2):
    docid_unicos = df1['DOCID'].unique()
    

    resultado = []
    
    for i in docid_unicos:
        # print(i)
    
        sel = df1[df1['DOCID']==i]
        # display(sel)

        texto = sel['TEXT'].unique()[0]
        # print(texto)

        sel2 = df2[df2['DOCID']==i]
        # display(sel2)


        #inicios
        sel3 = sel2[sel2['id']==0]
        # display(sel3)


        #finales
        maximo_frase = sel2['phrase'].max()
        aux = sel2[sel2['phrase']==maximo_frase]
        # display(aux)

        maximo_id = aux['id'].max()
        sel4 = aux[aux['id']==maximo_id]
        # display(sel4)





        frases = []
        inicios = []

        c = 0
        for index, row in sel3.iterrows():
            inicio = int(row['start'])

            if len(inicios) != 0:
                # print(inicios[c-1],inicio)
                # print(texto[inicios[c-1]:inicio])

                frase = texto[inicios[c-1]:inicio]
                # print(frase)
                frases.append(frase)

            inicios.append(inicio)
            c += 1


        inicio = int(sel4['end'].tolist()[0])
        # print(inicios[c-1],inicio)
        # print(texto[inicios[c-1]:inicio])

        frase = texto[inicios[c-1]:inicio]
        # print(frase)
        frases.append(frase)


        resultado.append(frases)
    # print(resultado)


    return resultado

frases = separacion(train_df, tokenizados_df_train)
frases1 = separacion(test_df, tokenizados_df_test)
frases2 = separacion(dev_df, tokenizados_df_dev)
# [a for a in frases]

# print(len(docd_t))
# print(len(prh_t))
# print(len(indices_inicio_t))
# print(len(indices_fin_t))
# print(len(indices_inicio_rel_t))
# print(len(indices_fin_rel_t))
# print(len(ids_t))
# print(len(palabras_t))
# print(len(frases))

# display(train_df)
# display(tokenizados_df_train)

# data = pd.DataFrame()

# for c, docid in enumerate(indices_inicio_t):
#     # print(docid)
#     # print(docd_t[c])
#     # print(prh_t[c])
#     # print(len(frases[c]))
    

#     for a, frase in enumerate(docid):
#         print(prh_t[c][a], frases[c][a], frase, indices_fin_t[c][a], ids_t[c][a], palabras_t[c][a], typ_t[c][a])
        
#         if len(frase)!= 0:
#             for v, token in enumerate(frase):
#                 print(token, indices_fin_t[c][a][v], ids_t[c][a][v], palabras_t[c][a][v], typ_t[c][a][v])
#                 fila = pd.DataFrame({'DOCID':docd_t[c], 'phrase':prh_t[c][a], 'TEXT_p':frases[c][a], 'start':token, 'end':indices_fin_t[c][a][v], 'id':ids_t[c][a][v], 'entity':palabras_t[c][a][v], 'type':typ_t[c][a][v]}, index=[0]) #'start_rel':indices_inicio_rel_t[c][a][v], 'end_rel':indices_fin_rel_t[c][a][v],
#                 data = pd.concat([data, fila])

def arreglo(docd_t, prh_t, typ_t, indices_inicio_t, indices_fin_t, indices_inicio_rel_t, indices_fin_rel_t, ids_t, palabras_t, frases):
    data = pd.DataFrame()

    for c, docid in enumerate(indices_inicio_t):
        # print(docid)
        # print(docd_t[c])
        # print(prh_t[c])
        # print(len(frases[c]))
        
        

        for a, frase in enumerate(docid):
            # print(prh_t[c][a], frases[c][a], frase, indices_fin_t[c][a], ids_t[c][a], palabras_t[c][a], typ_t[c][a])
            
            if len(frase)!= 0:
                for v, token in enumerate(frase):
                    # print(token, indices_fin_t[c][a][v], ids_t[c][a][v], palabras_t[c][a][v], typ_t[c][a][v])
                    fila = pd.DataFrame({'DOCID':docd_t[c], 'phrase':prh_t[c][a], 'TEXT_p':frases[c][a], 'start':token, 'end':indices_fin_t[c][a][v], 'id':ids_t[c][a][v], 'entity':palabras_t[c][a][v], 'type':typ_t[c][a][v]}, index=[0]) #'start_rel':indices_inicio_rel_t[c][a][v], 'end_rel':indices_fin_rel_t[c][a][v],
                    data = pd.concat([data, fila])
                    
                    
    return data   
        
    # display(data)

data = arreglo(docd_t, prh_t, typ_t, indices_inicio_t, indices_fin_t, indices_inicio_rel_t, indices_fin_rel_t, ids_t, palabras_t, frases)
data1 = arreglo(docd_t1, prh_t1, typ_t1, indices_inicio_t1, indices_fin_t1, indices_inicio_rel_t1, indices_fin_rel_t1, ids_t1, palabras_t1, frases1)
data2 = arreglo(docd_t2, prh_t2, typ_t2, indices_inicio_t2, indices_fin_t2, indices_inicio_rel_t2, indices_fin_rel_t2, ids_t2, palabras_t2, frases2)

# display(data)
# display(data1)
# display(data2)

# display(tokenizados_df_train)
# display(tokenizados_df_test)
# display(tokenizados_df_dev)

train_df[['DOCID','RML_start','RML_end','EVENT_start','EVENT_end']] = train_df[['DOCID','RML_start','RML_end','EVENT_start','EVENT_end']].apply(pd.to_numeric)
columns_to_convert = ['REL', 'RML_TEXT', 'EVENT_TEXT', 'TEXT', 'SCRIPTS', 'LABEL']
for col in columns_to_convert:
    train_df[col] = train_df[col].astype('category')

test_df[['DOCID','RML_start','RML_end','EVENT_start','EVENT_end']] = test_df[['DOCID','RML_start','RML_end','EVENT_start','EVENT_end']].apply(pd.to_numeric)
columns_to_convert = ['REL', 'RML_TEXT', 'EVENT_TEXT', 'TEXT', 'SCRIPTS', 'LABEL']
for col in columns_to_convert:
    test_df[col] = test_df[col].astype('category')

dev_df[['DOCID','RML_start','RML_end','EVENT_start','EVENT_end']] = dev_df[['DOCID','RML_start','RML_end','EVENT_start','EVENT_end']].apply(pd.to_numeric)
columns_to_convert = ['REL', 'RML_TEXT', 'EVENT_TEXT', 'TEXT', 'SCRIPTS', 'LABEL']
for col in columns_to_convert:
    dev_df[col] = dev_df[col].astype('category')





data[['DOCID','phrase','start','end','id']] = data[['DOCID','phrase','start','end','id']].apply(pd.to_numeric)
columns_to_convert = ['TEXT_p', 'entity', 'type']
for col in columns_to_convert:
    data[col] = data[col].astype('category')


data1[['DOCID','phrase','start','end','id']] = data1[['DOCID','phrase','start','end','id']].apply(pd.to_numeric)
columns_to_convert = ['TEXT_p', 'entity', 'type']
for col in columns_to_convert:
    data1[col] = data1[col].astype('category')


data2[['DOCID','phrase','start','end','id']] = data2[['DOCID','phrase','start','end','id']].apply(pd.to_numeric)
columns_to_convert = ['TEXT_p', 'entity', 'type']
for col in columns_to_convert:
    data2[col] = data2[col].astype('category')




# tokenizados_df_train[['DOCID','phrase','start','end','id', 'child','start_rel','end_rel']] = tokenizados_df_train[['DOCID','phrase','start','end','id', 'child','start_rel','end_rel']].apply(pd.to_numeric)
# columns_to_convert = ['text', 'type']
# for col in columns_to_convert:
#     tokenizados_df_train[col] = tokenizados_df_train[col].astype('category')


# tokenizados_df_test[['DOCID','phrase','start','end','id', 'child','start_rel','end_rel']] = tokenizados_df_test[['DOCID','phrase','start','end','id', 'child','start_rel','end_rel']].apply(pd.to_numeric)
# columns_to_convert = ['text', 'type']
# for col in columns_to_convert:
#     tokenizados_df_test[col] = tokenizados_df_test[col].astype('category')


# tokenizados_df_dev[['DOCID','phrase','start','end','id', 'child','start_rel','end_rel']] = tokenizados_df_dev[['DOCID','phrase','start','end','id', 'child','start_rel','end_rel']].apply(pd.to_numeric)
# columns_to_convert = ['text', 'type']
# for col in columns_to_convert:
#     tokenizados_df_dev[col] = tokenizados_df_dev[col].astype('category')

# display(train_df)
# display(data)

# print(docd_t)
# print(prh_t)
# print(typ_t)
# print(indices_inicio_t)
# print(indices_fin_t)
# print(indices_inicio_rel_t)
# print(indices_fin_rel_t)
# print(ids_t)
# print(palabras_t)

# display(test_df)
# display(data1)

# docid_unicos = train_df['DOCID'].unique()
# print(docid_unicos)
    

# resultado = []



# # for c, i in enumerate(docid_unicos):

# sel = test_df[test_df['DOCID'] == 100001]# 100978
# parrafo = sel['TEXT'].unique().tolist()[0] # todo entero

# sel2 = data1[data1['DOCID'] == 100001]

# # display(sel)
# # display(sel2)

# relations_g=[]
# tokens_g=[]

# relations=[]
# tokens=[]

# frasess = []

# c=0
# for index, row in sel.iterrows():
#     rml_start=row['RML_start']
#     rml_end=row['RML_end']
#     rml_text = row['RML_TEXT']

#     event_start=row['EVENT_start']
#     event_end=row['EVENT_end']
#     event_text = row['EVENT_TEXT']


#     rml_search = sel2[(sel2['start']==rml_start)|(sel2['end']==rml_end)]
#     dimens = rml_search.shape[0]
#     # print(dimens)
    
#     token_start_rml = rml_search['id'].tolist()[0]
#     token_end_rml = rml_search['id'].tolist()[dimens-1]
    
#     frase = rml_search['TEXT_p'].tolist()[0]
#     inicio_frase = parrafo.find(frase)

#     if (frase not in frasess):
#         if c == 0:
#             frasess.append(frase)
#             c+=1

#         else:
#             frasess.append(frase)
#             tokens_g.append(tokens)
#             relations_g.append(relations)


#             relations=[]
#             tokens=[]

        
    
        
        
    

#     # display(rml_search)


#     event_search = sel2[(sel2['start']==event_start)|(sel2['end']==event_end)]
#     token_start_event = event_search['id'].tolist()[0]
#     token_end_event = token_start_event

#     # display(event_search)
    



#     token1 = {
#         'text':unidecode(rml_text),
#         'start':rml_start-inicio_frase,
#         'end':rml_end-inicio_frase,
#         'token_start':token_start_rml,
#         'token_end':token_end_rml,
#         'entityLabel':'RML'
#     }
#     if token1 not in tokens:
#         tokens.append(token1)


#     token2 = {
#         'text':unidecode(event_text),
#         'start':event_start-inicio_frase,
#         'end':event_end-inicio_frase,
#         'token_start':token_start_event,
#         'token_end':token_end_event,
#         'entityLabel':'EVENT'
#     }
#     if token2 not in tokens:
#         tokens.append(token2)


#     rel = {
#         'child':token_start_rml,
#         'head':token_start_event,
#         'relationLabel':'REL'
#     }
#     if rel not in relations:
#         relations.append(rel)
    
        

# tokens_g.append(tokens)
# relations_g.append(relations)





# print(parrafo)
# for a, n in enumerate(frasess):
#     # print(n)
#     # print(tokens_g[a])
#     # print(relations_g[a])

#     final = {
#         'document':n,
#         'tokens':tokens_g[a],
#         'relations':relations_g[a]
#     }

#     resultado.append(final)

# print(resultado)

def transformacion(df1, df2):
    docid_unicos = df1['DOCID'].unique()
    # print(docid_unicos)
        

    resultado = []



    for c, i in enumerate(docid_unicos):

        sel = df1[df1['DOCID'] == i]
        parrafo = sel['TEXT'].unique().tolist()[0] # todo entero

        sel2 = df2[df2['DOCID'] == i]

        # display(sel)
        # display(sel2)

        relations_g=[]
        tokens_g=[]

        relations=[]
        tokens=[]

        frasess = []

        c=0
        for index, row in sel.iterrows():
            rml_start=row['RML_start']
            rml_end=row['RML_end']
            rml_text = row['RML_TEXT']

            event_start=row['EVENT_start']
            event_end=row['EVENT_end']
            event_text = row['EVENT_TEXT']

            try:
                rml_search = sel2[(sel2['start']==rml_start)|(sel2['end']==rml_end)]
                dimens = rml_search.shape[0]
                # print(dimens)
                
                token_start_rml = rml_search['id'].tolist()[0]
                token_end_rml = rml_search['id'].tolist()[dimens-1]
                
                frase = rml_search['TEXT_p'].tolist()[0]
                inicio_frase = parrafo.find(frase)
            except:
                pass

            if (frase not in frasess):
                if c == 0:
                    frasess.append(frase)
                    c+=1

                else:
                    frasess.append(frase)
                    tokens_g.append(tokens)
                    relations_g.append(relations)


                    relations=[]
                    tokens=[]

                
            
                
                
            

            # display(rml_search)

            try:
                event_search = sel2[(sel2['start']==event_start)|(sel2['end']==event_end)]
                token_start_event = event_search['id'].tolist()[0]
                token_end_event = token_start_event
            except:
                pass
            # display(event_search)
            



            token1 = {
                'text':unidecode(rml_text),
                'start':rml_start-inicio_frase,
                'end':rml_end-inicio_frase,
                'token_start':token_start_rml,
                'token_end':token_end_rml,
                'entityLabel':'RML'
            }
            if token1 not in tokens:
                tokens.append(token1)


            token2 = {
                'text':unidecode(event_text),
                'start':event_start-inicio_frase,
                'end':event_end-inicio_frase,
                'token_start':token_start_event,
                'token_end':token_end_event,
                'entityLabel':'EVENT'
            }
            if token2 not in tokens:
                tokens.append(token2)


            rel = {
                'child':token_start_rml,
                'head':token_start_event,
                'relationLabel':'REL'
            }
            if rel not in relations:
                relations.append(rel)
            
                

        tokens_g.append(tokens)
        relations_g.append(relations)





        # print(parrafo)
        for a, n in enumerate(frasess):
            # print(n)
            # print(tokens_g[a])
            # print(relations_g[a])

            final = {
                'document':unidecode(n),
                'tokens':tokens_g[a],
                'relations':relations_g[a]
            }

            resultado.append(final)

        # print(resultado)

    return resultado

from pprint import pprint
resul_train=transformacion(train_df, data)
resul_test=transformacion(test_df, data1)
resul_dev=transformacion(dev_df, data2)


# pprint(resul_train)
# print('-'*150)
# pprint(resul_test)
# print('-'*150)
# pprint(resul_dev)
# print('-'*150)

resul = resul_train + resul_test + resul_dev


path_procesado='./Basque/preprocessed'


import json


with open(path_procesado + '/relations_train.txt', 'w', encoding='utf-8') as archivo:
    contenido = json.dumps(resul_train, indent=4, ensure_ascii=False)
    archivo.write(contenido)

with open(path_procesado + '/relations_test.txt', 'w', encoding='utf-8') as archivo:
    contenido = json.dumps(resul_test, indent=4, ensure_ascii=False)
    archivo.write(contenido)

with open(path_procesado + '/relations_dev.txt', 'w', encoding='utf-8') as archivo:
    contenido = json.dumps(resul_dev, indent=4, ensure_ascii=False)
    archivo.write(contenido)





with open(path_procesado + '/relations_training.txt', 'w', encoding='utf-8') as archivo:
    contenido = json.dumps(resul, indent=4, ensure_ascii=False)
    archivo.write(contenido)

"""## NER"""

# display(tokenizados_df_train)
# display(tokenizados_df_test)
# display(tokenizados_df_dev)

# display(data)
# display(data1)
# display(data2)

tokenizados_df_train[['DOCID','phrase']] = tokenizados_df_train[['DOCID','phrase']].apply(pd.to_numeric)
tokenizados_df_test[['DOCID','phrase']] = tokenizados_df_test[['DOCID','phrase']].apply(pd.to_numeric)
tokenizados_df_dev[['DOCID','phrase']] = tokenizados_df_dev[['DOCID','phrase']].apply(pd.to_numeric)

# sel = tokenizados_df_train[tokenizados_df_train['DOCID'] == 100978]
# sel2 = data[data['DOCID'] == 100978]

# display(sel)
# display(sel2)

# frass = sel2['phrase'].unique().tolist()

# df_filtrado = sel[sel['phrase'].isin(frass)]
# display(df_filtrado)

def loco(df, df2, op):

    docid_uni = df['DOCID'].unique()
    
    if op == 1:
        nombre = '/NER_training.tsv'
    if op == 2:
        nombre = '/NER_TEST.tsv'
    if op == 3:
        nombre = '/NER_DEV.tsv'

    
    with open(path_procesado + nombre, 'w', encoding='utf-8') as archivo:

        for i in docid_uni:    
            sel = df[df['DOCID']==i]
            sel2 = df2[df2['DOCID']==i]
            
            frass = sel2['phrase'].unique().tolist()
            # print(frass)

            sela = sel[sel['phrase'].isin(frass)]
            # display(sel)

            # print(i)
            for j in frass:
                sel1 = sela[sela['phrase']==j]
                
                archivo.write('-DOCSTART- -X- O O\n')	
                
                palab = sel1['text'].tolist()
                typo = sel1['type'].tolist()

                a = 0
                c = 0

                while a != len(palab):

                    palabra = palab[a]
                    palabra = unidecode(str(palabra))

                    typr = typo[a]

                    if typr == 'PLAIN':
                        # print(str(palabra)+'	O\n')
                        archivo.write(str(palabra)+'	O\n')
                        c=0
                    
                    if typr == 'EVENT':
                        # print(str(palabra)+'	B-EVENT\n')
                        archivo.write(str(palabra)+'	B-EVENT\n')
                        c=0

                    if typr == 'RML':
                        if c == 0:
                            # print(str(palabra)+'	B-RML\n')
                            archivo.write(str(palabra)+'	B-RML\n')
                            c+=1
                        else:
                            # print(str(palabra)+'	I-RML\n')
                            archivo.write(str(palabra)+'	I-RML\n')

                    a+=1

loco(tokenizados_df_train, data, 1)
loco(tokenizados_df_test, data1, 2)
loco(tokenizados_df_dev, data2, 3)

"""##  GENERACION DE PRUEBAS"""

# als = test_df['TEXT'].unique()
# for i in range(0,3,1):
#     print(als[i])
print('\n'*15)
print('BASQUE - PREPROCESSING READY')