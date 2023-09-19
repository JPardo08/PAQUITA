import spacy

import random
import typer
from pathlib import Path
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from rel_pipe import make_relation_extractor, score_relations
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


def run_ner(user_input, language): 
    # MODIFICAR LOS PATHS
    if language == 'Spanish':
        path = "./Spanish/results_spanish_ner/model-best"
    elif language == 'Italian':
        path = "./Italian/results_italian_ner/model-best"
    elif language == 'Basque':
        path = "./Basque/results_basque_ner/model-best"
    elif language == 'E1 – SPA & ITA':
        path = "./E1/results_e1_ner/model-best"
    elif language == 'E2 – SPA & BAS':
        path = "./E2/results_e2_ner/model-best"
    elif language == 'E3 – ITA & BAS':
        path = "./E3/results_e3_ner/model-best"
    elif language == 'E4 - ALL':
        path = "./E4/results_e4_ner/model-best"
    else:
        raise ValueError(f"Unsupported language: {language}")

    nlp = spacy.load(path)

    doc = nlp(user_input)  # Asumiendo que la entrada es un solo texto y no una lista de textos
    ner_results = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]

    return ner_results

def run_re(user_input, language): 
    # MODIFICAR LOS PATHS
    if language == 'Spanish':
        ner_path = "./Spanish/results_spanish_ner/model-best"
        re_path = "./Spanish/results_spanish_re/model-best"
    elif language == 'Italian':
        ner_path = "./Italian/results_italian_ner/model-best"
        re_path = "./Italian/results_italian_re/model-best"
    elif language == 'Basque':
        ner_path = "./Basque/results_basque_ner/model-best"
        re_path = "./Basque/results_basque_re/model-best"
    elif language == 'E1 – SPA & ITA':
        ner_path = "./E1/results_e1_ner/model-best"
        re_path = "./E1/results_e1_re/model-best"
    elif language == 'E2 – SPA & BAS':
        ner_path = "./E2/results_e2_ner/model-best"
        re_path = "./E2/results_e2_re/model-best"
    elif language == 'E3 – ITA & BAS':
        ner_path = "./E3/results_e3_ner/model-best"
        re_path = "./E3/results_e3_re/model-best"
    elif language == 'E4 - ALL':
        ner_path = "./E4/results_e4_ner/model-best"
        re_path = "./E4/results_e4_re/model-best"
    else:
        raise ValueError(f"Unsupported language: {language}")

    # Primero cargamos y ejecutamos el modelo NER
    ner_nlp = spacy.load(ner_path)
    doc = ner_nlp(user_input)
    
    # Luego cargamos el modelo REL
    re_nlp = spacy.load(re_path)
    re_nlp.add_pipe('sentencizer')

    # Usamos las entidades del modelo NER en el modelo REL
    for name, proc in re_nlp.pipeline:
        doc = proc(doc) 

    # Extraemos las relaciones
    re_results = []
    for value, rel_dict in doc._.rel.items():
        for sent in doc.sents:
            for e in sent.ents:
                for b in sent.ents:
                    if e != b and e.start == value[0] and b.start == value[1]:
                        if rel_dict['REL'] >= 0.1: # Modificar el umbral para tomar cada relación en el texto
                            re_results.append((e.text, b.text, rel_dict['REL'])) # RESULTADO
    return re_results

