# streamlit run /Users/joelpardo/Desktop/PAQUITA/interface.py
import streamlit as st
from PIL import Image
from utils_paquita import run_ner, run_re  # Asume que tienes funciones para NER y RE
from annotated_text import annotated_text  # Importamos el módulo
import pandas as pd

# Carga tu logotipo
logo = Image.open('log.png')

# Configura la página
st.set_page_config(page_title="Paquita - NER & RE", page_icon=logo, layout="centered", initial_sidebar_state="expanded")

def main():
    st.image(logo, use_column_width=True)
    st.title('Paquita – A Named Entity Recognition and Relation Extraction Tool')
    st.markdown('Welcome to my NER and RE tool. Here you can input text and extract named entities as well as their relations.')

    # Sidebar para opciones de entrada
    st.sidebar.title('Input Options')

    # Seleccionar el idioma
    language = st.sidebar.selectbox('Choose Language', ['Spanish', 'Italian', 'Basque', 'E1 – SPA & ITA', 'E2 – SPA & BAS', 'E3 – ITA & BAS', 'E4 - ALL'])

    # Seleccionar visualización
    visualize_option = st.sidebar.radio('Choose Visualization', ['NER', 'RE', 'Both'])

    # Entrada de texto
    user_input = st.text_area("Enter your text here", height=200)

    # if st.button('Extract'):
    #     # Obtén y muestra los resultados de NER y/o RE
    #     if visualize_option in ['NER', 'Both']:
    #         ner_results = run_ner(user_input, language)
    #         st.subheader('Named Entity Recognition Results')
    #         st.write(ner_results)

    #     if visualize_option in ['RE', 'Both']:
    #         re_results = run_re(user_input, language)
    #         st.subheader('Relation Extraction Results')
    #         st.write(re_results)

    COLORS = {
    "EVENT": "#faa",
    "RML": "#afa"
    # Añade más mapeos según sea necesario...
    }


    if st.button('Extract'):
        annotated_results = []
        if visualize_option in ['NER', 'Both']:
            ner_results = run_ner(user_input, language)
            st.subheader('Named Entity Recognition Results')
            last_end = 0
            for ent_text, ent_label, start, end in ner_results:
                if start > last_end:
                    annotated_results.append(user_input[last_end:start])
                annotated_results.append((ent_text, ent_label, COLORS.get(ent_label, "#fea")))
                last_end = end
            annotated_results.append(user_input[last_end:])
            annotated_text(*annotated_results)

        if visualize_option in ['RE', 'Both']:
            re_results = run_re(user_input, language)
            st.subheader('Relation Extraction Results')
            if re_results:
                relation_df = pd.DataFrame(re_results, columns=["Entity 1", "Entity 2", "Relationship"])
                st.table(relation_df)
            else:
                st.write("No relations found.")



if __name__ == "__main__":
    main()

# https://github.com/tvst/st-annotated-text