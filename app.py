import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

st.set_page_config(page_title="Movie Recommendations by GenAI", layout="wide")
st.title("Movie Recommendations with GenAI")

@st.cache_data 
def load_data():
    df = pd.read_csv('database/processed/movies_cleaned.csv')
    df = df.dropna(subset=['text_for_ia']).reset_index(drop=True)
    return df

@st.cache_resource 
def load_models():
    print("Carregando modelos...")
    model_emb = SentenceTransformer('all-MiniLM-L6-v2')
    llm_chat = ChatOllama(model="llama3.2", temperature=0.7)
    return model_emb, llm_chat

@st.cache_data 
def get_embeddings(_model, texts):
    print("Calculando embeddings...")
    return _model.encode(texts, show_progress_bar=True)

df = load_data()
model, llm = load_models()
matriz_vetores = get_embeddings(model, df['text_for_ia'].tolist())
simularidade = cosine_similarity(matriz_vetores)


def recomendar_filme(nome_filme, top_k=5):
    
    filmes_encontrados = df[df['title'].str.lower() == nome_filme.lower()]
    
    if filmes_encontrados.empty:
        return None 
    
    idx = filmes_encontrados.index[0]
    scores = list(enumerate(simularidade[idx]))
    scores_ordenados = sorted(scores, key=lambda x: x[1], reverse=True)
    
    top_indices = [i[0] for i in scores_ordenados[1:top_k+1]]
    
    return df['title'].iloc[top_indices].tolist()

def genai_resposta(filme_user):
    recomendacoes = recomendar_filme(filme_user)

    if recomendacoes is None:
        return f"Desculpe, não encontrei o filme '{filme_user}' no banco de dados."
    
    template = """
    Você é um assistente especialista em cinema.
    O usuário disse que gosta do filme: {filme_input}.
    
    Baseado na nossa análise de dados, encontramos estes filmes similares para ele:
    {lista_filmes}
    
    Crie uma resposta curta e engajadora recomendando esses filmes. 
    Não invente filmes que não estão na lista.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    
    resposta = chain.invoke({
        "filme_input": filme_user,
        "lista_filmes": ", ".join(recomendacoes) 
    })
    
    return resposta.content

with st.form("form_recomendacao"):
    input_usuario = st.text_input("Gostaria de uma recomendação baseada em qual filme?")
    submitted = st.form_submit_button("Gerar Recomendação")

if submitted and input_usuario:
    with st.spinner('Consultando o Oráculo do Cinema...'):
        try:
            resultado = genai_resposta(input_usuario)
            st.success("Recomendação pronta!")
            st.markdown(resultado)
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
            st.info("Verifique se o Ollama está rodando no terminal.")