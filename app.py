import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('database/processed/movies_cleaned.csv')
df = df.dropna(subset=['text_for_ia']).reset_index(drop=True)
model = SentenceTransformer('all-MiniLM-L6-v2')

matriz_vetores = model.encode(df['text_for_ia'].tolist(), show_progress_bar=True)

print("Calculando similaridade semântica")
simularidade = cosine_similarity(matriz_vetores)

def recomendar_filme(nome_filme, top_k=5):

  filme = df[df['title'].str.lower() == nome_filme.lower()] 

  if filme.empty: 
    return f"Filme {nome_filme} não existe no banco de dados!"
  
  idx = filme.index[0]
  scores = list(enumerate(simularidade[idx]))

  scores_ordenados = sorted(scores, key=lambda x: x[1], reverse=True)
    
  top_indices = [i[0] for i in scores_ordenados[1:top_k+1]]
    
  return df['title'].iloc[top_indices].tolist()

print(recomendar_filme("The Dark Knight"))