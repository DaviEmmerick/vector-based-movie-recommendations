import ast 
import pandas as pd
df = pd.read_csv('database/raw/tmdb_5000_movies.csv')

def converter(obj):
    L = []

    for i in ast.literal_eval(obj):
        L.append(i['name']) 
    return " ".join(L) 

colunas_para_limpar = ['genres', 'keywords']

for col in colunas_para_limpar:
    df[col] = df[col].apply(converter)


df['text_for_ia'] = (
    df['original_title'] + " " + 
    df['overview'] + " " + 
    df['genres'] + " " + 
    df['keywords']
)

df.to_csv('database/processed/movies_cleaned.csv', index=False) 
print("Sucesso! Arquivo 'movies_cleaned.csv' gerado na pasta.")