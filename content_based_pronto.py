#MC906 - Introducao a Inteligencia Artificial
#Construindo um sistema de recomendacao de filmes

import pandas
from pandas.plotting import scatter_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel

movies = pandas.read_csv('ml-latest-small/movies.csv')
movies.head()

#--------------------------------------------------#
#Recomendacao baseada em conteudo 

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies['genres'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim

titles = movies['title']
indices = pandas.Series(movies.index, index=movies['title'])

#Funcao que calcula recomendacoes de filme com base no score de similaridade de cossenos entre generos
#OBS: ha filmes identicos com mais de uma entrada no dataset (ex: 'Saturn 3 (1980)'). Nesse caso, o metodo nao funciona
def genre_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    #print(sim_scores)
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

#Testando a validade do recomendador com sequels
print(genre_recommendations('Free Willy (1993)').head(5)) #aventura, infantil, drama
print(genre_recommendations('Free Willy 2: The Adventure Home (1995)').head(5)) #aventura, infantil, drama 

print(genre_recommendations('Karate Kid, The (1984)').head(5)) #drama 
print(genre_recommendations('Karate Kid, Part II, The (1986)').head(5)) #acao, aventura, drama
print(genre_recommendations('Karate Kid, Part III, The (1989)').head(5)) #acao, aventura, infantil, drama
print(genre_recommendations('Next Karate Kid, The (1994)').head(5)) #acao, infantil, romance
print(genre_recommendations('Karate Kid, The (2010)').head(5)) #acao, infantil, drama 

#Testando para diferentes generos
print(genre_recommendations('Mulan (1998)').head(5)) #aventura, animacao, infantil, comedia, drama, musical, romance
print(genre_recommendations('Life Is Beautiful (La Vita è bella) (1997)').head(5)) #drama, romance, guerra, comedia
print(genre_recommendations('Sound of Music, The (1965)').head(5)) #musical, romance
print(genre_recommendations('Shining, The (1980)').head(5)) #horror
print(genre_recommendations('Paranormal Activity 3 (2011)').head(5)) #horror
print(genre_recommendations('10 Things I Hate About You (1999)').head(5)) #comedia, romance

#Personalizando para um usuario
ratings = pandas.read_csv('ml-latest-small/ratings.csv')
print("Informe ID do usuario a quem se deseja fazer recomendacao (numero de 1 a 610): ")
user = int(input())

#Seleciona classificacoes em que o usuario deu 5 estrelas
ratings_by_user = ratings['userId'] == user 
ratings_by_user = ratings[ratings_by_user]
fivestar_ratings_by_user = ratings_by_user['rating'] == 5.0
fivestar_ratings_by_user = ratings_by_user[fivestar_ratings_by_user]

#para cada filme classificado com 5 estrelas, o recomendador fornece uma lista de 5 filmes similares
for row in range(fivestar_ratings_by_user.shape[0]): 
	movie = fivestar_ratings_by_user.iat[row, 1] #id do filme cujo titulo queremos extrair
	pos = movies.index[movies['movieId'] == movie].tolist()[0] #posicao na linha de movies.csv onde esse id se encontra
	name = movies.iat[pos, 1] #titulo do filme
	print("Porque o usuario de ID " + str(user) + " gostou de " + name + ", recomendamos tambem:")
	print(genre_recommendations(name).head(5))
	print("\n")

#--------------------------------------------------#