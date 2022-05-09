import numpy as np
import pandas as pd 
import seaborn as sns

# lendo o arquivo CSV com os dados do titanic
data = pd.read_csv("titanic-data.csv")

# Preview dos dados
# "PassengerId" -> é o id exclusivo da linha
# "Survived" -> é a variável de destino que estamos tentando prever (0 ou 1):
#   1 = Sobreviveu
#   0 = Não Sobreviveu
# "Pclass" -> qual a classe que o passageiro comprou (1, 2 ou 3):
#   1 = Primeira Classe
#   2 = Segunda Classe
#   3 = Terceira Classe
# "Name", "Sex" e "Age" - Nome, Sexo e Idade
# "SibSp" -> é o número total de irmãos e cônjuge dos passageiros
# "Parch" -> é o número total de pais e filhos dos passageiros
# "Ticket" -> é o número do bilhete do passageiro
# "Fare" -> a é a tarifa do passageiro
# "Cabin" -> é o número da cabine do passageiro
# "Embarked" -> é porto de embarque 
#   C = Cherburgo
#   Q = Queenstown
#   S = Southampton
print(data)

# Informações sobre os dados
# É possível observar que a coluna "Cabin" possui muitas informações em branco
# e que algumas colunas não foi possível determinar o tipo de dado (ex.: "Name")
data.info()

# Tamanho (em linhas) da minha base
len(data)

# Descartar as colunas que são consideradas desnecessárias para uma analise
filtro_data = data.drop(['Name','Ticket','Cabin'], axis=1)
filtro_data.isnull().sum()

# Como o feature "Embarked" tem somente 2 linhas que tem informação em branco, então vou avaliar qual o valor mais frequente
# para preencher com este valor. Assim estas linhas não serão descartadas das analises.
# Agrupando pela coluna desejada e somando os valores que são iguais (o valor S é o mais frequente)
filtro_data.groupby('Embarked')['PassengerId'].nunique()

 # Antes de remover todas as informações que tem valores em branco, que basicamente serão a que não tem a "idade", 
# será preenchido a coluna "Embarked" que são brancos com o valor "S"
# De 891 linhas de informação, ficaram 714 linhas
filtro_data["Embarked"] = filtro_data["Embarked"].fillna("S")
filtro_data = filtro_data.dropna()
len(filtro_data)

 # Iniciando uma analise dos dados
# avaliando um grafico (histograma) das idades dos passageiros sem considerar sobreviventes e não sobreviventes
filtro_data.Age.plot(kind='hist')

# Realizando uma correlação dos dados
# Pode-se observar que a maior correlação está entre valor do ticket com a classe da cabine
filtro_data.corr()

 # Avaliando a quantidade de pessoas que morreram e não morreram com relação a classe 
# Nos extremos, quem era da primeira classe teve uma taxa mais alta de sobreviver
# e quem era da terceira classe a taxa foi maior em não sobreviver
pd.crosstab(filtro_data.Pclass,filtro_data.Survived,margins=True).style.background_gradient(cmap='summer_r')

 # Gráfico de barras dos sobreviventes por sexo
sns.barplot(x="Sex", y="Survived", data=filtro_data)

 # Gráfico de barras dos sobreviventes por classe
sns.countplot(x='Survived',hue='Pclass',data=filtro_data,palette='rainbow')

# inserindo mais um coluna nos meus dados, AgeGroud que vai indicar a fase da idade do passageiro
# avaliando através do gráfic o percentual de sobrevivencia para cada destas faixa etárias
#Um novo grupo de categoria
ageGroups = ["Bebe","Criança","Adolescente","Jovem","Adulto","Senior"]
#Faixa de idade para os grupos
groupRanges = [0,5,12,18,35,60,81]
#Criando uma nova coluna AgeGroup de acordo com a faixa de idade
filtro_data["AgeGroup"] = pd.cut(filtro_data.Age, groupRanges, labels = ageGroups)
sns.barplot(x="AgeGroup", y="Survived", data=filtro_data)

# Avaliando os sobreviventes por local de embarque
sns.countplot(x = 'Embarked',hue='Survived',data=filtro_data, palette = "Set2" )


