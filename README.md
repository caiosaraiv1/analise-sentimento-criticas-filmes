# 🎬 Análise de Sentimento de Críticas de Filmes IMDb

## 📖 Sobre o Projeto

Este projeto implementa um modelo de **Análise de Sentimento** capaz de classificar automaticamente críticas de filmes do IMDb como **positivas** ou **negativas**. Utilizando técnicas de Processamento de Linguagem Natural (NLP) e Machine Learning, o sistema analisa o texto das avaliações e prevê o sentimento por trás de cada crítica.

> **Inspiração**: Este projeto foi desenvolvido com base nos conceitos aprendidos no curso **"NLP: aplicando processamento de linguagem natural para análise de sentimentos"** da [Alura](https://www.alura.com.br/).

## 🗂️ Dataset

O projeto utiliza o **IMDB Dataset of 50K Movie Reviews**, que contém:
- 50.000 críticas de filmes em inglês
- Classificação binária: positivo/negativo
- Distribuição balanceada (25k positivas, 25k negativas)

## 🎯 Objetivos

1. **Explorar e Pré-processar** um grande conjunto de dados de avaliações
2. **Vetorizar** o texto usando TF-IDF 
3. **Treinar e Otimizar** um modelo de Regressão Logística
4. **Analisar** as palavras mais influentes na classificação
5. **Construir** um pipeline completo para predição de novos reviews

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **Pandas** - Manipulação de dados
- **Scikit-learn** - Machine Learning
- **NLTK** - Processamento de linguagem natural
- **Joblib** - Persistência de modelos

## 📊 Metodologia

### 1. Exploração dos Dados
- Análise da distribuição das classes
- Investigação da frequência de palavras
- Identificação de stopwords

### 2. Pré-processamento
- Conversão para minúsculas
- Tokenização com RegexpTokenizer
- Remoção de stopwords
- Teste com e sem stemming

### 3. Vetorização
- **TF-IDF Vectorizer** com:
  - Max features: 10.000
  - N-grams: (1,2)
  - Captura de contexto com bigramas

### 4. Modelagem
- **Regressão Logística** com solver='saga'
- Divisão treino/teste (75%/25%)
- Avaliação de performance

### 5. Análise de Resultados
- Investigação dos pesos do modelo
- Identificação das palavras mais influentes
- Comparação de diferentes abordagens de pré-processamento

## 📈 Resultados

- **Acurácia final**: ~89%
- **Insight importante**: Stemming não melhorou a performance
- **Melhor abordagem**: Lowercase + remoção de stopwords
- **Palavras positivas influentes**: excellent, amazing, perfect, wonderful
- **Palavras negativas influentes**: terrible, awful, worst, boring

## 🚀 Como Usar

### Instalação
```bash
pip install pandas scikit-learn nltk joblib
```

### Download das dependências NLTK
```python
import nltk
nltk.download('stopwords')
```

### Executar o projeto
1. Clone o repositório
2. Baixe o dataset IMDB
3. Execute o notebook/script principal
4. Os modelos treinados serão salvos automaticamente

### Predição em novos reviews
```python
import joblib
from nltk import tokenize

# Carregar modelos
tfidf = joblib.load('tfidf_vectorizer.pkl')
modelo = joblib.load('modelo_regressao_logistica.pkl')

# Preprocessar e predizer
novo_review = "This movie was absolutely fantastic!"
review_processado = preprocessa_ajustado(novo_review)
review_tfidf = tfidf.transform([review_processado])
predicao = modelo.predict(review_tfidf)
```

## 📁 Estrutura do Projeto

```
├── .gitignore
├── IMDB Dataset.csv                                 # Dataset original
├── modelo_regressao_logistica.pkl                   # Modelo treinado
├── tfidf_vectorizer.pkl                             # Vetorizador TF-IDF
├── Classificação de Resenhas de Filmes.ipynb        # Notebook principal
└── README.md                                        # Este arquivo
```

## 🎓 Créditos

- **Dataset**: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Curso inspirador**: [NLP: aplicando processamento de linguagem natural para análise de sentimentos - Alura](https://www.alura.com.br/)
