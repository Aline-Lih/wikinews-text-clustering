# Agrupamento de Textos de Notícias em Categorias Distintas

Projeto de conclusão do curso de Big Data, desenvolvido em equipe (4 integrantes), sob orientação do tutor Alessandro Brassanini.

## Objetivo

Explorar e aplicar técnicas de **machine learning não supervisionado** para agrupar automaticamente artigos de notícias em português, sem categorias predefinidas, identificando padrões latentes no texto que permitam uma organização natural e intuitiva do conteúdo.

## Dataset

- **Fonte:** [WikiNoticias](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296929) (PlosOne), derivado do conteúdo colaborativo da Wikinews.
- **Formato:** JSON, com os campos `pageid`, `title`, `dates`, `body` e `category`.
- **Tipo de aprendizado:** não supervisionado (a coluna `category` original é descartada antes do treinamento).

## Metodologia

1. **Coleta e limpeza:** carregamento do JSON via URL, parsing de datas em formato português (`20 de fevereiro de 2005`) e análise temporal da distribuição de artigos por ano.
2. **Pré-processamento de texto (spaCy, `pt_core_news_sm`):**
   - conversão para minúsculas
   - remoção de stopwords (padrão + lista customizada)
   - remoção de pontuação
   - lematização
   - filtragem de classes gramaticais não informativas (preposições, artigos, conjunções)
3. **Vetorização:** TF-IDF (`max_features=1000`).
4. **Clustering:** comparação entre **KMeans**, **DBSCAN** e **Agglomerative Clustering**, avaliados por **Silhouette Score**.
5. **Validação dos rótulos:** os nomes de categoria atribuídos a cada cluster (Política, Economia, Tecnologia, Cultura, Saúde) foram checados por inspeção dos termos de maior peso TF-IDF em cada grupo, tornando a rotulagem auditável.
6. **Visualização:** distribuição de categorias, comparação de Silhouette Scores entre algoritmos, e projeção 2D dos clusters via **t-SNE**.

## Tecnologias

`Python` · `pandas` · `spaCy` · `scikit-learn` (TF-IDF, KMeans, DBSCAN, Agglomerative, t-SNE, Silhouette Score) · `Plotly` · `Matplotlib` · `Seaborn`

## Resultados

| Algoritmo | Silhouette Score |
|---|---|
| Agglomerative | **0.0760** (melhor) |
| KMeans | 0.0758 |
| DBSCAN | 0.0685 |

O Agglomerative Clustering teve o melhor desempenho, mas por margem mínima sobre o KMeans — os três algoritmos ficaram próximos. Vale notar que scores de Silhouette nessa faixa (perto de 0) indicam separação relativamente fraca entre os clusters, o que é comum em espaços TF-IDF de alta dimensionalidade e esparsidade — não significa que o agrupamento seja inútil, mas sim que os grupos não são fortemente compactos/isolados. Essa é uma limitação real do projeto, discutida abaixo.

## Limitações e direções futuras

- Os Silhouette Scores obtidos (0.07–0.08) são baixos em termos absolutos, indicando que os clusters não são fortemente separados — resultado esperado ao usar TF-IDF puro sobre textos de domínios variados, sem redução de dimensionalidade antes da clusterização.
- Os rótulos de categoria são atribuídos manualmente com base nos termos mais frequentes — não há garantia formal de correspondência semântica perfeita.
- DBSCAN é sensível à densidade variável do corpus, o que limitou seu desempenho neste dataset.
- Próximos passos possíveis: uso de embeddings contextuais (BERT/similares) no lugar de TF-IDF, busca sistemática de hiperparâmetros, análise de sentimento complementar, e clustering hierárquico multi-nível.

## Como executar

O notebook foi desenvolvido para rodar no Google Colab. Principais dependências:

```bash
pip install spacy
python -m spacy download pt_core_news_sm
```

Os dados (artigos e stopwords) são baixados automaticamente de links do Dropbox referenciados no próprio notebook.

## Autoria

Projeto em equipe de 4 pessoas — trabalho de conclusão de curso, 2024.

## Referências

Ver seção "Referências e Recursos Utilizados" no notebook, incluindo o artigo original do dataset: [*Unveiling a new dataset for Portuguese news classification*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296929).
