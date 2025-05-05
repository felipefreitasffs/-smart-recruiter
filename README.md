# Smart Recruiter - Projeto Datathon POSTECH Fase 5

**Projeto desenvolvido para o Datathon da Fase 5 do curso de Data Analytics da POSTECH.**

**Data:** Maio de 2025

---

## 1. Introdução

O projeto **Smart Recruiter** surge como resposta ao desafio proposto no Datathon da POSTECH, focado em aplicar Inteligência Artificial para otimizar processos de recrutamento e seleção na empresa **Decision**.

A Decision, especializada em serviços de bodyshop e recrutamento no setor de TI, enfrenta desafios comuns na área: encontrar o talento certo rapidamente, garantir um bom "match" técnico e cultural, e avaliar o engajamento dos candidatos de forma eficaz. O processo atual, embora combine tecnologia e expertise humana, pode ser demorado e suscetível a inconsistências.

Este projeto visa desenvolver um MVP (Minimum Viable Product) de uma solução baseada em Machine Learning para auxiliar a Decision a identificar os candidatos mais promissores para suas vagas de forma mais eficiente e assertiva, utilizando os dados fornecidos.

## 2. Objetivos do Projeto

* **Analisar** os dados históricos fornecidos pela Decision, compreendendo as informações de vagas, candidatos e o relacionamento entre eles.
* **Identificar padrões** e características relevantes que correlacionam candidatos a vagas, utilizando o histórico de interações e status dos candidatos (ex: contratado, desistiu).
* **Desenvolver e treinar** um modelo de Machine Learning capaz de prever a compatibilidade ("match") entre novos candidatos e vagas em aberto, ou prever o status futuro de um candidato em um processo seletivo.
* **Construir** uma aplicação web simples (usando Streamlit) para demonstrar o funcionamento do modelo de forma interativa (MVP).
* **Disponibilizar** o código e a aplicação de forma organizada e acessível.

## 3. Dataset

Os dados utilizados neste projeto foram fornecidos no contexto do Datathon e consistem em três arquivos JSON distintos:

1. **`vagas.json`**: Contém informações detalhadas sobre as vagas de emprego, como ID, título, cliente, requisitos técnicos e comportamentais, localização, tipo de contratação, etc. Estes dados são carregados no DataFrame `df_jobs`.
2. **`applicants.json`**: Contém informações detalhadas sobre os candidatos (candidatos), incluindo ID, informações pessoais (contato, localização), informações profissionais (título, área, conhecimentos), formação acadêmica, experiências, etc. Estes dados são carregados no DataFrame `df_applicants`.
3. **`prospects.json`**: Contém os dados relacionais cruciais que conectam os candidatos (`applicants.json`) às vagas (`vagas.json`). Para cada vaga, lista os candidatos que participaram do processo seletivo, incluindo o status de cada um (`situacao_candidado` - ex: "Encaminhado ao Requisitante", "Desistiu", "Contratado"), datas relevantes e comentários do recrutador. Estes dados são processados e carregados no DataFrame `df_prospects_relationship`, onde cada linha representa uma interação candidato-vaga.

A análise exploratória inicial e o processo de limpeza/preparação destes dados podem ser encontrados nos notebooks Jupyter dentro deste repositório (ex: `notebooks/01_Data_Loading_and_Exploration.ipynb`).

## 4. Metodologia e Abordagem

A abordagem central deste projeto é a **construção de um modelo de Machine Learning para otimizar o processo de "match" entre candidatos e vagas**, utilizando os três conjuntos de dados fornecidos.

Os passos principais incluem:

1. **Carregamento e Limpeza dos Dados:** Leitura dos arquivos JSON (`vagas.json`, `applicants.json`, `prospects.json`), tratamento de estruturas aninhadas, tratamento de valores ausentes, correção de inconsistências e formatação dos dados nos DataFrames `df_jobs`, `df_applicants`, e `df_prospects_relationship`.
2. **Análise Exploratória de Dados (AED):** Investigação profunda dos dados para extrair insights, visualizar distribuições e entender relações entre as variáveis nos três DataFrames e após possíveis junções (merges).
3. **Engenharia de Atributos:** Criação de novas features relevantes a partir dos dados brutos (ex: extração de skills, cálculo de tempo de experiência, tempo de vaga aberta, codificação de variáveis categóricas). Combinação de informações dos três DataFrames para criar um dataset unificado para modelagem.
4. **Definição do Problema de ML:** Com base no `df_prospects_relationship` (especialmente a coluna `situacao_candidado`), definir o objetivo do modelo: prever a probabilidade de um candidato ser "Contratado"? Prever se um candidato será "Encaminhado ao Requisitante"? Ou criar um score de "match"?
5. **Seleção e Treinamento do Modelo:** Escolha de algoritmos de ML adequados (ex: Classificação como Regressão Logística, Random Forest, LightGBM; ou algoritmos de Ranking/Recomendação), treinamento com os dados preparados e ajuste de hiperparâmetros.
6. **Avaliação do Modelo:** Medição do desempenho do modelo utilizando métricas apropriadas para o problema definido (ex: Acurácia, Precisão, Recall, F1-Score, AUC-ROC, Métricas de Ranking como NDCG).
7. **Desenvolvimento da Aplicação (MVP):** Criação de uma interface com Streamlit para permitir a interação com o modelo (ex: inserir dados de uma vaga e ver candidatos recomendados com seus scores/status previstos).

## 5. Tecnologias Utilizadas

* **Linguagem:** Python 3.x
* **Bibliotecas Principais:**
  * `pandas`: Manipulação e análise de dados.
  * `numpy`: Computação numérica.
  * `scikit-learn`: Modelagem de Machine Learning e pré-processamento.
  * `matplotlib` & `seaborn`: Visualização de dados.
  * `streamlit`: Criação da aplicação web interativa.
  * `json`: Leitura e processamento de arquivos JSON complexos.
  * `nltk` / `spacy` (Opcional): Processamento de Linguagem Natural para análise de skills/descrições.
* **Ambiente:** Jupyter Notebooks (para desenvolvimento e análise), VS Code (ou outro IDE).

## 6. Estrutura do Projeto

O repositório está organizado da seguinte forma (sugestão):

```
├── data/                                                        # Pasta para armazenar os dados
│   └──  raw/                                                    # Dados originais
│        ├── vagas.json
│        ├── applicants.json
│        └── prospects.json
├── notebooks/                                                   # Jupyter Notebooks com análises e desenvolvimento
│   ├── 01_Data_Loading_and_Exploration.ipynb
│   └── ...
├── app/                                                         # Código da aplicação Streamlit
│   ├── smart_recruiter_app.py
│   └── ... (outros arquivos da app, como modelos salvos)
└── README.md                                                    # Este arquivo
```

## 7. Como Executar o Projeto

Existem duas maneiras principais de configurar o ambiente para executar este projeto:

**Opção 1: Usando o VS Code Dev Container (Recomendado)**

Esta é a forma mais fácil e recomendada para garantir um ambiente consistente, especialmente se você usa o VS Code.

1. **Pré-requisitos:**

   * [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e em execução.
   * [Visual Studio Code](https://code.visualstudio.com/) instalado.
   * Extensão [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) instalada no VS Code.
2. **Clonar o Repositório:**

   ```bash
   git clone https://github.com/felipefreitasffs/-smart-recruiter.git
   cd smart-recruiter
   ```
3. **Abrir no Container:**

   * Abra a pasta do projeto (`smart-recruiter`) no VS Code.
   * O VS Code deve detectar automaticamente o arquivo `.devcontainer/devcontainer.json` e perguntar se você deseja reabrir o projeto no container (`Reopen in Container`). Clique neste botão.
   * Alternativamente, você pode abrir a Paleta de Comandos (Ctrl+Shift+P ou Cmd+Shift+P) e procurar por `Remote-Containers: Reopen in Container`.
   * Aguarde o VS Code construir (na primeira vez) e iniciar o container. Isso pode levar alguns minutos.
4. **Verificar Instalação (Dentro do Container):**

   * Após o container iniciar, abra um terminal dentro do VS Code (Terminal > New Terminal). Ele já estará dentro do ambiente configurado.
   * As dependências listadas em `requirements.txt` (se especificadas no Dockerfile ou `postCreateCommand` do devcontainer) já devem estar instaladas. Caso contrário, execute: `pip install -r requirements.txt`.
5. **Executar Notebooks e Aplicação:**

   * Navegue até a pasta `notebooks/` e abra os arquivos `.ipynb`. O VS Code com a extensão Jupyter permitirá executá-los diretamente.
   * Para rodar a aplicação Streamlit, use o terminal do VS Code (já dentro do container):
     ```bash
     cd app/
     streamlit run smart_recruiter_app.py
     ```

     O VS Code geralmente encaminha a porta automaticamente para que você possa acessar a aplicação no seu navegador local.

**Opção 2: Configuração Manual com Ambiente Virtual**

Se você não estiver usando o VS Code Dev Containers, siga estes passos:

1. **Pré-requisitos:**

   * Python 3.x instalado.
   * Git instalado.
2. **Clonar o Repositório:**

   ```bash
   git clone https://github.com/felipefreitasffs/-smart-recruiter.git
   cd smart-recruiter
   ```
3. **Colocar os Dados:** Certifique-se de que os arquivos `vagas.json`, `applicants.json`, e `prospects.json` estejam na pasta `data/raw/`.
4. **Criar e Ativar Ambiente Virtual:**

   ```bash
   ython -m venv venv
   source venv/bin/activate  # Linux/macOS
   # ou
   .\venv\Scripts\activate  # Windows
   ```
5. **Instalar as Dependências:**

   ```bash
   pip install -r requirements.txt
   ```

   *(Certifique-se de que o arquivo `requirements.txt` está atualizado)*
6. **Executar os Notebooks:**
   Inicie o Jupyter Lab ou Notebook a partir do terminal com o ambiente virtual ativado e navegue até a pasta `notebooks/`.

   ```bash
   jupyter lab
   ```
7. **Executar a Aplicação Streamlit:**
   No terminal com o ambiente virtual ativado:

   ```bash
   cd app/
   streamlit run smart_recruiter_app.py
   ```

---
