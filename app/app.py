import streamlit as st
import pandas as pd
import re
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

# Configuração básica do app
st.set_page_config(
    page_title="Match Candidatos-Vagas",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🔍 Análise Avançada de Candidatos e Vagas")

# Carregar dados e remover duplicatas com cache persistente
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    try:
        df = pd.read_csv('data/processed/recomendacoes_por_vaga.csv')

        # Pré-processamento otimizado
        df = df.sort_values('similaridade', ascending=False)
        df = df.drop_duplicates(['nome', 'titulo_vaga'], keep='first')

        # Criar índices para busca mais rápida
        if not df.empty:
            df['perfil_lower'] = df['perfil_candidato'].str.lower()

        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()

# Carregar dados uma única vez
df = load_data()

# Sidebar com filtros principais
with st.sidebar:
    st.header("🔧 Filtros Principais")

    # Seleção do modo de análise
    analysis_mode = st.radio(
        "Modo de Análise:",
        ["📊 Por Vaga Específica", "🔍 Por Habilidades", "👤 Por Candidato"],
        index=0
    )

    # Filtros comuns
    min_similaridade = st.slider("Similaridade mínima:", 0.0, 1.0, 0.7, 0.01)
    top_n = st.slider("Número de resultados:", 1, 20, 10)

    # Filtros específicos por modo
    if analysis_mode == "📊 Por Vaga Específica":
        vaga_selecionada = st.selectbox("Selecione a Vaga:", options=df['titulo_vaga'].unique())
    elif analysis_mode == "🔍 Por Habilidades":
        st.header("🔎 Filtros de Habilidades")
        skills_input = st.text_area(
            "Digite as habilidades para buscar (uma por linha ou separadas por vírgula):",
            "Python, SQL, Machine Learning por exemplo"
        )
        search_method = st.radio("Método de busca:",
                               ["OR (qualquer skill)", "AND (todas as skills)"],
                               index=0)
        case_sensitive = st.checkbox("Diferenciar maiúsculas/minúsculas", False)
    else:  # Modo Por Candidato
        candidato_selecionado = st.selectbox("Selecione o Candidato:", options=df['nome'].unique())

# Funções auxiliares otimizadas com cache
@st.cache_data(ttl=600)
def parse_skills(input_text):
    skills = []
    for line in input_text.split('\n'):
        skills.extend([skill.strip() for skill in line.split(',') if skill.strip()])
    return skills

def search_skills_optimized(_df, skills, method="OR", case_sensitive=False):
    if not skills:
        return pd.Series(False, index=_df.index)

    flags = re.IGNORECASE if not case_sensitive else 0
    patterns = [re.compile(r'\b' + re.escape(skill) + r'\b', flags=flags) for skill in skills]

    def check_text(text):
        if not isinstance(text, str):
            return False
        matches = [bool(pattern.search(text)) for pattern in patterns]
        return all(matches) if method.startswith("AND") else any(matches)

    return _df['perfil_lower' if not case_sensitive else 'perfil_candidato'].apply(check_text)

# Processar dados conforme o modo selecionado
if analysis_mode == "📊 Por Vaga Específica":
    df_filtrado = df[(df['titulo_vaga'] == vaga_selecionada) &
                    (df['similaridade'] >= min_similaridade)]
    df_filtrado = df_filtrado.sort_values('similaridade', ascending=False).head(top_n)
elif analysis_mode == "🔍 Por Habilidades":
    skills = parse_skills(skills_input)
    if skills:
        mask = search_skills_optimized(df, skills, search_method, case_sensitive)
        df_filtrado = df[mask & (df['similaridade'] >= min_similaridade)]
        df_filtrado = df_filtrado.sort_values('similaridade', ascending=False).head(top_n)
    else:
        df_filtrado = pd.DataFrame()
else:  # Modo Por Candidato
    df_filtrado = df[(df['nome'] == candidato_selecionado) &
                    (df['similaridade'] >= min_similaridade)]
    df_filtrado = df_filtrado.sort_values('similaridade', ascending=False).head(top_n)

# Criar abas para diferentes visualizações
if analysis_mode == "👤 Por Candidato":
    tab1, tab2, tab3 = st.tabs(["🗂 Resultados", "💼 Melhores Vagas", "🧩 Análise de Skills"])
else:
    tab1, tab2 = st.tabs(["🗂 Resultados", "🧩 Análise de Skills"])

with tab1:
    # Resultados principais
    if analysis_mode == "📊 Por Vaga Específica":
        st.subheader(f"Top {top_n} candidatos para: {vaga_selecionada}")
    elif analysis_mode == "🔍 Por Habilidades":
        st.subheader(f"Top {top_n} candidatos com as habilidades buscadas")
    else:
        st.subheader(f"Informações do candidato: {candidato_selecionado}")

    if not df_filtrado.empty:
        # Estatísticas rápidas
        cols = st.columns(4)
        cols[0].metric("Resultados", len(df_filtrado))
        cols[1].metric("Similaridade Média", f"{df_filtrado['similaridade'].mean():.2f}")
        cols[2].metric("Similaridade Máxima", f"{df_filtrado['similaridade'].max():.2f}")

        if analysis_mode == "🔍 Por Habilidades":
            cols[3].metric("Vagas Distintas", df_filtrado['titulo_vaga'].nunique())
        elif analysis_mode == "👤 Por Candidato":
            cols[3].metric("Vagas Compatíveis", len(df_filtrado))

        # Destaque de skills nos perfis (se for busca por skills)
        if analysis_mode == "🔍 Por Habilidades" and skills:
            def highlight_skills(text):
                if not isinstance(text, str):
                    return text
                for skill in skills:
                    flags = re.IGNORECASE if not case_sensitive else 0
                    pattern = re.compile(r'(' + re.escape(skill) + r')', flags=flags)
                    text = pattern.sub(r'<span style="background-color: #FFFF00">\1</span>', text)
                return text

            df_filtrado['Perfil com destaque'] = df_filtrado['perfil_candidato'].apply(highlight_skills)
            st.write(
                df_filtrado[['nome', 'titulo_vaga', 'similaridade', 'Perfil com destaque']].sort_values(
                    'similaridade', ascending=False
                ).to_html(escape=False, index=False),
                unsafe_allow_html=True
            )
        else:
            st.dataframe(
                df_filtrado[['nome', 'titulo_vaga', 'perfil_candidato', 'similaridade']],
                column_config={
                    "similaridade": st.column_config.ProgressColumn(
                        "Similaridade",
                        format="%.2f",
                        min_value=0,
                        max_value=1
                    )
                },
                hide_index=True,
                use_container_width=True
            )

        # Gráfico de barras otimizado
        if analysis_mode != "👤 Por Candidato":
            st.bar_chart(df_filtrado, x='nome', y='similaridade')
        else:
            st.bar_chart(df_filtrado, x='titulo_vaga', y='similaridade')
    else:
        st.warning("Nenhum resultado encontrado com os critérios selecionados.")

# Adicionar aba específica para Melhores Vagas quando no modo Por Candidato
if analysis_mode == "👤 Por Candidato":
    with tab2:
        st.subheader(f"💼 Melhores Vagas para {candidato_selecionado}")

        if not df_filtrado.empty:
            # Criar visualização detalhada das melhores vagas
            cols = st.columns([1, 3])

            with cols[0]:
                st.metric("Vagas Compatíveis", len(df_filtrado))
                st.metric("Similaridade Média", f"{df_filtrado['similaridade'].mean():.2f}")

                # Mostrar perfil do candidato
                st.write("**Perfil do Candidato:**")
                st.write(df_filtrado['perfil_candidato'].iloc[0])

            with cols[1]:
                # Gráfico de radar otimizado
                fig = px.line_polar(
                    df_filtrado,
                    r='similaridade',
                    theta='titulo_vaga',
                    line_close=True,
                    title=f"Compatibilidade com Vagas"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Tabela detalhada
                st.dataframe(
                    df_filtrado[['titulo_vaga', 'similaridade']].sort_values('similaridade', ascending=False),
                    column_config={
                        "similaridade": st.column_config.ProgressColumn(
                            "Similaridade",
                            format="%.2f",
                            min_value=0,
                            max_value=1
                        )
                    },
                    hide_index=True
                )
        else:
            st.warning("Nenhuma vaga compatível encontrada para este candidato.")

with tab3 if analysis_mode == "👤 Por Candidato" else tab2:
    # Análise de Skills otimizada
    st.subheader("🧠 Distribuição de Habilidades")

    if analysis_mode == "🔍 Por Habilidades" and skills:
        # Contagem de skills otimizada
        with st.spinner("Analisando habilidades..."):
            try:
                vectorizer = CountVectorizer(vocabulary=[s.lower() for s in skills], binary=True)
                skill_matrix = vectorizer.fit_transform(df_filtrado['perfil_candidato'].fillna(''))
                skill_counts = pd.DataFrame(skill_matrix.toarray(), columns=vectorizer.get_feature_names_out())

                st.bar_chart(skill_counts.sum().sort_values(ascending=False))

                # Heatmap de skills por vaga
                skill_df = pd.concat([df_filtrado['titulo_vaga'], skill_counts], axis=1)
                heatmap_data = skill_df.groupby('titulo_vaga').mean()

                if not heatmap_data.empty:
                    st.subheader("🧩 Skills por Vaga")
                    st.dataframe(
                        heatmap_data.style.background_gradient(cmap='Blues', axis=None),
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Erro na análise de skills: {str(e)}")
    else:
        st.info("Selecione o modo 'Por Habilidades' e defina skills para ver esta análise")

# Rodapé
st.caption("Desenvolvido com Streamlit | Análise de Candidaturas e Vagas")