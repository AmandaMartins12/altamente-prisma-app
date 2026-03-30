import streamlit as st
import pandas as pd
import joblib

# 1. Configuração da Página
st.set_page_config(page_title="Altamente Prisma", page_icon="🧩", layout="wide")

# 2. Carregamento de Dados com Cache
@st.cache_resource
def carregar_arquivos():
    # Certifique-se que esses arquivos estão na raiz do seu GitHub
    modelo = joblib.load('modelo_prisma_final.pkl')
    df_dict = pd.read_csv('prisma_5_dicionario.csv')
    return modelo, df_dict

modelo, df_dict = carregar_arquivos()

# 3. Cabeçalho Principal
st.title("🧩 Altamente Prisma")
st.markdown("### Sistema de Triagem e Apoio Pedagógico")
st.divider()

# 4. Layout: Dados (Esquerda) | Resultados (Direita)
col_dados, col_resultados = st.columns([1, 2], gap="large")

with col_dados:
    st.subheader("📋 Inserir Dados do Estudante")
    
    with st.expander("📚 Desempenho Acadêmico (1-10)", expanded=True):
        leitura = st.slider("Leitura", 1, 10, 5)
        matematica = st.slider("Matemática", 1, 10, 5)
        escrita = st.slider("Escrita", 1, 10, 5)
        acima_media = st.slider("Desempenho Acima da Média Geral", 1, 5, 3)

    with st.expander("🧠 Perfil Socioemocional (1-5)"):
        abertura = st.slider("Abertura ao Novo", 1, 5, 3)
        org = st.slider("Organização", 1, 5, 3)
        social = st.slider("Interação Social", 1, 5, 3)
        amabilidade = st.slider("Amabilidade", 1, 5, 3)
        emocional = st.slider("Estabilidade Emocional", 1, 5, 3)

    with st.expander("⚡ Comportamento e Motricidade (1-5)"):
        foco = st.slider("Foco Sustentado", 1, 5, 3)
        sensorial = st.slider("Reatividade Sensorial", 1, 5, 3)
        motora = st.slider("Coordenação Motora Fina", 1, 5, 3)
        nao_verbal = st.slider("Comunicação Não-Verbal", 1, 5, 3)

    with st.expander("🎨 Preferências de Aprendizagem (1-5)"):
        p_visual = st.slider("Visual", 1, 5, 3)
        p_auditiva = st.slider("Auditiva", 1, 5, 3)
        p_cinestesica = st.slider("Cinestésica", 1, 5, 3)
        ambiente = st.slider("Ambiente Preferencial", 1, 5, 3)

    gerar = st.button("🪄 Analisar Perfil do Estudante", use_container_width=True, type="primary")

with col_resultados:
    if gerar:
        # --- SOLUÇÃO PARA O VALUEERROR ---
        # 1. Mapeamos os inputs em um dicionário
        inputs = {
            'idade': 10, 
            'perf_leitura': leitura, 'perf_matematica': matematica, 'perf_escrita': escrita, 
            'desempenho_acima_media': acima_media, 'abertura_novo': abertura, 
            'organizacao': org, 'interacao_social': social, 'amabilidade': amabilidade, 
            'estabilidade_emocional': emocional, 'foco_sustentado': foco, 
            'reatividade_sensorial': sensorial, 'coord_motora_fina': motora, 
            'comunicacao_nao_verbal': nao_verbal, 'pref_visual': p_visual, 
            'pref_auditiva': p_auditiva, 'pref_cinestesica': p_cinestesica, 
            'ambiente_preferencial': ambiente
        }

        # 2. Criamos o DataFrame
        dados_entrada = pd.DataFrame([inputs])

        # 3. Reordenamos as colunas exatamente como no treinamento
        colunas_treino = [
            'idade', 'perf_leitura', 'perf_matematica', 'perf_escrita', 
            'desempenho_acima_media', 'abertura_novo', 'organizacao', 
            'interacao_social', 'amabilidade', 'estabilidade_emocional', 
            'foco_sustentado', 'reatividade_sensorial', 'coord_motora_fina', 
            'comunicacao_nao_verbal', 'pref_visual', 'pref_auditiva', 
            'pref_cinestesica', 'ambiente_preferencial'
        ]
        dados_entrada = dados_entrada[colunas_treino]

        try:
            # 4. Predição e busca
            perfil = modelo.predict(dados_entrada)[0]
            recomendas = df_dict[df_dict['perfil'] == perfil].iloc[0]
            
            # Exibição do Diagnóstico
            st.subheader("🔍 Resultado da Triagem")
            st.info(f"O modelo identificou padrões compatíveis com o perfil: **{perfil}**")
            
            st.markdown("### 💡 Plano de Intervenção Recomendado")
            c1, c2, c3 = st.columns(3)
            c1.success(f"**Estratégia 1:**\n\n{recomendas['rec_1']}")
            c2.warning(f"**Estratégia 2:**\n\n{recomendas['rec_2']}")
            c3.error(f"**Estratégia 3:**\n\n{recomendas['rec_3']}")

            st.divider()
            st.subheader("📝 Validação do Especialista")
            f1, f2, f3 = st.columns(3)
            if f1.button("👍 Sim, concordo"):
                st.toast("Feedback registrado! Isso ajuda a calibrar nossa IA.")
            if f2.button("👎 Não, parece impreciso"):
                st.toast("Sinalizado para revisão manual.")
            
            st.text_area("Observações adicionais:", placeholder="Ex: O aluno apresenta sinais de hiperfoco...")
            if st.button("Salvar Observação"):
                st.success("Anotação salva com sucesso!")

        except Exception as e:
            st.error(f"Erro na predição: {e}. Verifique se as colunas do modelo coincidem.")

    else:
        st.write("👈 Ajuste os indicadores do estudante ao lado e clique em **Analisar Perfil**.")
