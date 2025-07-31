import streamlit as st
import os
import tempfile
from datetime import datetime
from typing import List, Dict
import json

from config import APP_TITLE, APP_DESCRIPTION, UPLOAD_FOLDER
from rag_pipeline import RAGPipeline
from utils import create_directories, validate_pdf_file, get_file_info

# Configuração da página
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para estilo ChatGPT
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    
    .assistant-message {
        background-color: #f8f9fa;
        border-left-color: #28a745;
    }
    
    .source-citation {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9em;
        border-left: 3px solid #007bff;
    }
    
    .stats-box {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.8em;
        color: #6c757d;
    }
    
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .file-info {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Inicializa variáveis de sessão"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

def display_header():
    """Exibe o cabeçalho da aplicação"""
    st.markdown(f"""
    <div class="main-header">
        <h1>📚 {APP_TITLE}</h1>
        <p>{APP_DESCRIPTION}</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Exibe a barra lateral com controles"""
    with st.sidebar:
        st.markdown("### ⚙️ Configurações")
        
        # Upload de arquivos
        st.markdown("#### 📁 Upload de PDF")
        uploaded_files = st.file_uploader(
            "Selecione arquivos PDF",
            type=['pdf'],
            accept_multiple_files=True,
            help="Você pode selecionar múltiplos arquivos PDF"
        )
        
        if uploaded_files:
            process_uploaded_files(uploaded_files)
        
        # Estatísticas do índice
        st.markdown("#### 📊 Estatísticas")
        stats = st.session_state.rag_pipeline.get_index_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documentos", stats['total_documents'])
        with col2:
            st.metric("Chunks", stats['total_chunks'])
        
        # Controles
        st.markdown("#### 🛠️ Controles")
        
        if st.button("🔄 Recarregar Índice", help="Carrega índice existente"):
            st.session_state.rag_pipeline.load_index()
            st.rerun()
        
        if st.button("🗑️ Limpar Tudo", help="Limpa índice e histórico"):
            st.session_state.rag_pipeline.clear_index()
            st.session_state.messages = []
            st.session_state.uploaded_files = []
            st.rerun()
        
        # Debug mode
        st.markdown("#### 🔍 Debug")
        debug_mode = st.checkbox("Modo Debug", help="Mostra informações de diagnóstico")
        st.session_state.debug_mode = debug_mode
        
        if debug_mode:
            from rag_pipeline import check_replicate_token
            token_configured, token_value = check_replicate_token()
            st.write(f"**Token configurado**: {token_configured}")
            st.write(f"**Token**: {token_value[:10]}..." if len(token_value) > 10 else token_value)
        
        # Informações sobre arquivos processados
        if st.session_state.uploaded_files:
            st.markdown("#### 📋 Arquivos Processados")
            for file_info in st.session_state.uploaded_files:
                st.markdown(f"""
                <div class="file-info">
                    <strong>{file_info['name']}</strong><br>
                    Tamanho: {file_info['size_mb']} MB<br>
                    Status: ✅ Processado
                </div>
                """, unsafe_allow_html=True)

def process_uploaded_files(uploaded_files):
    """Processa arquivos PDF enviados"""
    for uploaded_file in uploaded_files:
        # Valida arquivo
        is_valid, message = validate_pdf_file(uploaded_file)
        
        if not is_valid:
            st.error(f"Erro no arquivo {uploaded_file.name}: {message}")
            continue
        
        # Verifica se já foi processado
        file_name = uploaded_file.name
        if any(f['name'] == file_name for f in st.session_state.uploaded_files):
            continue
        
        # Salva arquivo temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Processa o arquivo
            success = st.session_state.rag_pipeline.process_pdf_file(tmp_file_path)
            
            if success:
                # Adiciona à lista de arquivos processados
                file_info = get_file_info(tmp_file_path)
                st.session_state.uploaded_files.append(file_info)
                
        finally:
            # Remove arquivo temporário
            os.unlink(tmp_file_path)

def display_chat_interface():
    """Exibe a interface de chat"""
    # Área de chat
    chat_container = st.container()
    
    with chat_container:
        # Exibe mensagens do histórico
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Exibe fontes se disponível
                if "sources" in message and message["sources"]:
                    st.markdown("#### 📚 Fontes utilizadas:")
                    for source in message["sources"]:
                        citation = f"**{source['source']}** (página {source['page']})"
                        st.markdown(f"<div class='source-citation'>{citation}</div>", 
                                  unsafe_allow_html=True)
                
                # Exibe estatísticas se disponível
                if "stats" in message and message["stats"]:
                    stats = message["stats"]
                    stats_text = f"⏱️ Tempo: {stats['response_time']}s | 📊 Tokens: {stats['context_tokens']} (contexto) + {stats['response_tokens']} (resposta) | 🔍 Chunks: {stats['chunks_retrieved']}"
                    st.markdown(f"<div class='stats-box'>{stats_text}</div>", 
                              unsafe_allow_html=True)
    
    # Input do usuário
    if prompt := st.chat_input("Digite sua pergunta sobre os documentos..."):
        # Adiciona mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Exibe mensagem do usuário
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Gera resposta
        with st.chat_message("assistant"):
            with st.spinner("🔍 Buscando informações nos documentos..."):
                response, relevant_chunks, stats = st.session_state.rag_pipeline.get_answer(prompt)
            
            st.markdown(response)
            
            # Exibe fontes
            if relevant_chunks:
                sources = []
                for chunk in relevant_chunks:
                    source_info = {
                        'source': chunk.metadata.get('source', 'Desconhecido'),
                        'page': chunk.metadata.get('page', 'N/A')
                    }
                    sources.append(source_info)
                
                st.markdown("#### 📚 Fontes utilizadas:")
                for source in sources:
                    citation = f"**{source['source']}** (página {source['page']})"
                    st.markdown(f"<div class='source-citation'>{citation}</div>", 
                              unsafe_allow_html=True)
            
            # Exibe estatísticas
            if stats:
                stats_text = f"⏱️ Tempo: {stats['response_time']}s | 📊 Tokens: {stats['context_tokens']} (contexto) + {stats['response_tokens']} (resposta) | 🔍 Chunks: {stats['chunks_retrieved']}"
                st.markdown(f"<div class='stats-box'>{stats_text}</div>", 
                          unsafe_allow_html=True)
        
        # Adiciona resposta ao histórico
        message_data = {
            "role": "assistant", 
            "content": response,
            "timestamp": datetime.now().isoformat()
        }
        
        if relevant_chunks:
            sources = []
            for chunk in relevant_chunks:
                source_info = {
                    'source': chunk.metadata.get('source', 'Desconhecido'),
                    'page': chunk.metadata.get('page', 'N/A')
                }
                sources.append(source_info)
            message_data["sources"] = sources
        
        if stats:
            message_data["stats"] = stats
        
        st.session_state.messages.append(message_data)

def main():
    """Função principal da aplicação"""
    # Inicializa estado da sessão
    initialize_session_state()
    
    # Cria diretórios necessários
    create_directories()
    
    # Exibe cabeçalho
    display_header()
    
    # Layout principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Verifica se há documentos carregados
        if not st.session_state.uploaded_files:
            st.info("""
            👋 **Bem-vindo ao PDF Chat RAG!**
            
            Para começar:
            1. 📁 Faça upload de arquivos PDF na barra lateral
            2. ⏳ Aguarde o processamento dos documentos
            3. 💬 Comece a fazer perguntas sobre o conteúdo
            
            A IA responderá baseada exclusivamente nos documentos que você enviar.
            """)
        else:
            display_chat_interface()
    
    with col2:
        display_sidebar()

if __name__ == "__main__":
    main() 