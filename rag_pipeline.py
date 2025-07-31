import os
import json
import time
from typing import List, Dict, Optional, Tuple
import streamlit as st
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import requests

# Diagnóstico do token do Hugging Face
def check_hf_token():
    """Verifica se o token do Hugging Face está configurado corretamente"""
    # Tenta ler do Streamlit secrets primeiro
    try:
        token_from_secrets = st.secrets.get("HUGGINGFACE_API_TOKEN", None)
        if token_from_secrets:
            os.environ["HUGGINGFACE_API_TOKEN"] = token_from_secrets
            if st.session_state.get('debug_mode', False):
                st.write("✅ Token carregado do Streamlit secrets")
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.write(f"⚠️ Erro ao ler secrets: {str(e)}")
    
    # Verifica se está no ambiente
    token_configured = "HUGGINGFACE_API_TOKEN" in os.environ
    token_value = os.environ.get("HUGGINGFACE_API_TOKEN", "Token não encontrado")
    
    if st.session_state.get('debug_mode', False):
        st.write(f"Token configurado: {token_configured}")
        st.write(f"Token valor: {token_value[:10]}..." if len(token_value) > 10 else token_value)
    
    return token_configured, token_value

def test_hf_connection():
    """Testa a conexão com o Hugging Face"""
    try:
        token = os.environ.get("HUGGINGFACE_API_TOKEN")
        if not token:
            return False, "Token não encontrado"
        
        # Teste simples da API
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get("https://huggingface.co/api/models", headers=headers)
        
        if response.status_code == 200:
            if st.session_state.get('debug_mode', False):
                st.write("✅ Conexão Hugging Face OK")
            return True, "Conexão OK"
        else:
            return False, f"Erro HTTP: {response.status_code}"
    except Exception as e:
        return False, f"Erro na conexão: {str(e)}"
# Configurações inline para evitar problemas de import
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS_CONTEXT = 1500
TOP_K_RESULTS = 5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "distilgpt2"  # Modelo mais simples e confiável
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1000
INDEX_FOLDER = "vector_index"

SYSTEM_PROMPT = """Você é um assistente especializado em responder perguntas baseado exclusivamente no conteúdo dos documentos PDF fornecidos.

IMPORTANTE:
- Responda APENAS com base no conteúdo dos documentos
- Se a informação não estiver nos documentos, diga claramente que não tem essa informação
- Cite sempre a fonte (nome do arquivo e página quando possível)
- Seja preciso e fiel ao conteúdo original
- Use linguagem clara e profissional

Contexto dos documentos:
{context}

Pergunta do usuário: {question}"""
from utils import (
    extract_text_from_pdf, get_file_hash, save_cache_data, 
    load_cache_data, count_tokens, truncate_text_by_tokens,
    format_source_citation
)

class RAGPipeline:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.documents = []
        self.initialize_embeddings()
        
    def initialize_embeddings(self):
        """Inicializa o modelo de embeddings"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            st.error(f"Erro ao carregar modelo de embeddings: {str(e)}")
    
    def process_pdf_file(self, file_path: str) -> bool:
        """Processa um arquivo PDF e adiciona ao índice vetorial"""
        try:
            # Verifica cache
            file_hash = get_file_hash(file_path)
            cache_data = load_cache_data(file_hash)
            
            if cache_data:
                st.info(f"Arquivo {os.path.basename(file_path)} já processado. Carregando do cache...")
                self.add_documents_from_cache(cache_data)
                return True
            
            # Extrai texto do PDF
            with st.spinner(f"Extraindo texto de {os.path.basename(file_path)}..."):
                pdf_chunks = extract_text_from_pdf(file_path)
            
            if not pdf_chunks:
                st.error(f"Não foi possível extrair texto do arquivo {os.path.basename(file_path)}")
                return False
            
            # Cria documentos LangChain
            documents = []
            for chunk in pdf_chunks:
                doc = Document(
                    page_content=chunk['text'],
                    metadata={
                        'source': chunk['source'],
                        'page': chunk['page'],
                        'method': chunk['method']
                    }
                )
                documents.append(doc)
            
            # Divide em chunks menores
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            # Salva no cache
            cache_data = {
                'file_hash': file_hash,
                'file_path': file_path,
                'documents': [
                    {
                        'page_content': doc.page_content,
                        'metadata': doc.metadata
                    } for doc in split_docs
                ]
            }
            save_cache_data(file_hash, cache_data)
            
            # Adiciona ao índice vetorial
            self.add_documents_to_index(split_docs)
            
            st.success(f"Arquivo {os.path.basename(file_path)} processado com sucesso!")
            return True
            
        except Exception as e:
            st.error(f"Erro ao processar arquivo {os.path.basename(file_path)}: {str(e)}")
            return False
    
    def add_documents_from_cache(self, cache_data: Dict):
        """Adiciona documentos do cache ao índice"""
        documents = []
        for doc_data in cache_data['documents']:
            doc = Document(
                page_content=doc_data['page_content'],
                metadata=doc_data['metadata']
            )
            documents.append(doc)
        
        self.add_documents_to_index(documents)
    
    def add_documents_to_index(self, documents: List[Document]):
        """Adiciona documentos ao índice vetorial"""
        if not documents:
            return
        
        # Adiciona à lista de documentos
        self.documents.extend(documents)
        
        # Cria ou atualiza o índice vetorial
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        
        # Salva o índice
        self.save_index()
    
    def save_index(self):
        """Salva o índice vetorial"""
        if self.vector_store:
            self.vector_store.save_local(INDEX_FOLDER)
    
    def load_index(self) -> bool:
        """Carrega o índice vetorial existente"""
        try:
            if os.path.exists(INDEX_FOLDER) and os.listdir(INDEX_FOLDER):
                self.vector_store = FAISS.load_local(
                    INDEX_FOLDER, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                st.success("Índice vetorial carregado com sucesso!")
                return True
        except Exception as e:
            st.warning(f"Não foi possível carregar índice existente: {str(e)}")
        
        return False
    
    def search_relevant_chunks(self, query: str) -> List[Document]:
        """Busca chunks relevantes para a pergunta"""
        if not self.vector_store:
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(
                query, k=TOP_K_RESULTS
            )
            
            # Filtra por score de similaridade
            filtered_results = []
            for doc, score in results:
                if score < 1.5:  # Threshold de similaridade
                    filtered_results.append(doc)
            
            return filtered_results
            
        except Exception as e:
            st.error(f"Erro na busca: {str(e)}")
            return []
    
    def generate_response(self, query: str, context_docs: List[Document]) -> Tuple[str, Dict]:
        """Gera resposta usando o LLM"""
        if not context_docs:
            return "Desculpe, não encontrei informações relevantes nos documentos para responder sua pergunta. Por favor, verifique se os documentos contêm informações sobre o tema.", {}
        
        # Verifica se o token do Hugging Face está configurado
        token_configured, token_value = check_hf_token()
        
        if not token_configured:
            return """❌ **Token do Hugging Face não configurado!**

Para resolver:
1. No Streamlit Cloud, vá em **Settings → Secrets**
2. Adicione: `HUGGINGFACE_API_TOKEN = "seu_token_aqui"`
3. Clique em **Save** e aguarde ~1 minuto

**Token atual**: """ + token_value, {}
        
        # Testa a conexão com o Hugging Face
        connection_ok, connection_msg = test_hf_connection()
        if not connection_ok:
            return f"""❌ **Erro na conexão com Hugging Face!**

**Detalhes**: {connection_msg}

**Possíveis causas**:
1. Token inválido ou expirado
2. Problema de conectividade
3. Erro na API do Hugging Face

**Token usado**: {token_value[:10]}...

**Solução**: Verifique o token no [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)""", {}
        
        # Prepara o contexto
        context_texts = []
        sources_used = []
        
        for doc in context_docs:
            context_texts.append(doc.page_content)
            source_info = {
                'source': doc.metadata.get('source', 'Desconhecido'),
                'page': doc.metadata.get('page', 'N/A')
            }
            sources_used.append(source_info)
        
        context = "\n\n".join(context_texts)
        
        # Trunca o contexto se necessário
        context_tokens = count_tokens(context)
        if context_tokens > MAX_TOKENS_CONTEXT:
            context = truncate_text_by_tokens(context, MAX_TOKENS_CONTEXT)
        
        # Prepara o prompt
        prompt = SYSTEM_PROMPT.format(context=context, question=query)
        
        try:
            # Gera resposta usando Hugging Face Inference API
            start_time = time.time()
            
            # Verifica token
            token_configured, token_value = check_hf_token()
            if not token_configured:
                return "❌ Token do Hugging Face não configurado. Configure HUGGINGFACE_API_TOKEN no Streamlit Cloud.", {}
            
            # Chama a API do Hugging Face
            headers = {
                "Authorization": f"Bearer {token_value}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": LLM_MAX_TOKENS,
                    "temperature": LLM_TEMPERATURE,
                    "do_sample": True,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{LLM_MODEL}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    response_text = result[0].get('generated_text', '')
                    # Remove o prompt original da resposta
                    if prompt in response_text:
                        response_text = response_text.replace(prompt, '').strip()
                else:
                    response_text = str(result)
            else:
                raise Exception(f"Erro HTTP {response.status_code}: {response.text}")
            
            # Estatísticas
            stats = {
                'response_time': round(response_time, 2),
                'context_tokens': context_tokens,
                'response_tokens': count_tokens(response_text),
                'sources_used': sources_used,
                'chunks_retrieved': len(context_docs)
            }
            
            return response_text, stats
            
        except Exception as e:
            error_msg = f"Erro ao gerar resposta: {str(e)}"
            
            # Log detalhado para debug
            if st.session_state.get('debug_mode', False):
                st.write(f"**Erro detalhado**: {str(e)}")
                st.write(f"**Tipo de erro**: {type(e).__name__}")
            
            if "API token" in str(e).lower() or "404" in str(e) or "authentication" in str(e).lower():
                error_msg = f"""❌ **Erro na API do Hugging Face!**

**Detalhes**: {str(e)}

**Token usado**: {token_value[:10]}...

**Possíveis causas**:
1. Token expirado ou inválido
2. Problema de conectividade
3. Erro na API do Hugging Face

**Solução**: 
1. Verifique o token em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Gere um novo token se necessário
3. Atualize no Streamlit Cloud (Settings → Secrets)
"""
            
            return error_msg, {}
    
    def get_answer(self, query: str) -> Tuple[str, List[Document], Dict]:
        """Pipeline completo: busca contexto e gera resposta"""
        if not self.vector_store:
            return "Nenhum documento foi carregado ainda. Por favor, faça upload de arquivos PDF primeiro.", [], {}
        
        # Busca chunks relevantes
        relevant_chunks = self.search_relevant_chunks(query)
        
        if not relevant_chunks:
            return "Não encontrei informações relevantes nos documentos para responder sua pergunta.", [], {}
        
        # Gera resposta
        response, stats = self.generate_response(query, relevant_chunks)
        
        return response, relevant_chunks, stats
    
    def clear_index(self):
        """Limpa o índice vetorial"""
        self.vector_store = None
        self.documents = []
        
        # Remove arquivos do índice
        if os.path.exists(INDEX_FOLDER):
            for file in os.listdir(INDEX_FOLDER):
                os.remove(os.path.join(INDEX_FOLDER, file))
        
        st.success("Índice vetorial limpo com sucesso!")
    
    def get_index_stats(self) -> Dict:
        """Retorna estatísticas do índice"""
        if not self.vector_store:
            return {'total_documents': 0, 'total_chunks': 0}
        
        return {
            'total_documents': len(set(doc.metadata.get('source') for doc in self.documents)),
            'total_chunks': len(self.documents)
        } 