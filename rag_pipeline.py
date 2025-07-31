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
import replicate

# Diagnóstico do token do Replicate
def check_replicate_token():
    """Verifica se o token do Replicate está configurado corretamente"""
    token_configured = "REPLICATE_API_TOKEN" in os.environ
    token_value = os.environ.get("REPLICATE_API_TOKEN", "Token não encontrado")
    
    if st.session_state.get('debug_mode', False):
        st.write(f"Token configurado: {token_configured}")
        st.write(f"Token valor: {token_value[:10]}..." if len(token_value) > 10 else token_value)
    
    return token_configured, token_value

def get_replicate_client():
    """Cria e testa o cliente Replicate"""
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise Exception("Token REPLICATE_API_TOKEN não encontrado no ambiente")
    
    try:
        client = replicate.Client(api_token=token)
        
        # Teste básico do cliente
        if st.session_state.get('debug_mode', False):
            st.write(f"Cliente criado com token: {token[:10]}...")
        
        return client
    except Exception as e:
        raise Exception(f"Erro ao criar cliente Replicate: {str(e)}")

def test_replicate_connection():
    """Testa a conexão com o Replicate"""
    try:
        client = get_replicate_client()
        
        # Teste simples com modelo hello-world
        versions = list(client.models.get("replicate/hello-world").versions.list())
        
        if st.session_state.get('debug_mode', False):
            st.write(f"Teste Replicate OK: {len(versions)} versões encontradas")
        
        # Teste específico com o modelo que vamos usar
        try:
            from config import LLM_MODEL
            model_versions = list(client.models.get(LLM_MODEL).versions.list())
            if st.session_state.get('debug_mode', False):
                st.write(f"Modelo {LLM_MODEL} OK: {len(model_versions)} versões encontradas")
        except Exception as model_error:
            if st.session_state.get('debug_mode', False):
                st.write(f"Erro no modelo {LLM_MODEL}: {str(model_error)}")
        
        return True, "Conexão OK"
    except Exception as e:
        return False, f"Erro na conexão: {str(e)}"
from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, MAX_TOKENS_CONTEXT, TOP_K_RESULTS,
    EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    SYSTEM_PROMPT, INDEX_FOLDER
)
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
        
        # Verifica se o token do Replicate está configurado
        token_configured, token_value = check_replicate_token()
        
        if not token_configured:
            return """❌ **Token do Replicate não configurado!**

Para resolver:
1. No Streamlit Cloud, vá em **Settings → Secrets**
2. Adicione: `REPLICATE_API_TOKEN = "seu_token_aqui"`
3. Clique em **Save** e aguarde ~1 minuto

**Token atual**: """ + token_value, {}
        
        # Testa a conexão com o Replicate
        connection_ok, connection_msg = test_replicate_connection()
        if not connection_ok:
            return f"""❌ **Erro na conexão com Replicate!**

**Detalhes**: {connection_msg}

**Possíveis causas**:
1. Token inválido ou expirado
2. Problema de conectividade
3. Versão da biblioteca replicate

**Token usado**: {token_value[:10]}...

**Solução**: Verifique o token no [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)""", {}
        
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
            # Gera resposta usando Replicate
            start_time = time.time()
            
            # Usa o cliente Replicate já testado
            replicate_client = get_replicate_client()
            
            output = replicate_client.run(
                LLM_MODEL,
                input={
                    "prompt": prompt,
                    "temperature": LLM_TEMPERATURE,
                    "max_tokens": LLM_MAX_TOKENS,
                    "top_p": 0.9,
                    "top_k": 50
                }
            )
            
            response_time = time.time() - start_time
            
            # Processa a resposta
            if isinstance(output, list):
                response = "".join(output)
            else:
                response = str(output)
            
            # Estatísticas
            stats = {
                'response_time': round(response_time, 2),
                'context_tokens': context_tokens,
                'response_tokens': count_tokens(response),
                'sources_used': sources_used,
                'chunks_retrieved': len(context_docs)
            }
            
            return response, stats
            
        except Exception as e:
            error_msg = f"Erro ao gerar resposta: {str(e)}"
            
            # Log detalhado para debug
            if st.session_state.get('debug_mode', False):
                st.write(f"**Erro detalhado**: {str(e)}")
                st.write(f"**Tipo de erro**: {type(e).__name__}")
            
            if "API token" in str(e).lower() or "404" in str(e) or "authentication" in str(e).lower():
                error_msg = f"""❌ **Erro na API do Replicate!**

**Detalhes**: {str(e)}

**Token usado**: {token_value[:10]}...

**Possíveis causas**:
1. Token expirado ou inválido
2. Problema de conectividade
3. Erro na API do Replicate

**Solução**: 
1. Verifique o token em [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)
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