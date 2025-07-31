import os
import hashlib
import json
import fitz  # PyMuPDF
from typing import List, Dict, Optional, Tuple
import streamlit as st
from unstructured.partition.auto import partition
import tiktoken
from config import UPLOAD_FOLDER, CACHE_FOLDER, INDEX_FOLDER, MAX_FILE_SIZE_MB

def create_directories():
    """Cria os diretórios necessários para o funcionamento da aplicação"""
    directories = [UPLOAD_FOLDER, CACHE_FOLDER, INDEX_FOLDER]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_file_hash(file_path: str) -> str:
    """Gera hash MD5 do arquivo para cache"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def validate_pdf_file(uploaded_file) -> Tuple[bool, str]:
    """Valida se o arquivo PDF é válido"""
    if uploaded_file is None:
        return False, "Nenhum arquivo selecionado"
    
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"Arquivo muito grande. Máximo: {MAX_FILE_SIZE_MB}MB"
    
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False, "Apenas arquivos PDF são suportados"
    
    return True, "Arquivo válido"

def extract_text_from_pdf(file_path: str) -> List[Dict]:
    """Extrai texto de PDF usando PyMuPDF e fallback para unstructured"""
    chunks = []
    
    try:
        # Primeira tentativa: PyMuPDF
        doc = fitz.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            if text.strip():  # Se há texto extraído
                chunks.append({
                    'text': text.strip(),
                    'page': page_num + 1,
                    'source': os.path.basename(file_path),
                    'method': 'pymupdf'
                })
        
        doc.close()
        
        # Se não extraiu texto suficiente, tenta unstructured
        if not chunks or all(len(chunk['text']) < 50 for chunk in chunks):
            chunks = extract_with_unstructured(file_path)
            
    except Exception as e:
        # Fallback para unstructured
        chunks = extract_with_unstructured(file_path)
    
    return chunks

def extract_with_unstructured(file_path: str) -> List[Dict]:
    """Extrai texto usando unstructured (para PDFs escaneados)"""
    chunks = []
    
    try:
        elements = partition(filename=file_path)
        
        for i, element in enumerate(elements):
            if hasattr(element, 'text') and element.text.strip():
                chunks.append({
                    'text': element.text.strip(),
                    'page': getattr(element, 'metadata', {}).get('page_number', i + 1),
                    'source': os.path.basename(file_path),
                    'method': 'unstructured'
                })
    except Exception as e:
        st.error(f"Erro ao extrair texto do PDF: {str(e)}")
    
    return chunks

def count_tokens(text: str) -> int:
    """Conta tokens em um texto usando tiktoken"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        # Fallback simples: 1 token ≈ 4 caracteres
        return len(text) // 4

def truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    """Trunca texto para um número máximo de tokens"""
    tokens = count_tokens(text)
    if tokens <= max_tokens:
        return text
    
    # Trunca aproximadamente
    chars_to_keep = int(len(text) * (max_tokens / tokens))
    return text[:chars_to_keep] + "..."

def save_cache_data(filename: str, data: Dict):
    """Salva dados no cache"""
    cache_file = os.path.join(CACHE_FOLDER, f"{filename}.json")
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_cache_data(filename: str) -> Optional[Dict]:
    """Carrega dados do cache"""
    cache_file = os.path.join(CACHE_FOLDER, f"{filename}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    return None

def format_source_citation(source: str, page: int = None) -> str:
    """Formata citação da fonte"""
    if page:
        return f"**Fonte:** {source} (página {page})"
    return f"**Fonte:** {source}"

def get_file_info(file_path: str) -> Dict:
    """Obtém informações do arquivo"""
    stat = os.stat(file_path)
    return {
        'name': os.path.basename(file_path),
        'size_mb': round(stat.st_size / (1024 * 1024), 2),
        'modified': stat.st_mtime
    } 