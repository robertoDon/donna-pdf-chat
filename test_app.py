"""
Script de teste para verificar se a aplicação PDF Chat RAG está funcionando corretamente.
Execute este script para validar a instalação e configuração.
"""

import os
import sys
import tempfile
from pathlib import Path

def test_imports():
    """Testa se todas as dependências estão instaladas"""
    print("🔍 Testando imports...")
    
    try:
        import streamlit
        print("✅ Streamlit importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar Streamlit: {e}")
        return False
    
    try:
        import langchain
        print("✅ LangChain importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar LangChain: {e}")
        return False
    
    try:
        import fitz  # PyMuPDF
        print("✅ PyMuPDF importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar PyMuPDF: {e}")
        return False
    
    try:
        import faiss
        print("✅ FAISS importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar FAISS: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar Sentence Transformers: {e}")
        return False
    
    try:
        import replicate
        print("✅ Replicate importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar Replicate: {e}")
        return False
    
    return True

def test_project_modules():
    """Testa se os módulos do projeto podem ser importados"""
    print("\n🔍 Testando módulos do projeto...")
    
    try:
        from config import APP_TITLE, EMBEDDING_MODEL
        print(f"✅ Config importado - App: {APP_TITLE}")
    except ImportError as e:
        print(f"❌ Erro ao importar config: {e}")
        return False
    
    try:
        from utils import create_directories, count_tokens
        print("✅ Utils importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar utils: {e}")
        return False
    
    try:
        from rag_pipeline import RAGPipeline
        print("✅ RAG Pipeline importado com sucesso")
    except ImportError as e:
        print(f"❌ Erro ao importar rag_pipeline: {e}")
        return False
    
    return True

def test_embeddings():
    """Testa se o modelo de embeddings pode ser carregado"""
    print("\n🔍 Testando modelo de embeddings...")
    
    try:
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL
        
        model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ Modelo de embeddings carregado: {EMBEDDING_MODEL}")
        
        # Testa uma inferência simples
        test_text = "Este é um teste de embeddings."
        embedding = model.encode(test_text)
        print(f"✅ Embedding gerado com sucesso (dimensão: {len(embedding)})")
        
        return True
    except Exception as e:
        print(f"❌ Erro ao testar embeddings: {e}")
        return False

def test_pdf_processing():
    """Testa se o processamento de PDF funciona"""
    print("\n🔍 Testando processamento de PDF...")
    
    try:
        import fitz
        from utils import extract_text_from_pdf
        
        # Cria um PDF de teste simples
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            # Cria um PDF básico com PyMuPDF
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Este é um teste de processamento de PDF.")
            doc.save(tmp_file.name)
            doc.close()
            
            # Testa a extração
            chunks = extract_text_from_pdf(tmp_file.name)
            
            if chunks:
                print(f"✅ PDF processado com sucesso - {len(chunks)} chunks extraídos")
                os.unlink(tmp_file.name)
                return True
            else:
                print("❌ Nenhum chunk foi extraído do PDF")
                os.unlink(tmp_file.name)
                return False
                
    except Exception as e:
        print(f"❌ Erro ao testar processamento de PDF: {e}")
        return False

def test_rag_pipeline():
    """Testa se o pipeline RAG pode ser inicializado"""
    print("\n🔍 Testando pipeline RAG...")
    
    try:
        from rag_pipeline import RAGPipeline
        
        pipeline = RAGPipeline()
        print("✅ Pipeline RAG inicializado com sucesso")
        
        # Testa estatísticas
        stats = pipeline.get_index_stats()
        print(f"✅ Estatísticas do índice: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Erro ao testar pipeline RAG: {e}")
        return False

def test_directories():
    """Testa se os diretórios necessários podem ser criados"""
    print("\n🔍 Testando criação de diretórios...")
    
    try:
        from utils import create_directories
        from config import UPLOAD_FOLDER, CACHE_FOLDER, INDEX_FOLDER
        
        create_directories()
        
        for directory in [UPLOAD_FOLDER, CACHE_FOLDER, INDEX_FOLDER]:
            if os.path.exists(directory):
                print(f"✅ Diretório criado: {directory}")
            else:
                print(f"❌ Diretório não foi criado: {directory}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Erro ao testar criação de diretórios: {e}")
        return False

def test_replicate_config():
    """Testa se a configuração do Replicate está correta"""
    print("\n🔍 Testando configuração do Replicate...")
    
    try:
        import replicate
        from config import REPLICATE_API_TOKEN
        
        if REPLICATE_API_TOKEN:
            print("✅ Token do Replicate configurado")
            return True
        else:
            print("⚠️ Token do Replicate não configurado (opcional)")
            print("   Para usar o Mistral 7B, configure REPLICATE_API_TOKEN no arquivo .env")
            return True
    except Exception as e:
        print(f"❌ Erro ao testar configuração do Replicate: {e}")
        return False

def main():
    """Executa todos os testes"""
    print("🚀 Iniciando testes da aplicação PDF Chat RAG\n")
    
    tests = [
        ("Imports", test_imports),
        ("Módulos do Projeto", test_project_modules),
        ("Embeddings", test_embeddings),
        ("Processamento de PDF", test_pdf_processing),
        ("Pipeline RAG", test_rag_pipeline),
        ("Diretórios", test_directories),
        ("Configuração Replicate", test_replicate_config),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 Teste: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSOU")
            else:
                print(f"❌ {test_name}: FALHOU")
        except Exception as e:
            print(f"❌ {test_name}: ERRO - {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 RESULTADO DOS TESTES")
    print('='*50)
    print(f"✅ Testes passaram: {passed}/{total}")
    print(f"❌ Testes falharam: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 Todos os testes passaram! A aplicação está pronta para uso.")
        print("\nPara executar a aplicação:")
        print("   streamlit run app.py")
    else:
        print(f"\n⚠️ {total - passed} teste(s) falharam. Verifique os erros acima.")
        print("\nPossíveis soluções:")
        print("1. Instale as dependências: pip install -r requirements.txt")
        print("2. Configure o token do Replicate (opcional)")
        print("3. Verifique se todos os arquivos estão presentes")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 