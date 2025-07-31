import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configurações da aplicação
APP_TITLE = "PDF Chat RAG"
APP_DESCRIPTION = "Chat com IA baseado em seus documentos PDF"

# Configurações do RAG
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS_CONTEXT = 1500
TOP_K_RESULTS = 5

# Configurações de embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Configurações do LLM
LLM_MODEL = "a16z-infra/llama-2-7b-chat"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1000

# Configurações de armazenamento
UPLOAD_FOLDER = "uploads"
CACHE_FOLDER = "cache"
INDEX_FOLDER = "vector_index"

# Configurações do Replicate
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")

# Configurações da interface
MAX_FILE_SIZE_MB = 50
SUPPORTED_FILE_TYPES = ["pdf"]

# Mensagens do sistema
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