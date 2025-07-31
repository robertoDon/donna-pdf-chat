"""
Exemplo de como integrar modelos locais (llama.cpp) como alternativa ao Replicate.
Para usar, substitua a função generate_response em rag_pipeline.py por esta implementação.
"""

import subprocess
import json
import tempfile
from typing import List, Dict, Tuple
from config import LLM_TEMPERATURE, LLM_MAX_TOKENS, SYSTEM_PROMPT

class LocalLLM:
    def __init__(self, model_path: str = "models/mistral-7b-instruct.gguf"):
        """
        Inicializa o modelo local
        
        Args:
            model_path: Caminho para o arquivo .gguf do modelo
        """
        self.model_path = model_path
        self.llama_cpp_path = "llama.cpp"  # Assumindo que llama.cpp está instalado
        
    def generate_response(self, query: str, context_docs: List) -> Tuple[str, Dict]:
        """
        Gera resposta usando modelo local via llama.cpp
        
        Args:
            query: Pergunta do usuário
            context_docs: Documentos de contexto
            
        Returns:
            Tuple com resposta e estatísticas
        """
        if not context_docs:
            return "Desculpe, não encontrei informações relevantes nos documentos.", {}
        
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
        
        # Prepara o prompt
        prompt = SYSTEM_PROMPT.format(context=context, question=query)
        
        try:
            # Cria arquivo temporário com o prompt
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(prompt)
                prompt_file = f.name
            
            # Comando para llama.cpp
            cmd = [
                self.llama_cpp_path,
                "-m", self.model_path,
                "-f", prompt_file,
                "--temp", str(LLM_TEMPERATURE),
                "--ctx-size", str(LLM_MAX_TOKENS * 2),  # Contexto maior para acomodar prompt
                "--repeat-penalty", "1.1",
                "--top-p", "0.9",
                "--top-k", "50",
                "--n-predict", str(LLM_MAX_TOKENS)
            ]
            
            # Executa o comando
            import time
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutos de timeout
            )
            
            response_time = time.time() - start_time
            
            # Remove arquivo temporário
            import os
            os.unlink(prompt_file)
            
            if result.returncode != 0:
                raise Exception(f"Erro ao executar llama.cpp: {result.stderr}")
            
            # Processa a resposta
            response = result.stdout.strip()
            
            # Remove o prompt da resposta (se presente)
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            # Estatísticas
            stats = {
                'response_time': round(response_time, 2),
                'context_tokens': len(context.split()),  # Aproximação
                'response_tokens': len(response.split()),  # Aproximação
                'sources_used': sources_used,
                'chunks_retrieved': len(context_docs),
                'model': 'local_llama_cpp'
            }
            
            return response, stats
            
        except Exception as e:
            error_msg = f"Erro ao gerar resposta com modelo local: {str(e)}"
            return error_msg, {}

# Exemplo de uso:
"""
# Para usar o modelo local, modifique rag_pipeline.py:

from local_llm_example import LocalLLM

class RAGPipeline:
    def __init__(self):
        # ... código existente ...
        self.local_llm = LocalLLM("caminho/para/seu/modelo.gguf")
    
    def generate_response(self, query: str, context_docs: List[Document]) -> Tuple[str, Dict]:
        # Use o modelo local em vez do Replicate
        return self.local_llm.generate_response(query, context_docs)
"""

# Instruções para instalar llama.cpp:
"""
1. Clone o repositório llama.cpp:
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   make

2. Baixe um modelo GGUF:
   # Exemplo: Mistral 7B
   wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf

3. Configure o caminho no LocalLLM:
   local_llm = LocalLLM("caminho/para/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
""" 