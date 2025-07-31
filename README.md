# 📚 PDF Chat RAG

Uma aplicação de chat com IA baseada em documentos PDF usando RAG (Retrieval-Augmented Generation) com interface estilo ChatGPT.

## 🚀 Características

- **Interface ChatGPT**: Interface limpa e profissional similar ao ChatGPT
- **RAG Completo**: Pipeline completo de recuperação e geração baseado em documentos
- **Suporte a Múltiplos PDFs**: Processa e indexa múltiplos arquivos simultaneamente
- **Extração Inteligente**: Usa PyMuPDF + fallback para unstructured (PDFs escaneados)
- **Cache Inteligente**: Evita reprocessamento de arquivos já processados
- **Citações de Fonte**: Mostra exatamente de onde veio cada informação
- **Estatísticas Detalhadas**: Tempo de resposta, tokens utilizados, etc.
- **Deploy Streamlit Cloud**: Pronto para deploy na nuvem

## 🛠️ Tecnologias Utilizadas

- **Frontend**: Streamlit
- **RAG Pipeline**: LangChain
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **PDF Processing**: PyMuPDF + Unstructured
- **LLM**: Mistral 7B via Replicate API
- **Cache**: Sistema de cache local com hash MD5

## 📋 Pré-requisitos

- Python 3.8+
- Token da API Replicate (opcional, para usar Mistral 7B)

## 🚀 Instalação

1. **Clone o repositório:**
```bash
git clone <url-do-repositorio>
cd pdf-chat-rag
```

2. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

3. **Configure as variáveis de ambiente (opcional):**
```bash
# Crie um arquivo .env
echo "REPLICATE_API_TOKEN=seu_token_aqui" > .env
```

4. **Execute a aplicação:**
```bash
streamlit run app.py
```

## 📖 Como Usar

### 1. Upload de Documentos
- Clique em "Browse files" na barra lateral
- Selecione um ou mais arquivos PDF
- Aguarde o processamento automático

### 2. Fazendo Perguntas
- Digite sua pergunta na caixa de chat
- A IA responderá baseada exclusivamente nos documentos
- Veja as fontes utilizadas abaixo de cada resposta

### 3. Controles
- **Recarregar Índice**: Carrega índice existente
- **Limpar Tudo**: Remove todos os documentos e histórico

## ⚙️ Configuração

### Modelo de Embeddings
O modelo padrão é `sentence-transformers/all-MiniLM-L6-v2`. Para alterar, edite `config.py`:

```python
EMBEDDING_MODEL = "seu_modelo_aqui"
```

### Modelo LLM
O modelo padrão é Mistral 7B via Replicate. Para usar outro modelo:

1. **Via Replicate**: Altere `LLM_MODEL` em `config.py`
2. **Localmente**: Modifique `rag_pipeline.py` para usar llama.cpp

### Parâmetros RAG
Ajuste os parâmetros em `config.py`:

```python
CHUNK_SIZE = 1000          # Tamanho dos chunks
CHUNK_OVERLAP = 200        # Sobreposição entre chunks
MAX_TOKENS_CONTEXT = 1500  # Máximo de tokens no contexto
TOP_K_RESULTS = 5          # Número de chunks recuperados
```

## 🏗️ Estrutura do Projeto

```
pdf-chat-rag/
├── app.py                 # Aplicação principal Streamlit
├── rag_pipeline.py        # Pipeline RAG completo
├── utils.py              # Funções utilitárias
├── config.py             # Configurações e constantes
├── requirements.txt      # Dependências Python
├── README.md            # Documentação
├── uploads/             # Arquivos temporários
├── cache/               # Cache de processamento
└── vector_index/        # Índice vetorial FAISS
```

## 🔧 Deploy no Streamlit Cloud

1. **Faça push para o GitHub**
2. **Conecte no Streamlit Cloud**
3. **Configure variáveis de ambiente** (se necessário):
   - `REPLICATE_API_TOKEN`: Seu token da Replicate

### Configuração do Streamlit Cloud

No painel do Streamlit Cloud, adicione:

```toml
[server]
maxUploadSize = 50
```

## 🐛 Solução de Problemas

### Erro de Token da API
```
Erro: Token da API Replicate não configurado
```
**Solução**: Configure a variável `REPLICATE_API_TOKEN` no Streamlit Cloud ou arquivo `.env`

### PDF não processado
```
Não foi possível extrair texto do PDF
```
**Solução**: O PDF pode estar corrompido ou ser uma imagem escaneada. Tente outro arquivo.

### Erro de memória
```
Out of memory
```
**Solução**: Reduza `CHUNK_SIZE` ou `TOP_K_RESULTS` em `config.py`

## 📊 Monitoramento

A aplicação exibe estatísticas detalhadas:
- **Tempo de resposta**: Tempo para gerar resposta
- **Tokens**: Tokens no contexto e resposta
- **Chunks**: Número de trechos recuperados
- **Fontes**: Documentos e páginas utilizados

## 🔒 Segurança

- Arquivos são processados localmente
- Cache é armazenado localmente
- Nenhum dado é enviado para serviços externos (exceto LLM)
- Arquivos temporários são removidos automaticamente

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 🙏 Agradecimentos

- [LangChain](https://langchain.com/) - Framework RAG
- [Streamlit](https://streamlit.io/) - Interface web
- [FAISS](https://github.com/facebookresearch/faiss) - Vector store
- [Mistral AI](https://mistral.ai/) - Modelo LLM
- [Replicate](https://replicate.com/) - API de inferência

---

**Desenvolvido com ❤️ para facilitar a consulta de documentos PDF** 