# 🚀 Guia de Configuração Rápida

## 1. Instalação das Dependências

```bash
pip install -r requirements.txt
```

## 2. Configuração do Replicate (Opcional)

Para usar o Mistral 7B via Replicate:

1. Crie uma conta em [replicate.com](https://replicate.com)
2. Obtenha seu token API em [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)
3. Crie um arquivo `.env` na raiz do projeto:

```bash
echo "REPLICATE_API_TOKEN=seu_token_aqui" > .env
```

## 3. Teste da Instalação

Execute o script de teste para verificar se tudo está funcionando:

```bash
python test_app.py
```

## 4. Executar a Aplicação

```bash
streamlit run app.py
```

## 5. Acessar a Aplicação

Abra seu navegador e acesse: `http://localhost:8501`

## 🔧 Configurações Avançadas

### Usar Modelo Local (Alternativa ao Replicate)

1. Instale o llama.cpp:
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
```

2. Baixe um modelo GGUF:
```bash
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

3. Modifique `rag_pipeline.py` para usar o modelo local (veja `local_llm_example.py`)

### Deploy no Streamlit Cloud

1. Faça push para o GitHub
2. Conecte no [Streamlit Cloud](https://streamlit.io/cloud)
3. Configure a variável de ambiente `REPLICATE_API_TOKEN` no painel

## 🐛 Solução de Problemas Comuns

### Erro: "No module named 'langchain'"
```bash
pip install -r requirements.txt
```

### Erro: "Token da API Replicate não configurado"
- Configure o token no arquivo `.env` ou
- Use um modelo local (veja configurações avançadas)

### Erro: "Out of memory"
- Reduza `CHUNK_SIZE` em `config.py`
- Use um modelo menor ou com quantização

### PDF não processado
- Verifique se o PDF não está corrompido
- Tente outro arquivo PDF
- PDFs escaneados podem demorar mais para processar

## 📞 Suporte

Se encontrar problemas:
1. Execute `python test_app.py` para diagnóstico
2. Verifique se todas as dependências estão instaladas
3. Consulte o `README.md` para documentação completa 