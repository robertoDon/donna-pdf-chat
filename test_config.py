#!/usr/bin/env python3
"""
Teste simples para verificar se config.py está funcionando
"""

try:
    from config import APP_TITLE, APP_DESCRIPTION
    print("✅ Importação bem-sucedida!")
    print(f"APP_TITLE: {APP_TITLE}")
    print(f"APP_DESCRIPTION: {APP_DESCRIPTION}")
except Exception as e:
    print(f"❌ Erro na importação: {e}")
    print(f"Tipo do erro: {type(e).__name__}")
    
    # Tenta importar o módulo diretamente
    try:
        import config
        print("✅ Módulo config importado diretamente")
        print(f"Variáveis disponíveis: {dir(config)}")
    except Exception as e2:
        print(f"❌ Erro ao importar módulo: {e2}") 