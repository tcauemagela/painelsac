# Planilha Padrão

Este diretório contém a planilha padrão que será carregada automaticamente pelo sistema.

## Como Usar

1. Coloque sua planilha padrão neste diretório com o nome `planilha_padrao.xlsx`
2. O sistema irá carregar automaticamente este arquivo sempre que iniciar
3. Você ainda pode fazer upload de arquivos adicionais pela interface
4. Os dados da planilha padrão serão mesclados com os uploads adicionais

## Formato Esperado

A planilha deve conter as seguintes colunas (com nomes flexíveis, o sistema mapeia automaticamente):

- NU_REGISTRO ou equivalentes (numero registro, etc.)
- DS_ASSUNTO ou equivalentes (assunto, categoria, etc.)
- SUB_ASSUNTO ou equivalentes (subassunto, subcategoria, etc.)
- DS_OBSERVACAO ou equivalentes (observacao, descricao, etc.)
- DT_REGISTRO_ATENDIMENTO ou equivalentes (data registro, etc.)
- DS_FILIAL ou equivalentes (filial, unidade, loja)
- OPERADORA (opcional)
- CD_USUARIO ou equivalentes (usuario, user, etc.)

## Formatos Suportados

- `.xlsx` (Excel)
- `.xls` (Excel antigo)
- `.csv` (CSV com auto-detecção de encoding e separador)
