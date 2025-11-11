# Sistema de Monitoramento NIP - Análise de Reclamações e Ouvidoria

**Versão:** 5.0
**Data:** 07/11/2025
**Desenvolvido por:** Cauê Magela [AI-SPEC/CS]

---

## Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura do Sistema](#arquitetura-do-sistema)
3. [Fluxo de Execução](#fluxo-de-execução)
4. [Estrutura de Diretórios](#estrutura-de-diretórios)
5. [Componentes Principais](#componentes-principais)
6. [Sistema de Classificação](#sistema-de-classificação)
7. [Stack Tecnológica](#stack-tecnológica)
8. [Instalação e Configuração](#instalação-e-configuração)
9. [Uso do Sistema](#uso-do-sistema)

---

## Visão Geral

Sistema web para análise, classificação automática e visualização de dados de reclamações e ouvidoria. Implementa Machine Learning (K-NN + Embeddings) para classificação automática de categorias e subcategorias, oferecendo dashboards interativos em tempo real.

### Funcionalidades Principais

- Upload Múltiplo: Suporte para múltiplos arquivos Excel/CSV simultaneamente
- Classificação Inteligente: Machine Learning para auto-classificação de categorias
- Dashboards Interativos: Visualizações dinâmicas com Plotly
- Sistema de Cache: Persistência de dados com SQLite
- Filtros Avançados: Filtros temporais e categóricos
- Exportação Multi-formato: Excel, CSV e JSON
- Interface Moderna: Dark mode com animações CSS

---

## Arquitetura do Sistema

### Padrões Arquiteturais

O sistema utiliza princípios SOLID e Clean Architecture, organizando o código em quatro camadas distintas:

#### 1. Presentation Layer (Interface)

Responsável pela interação com o usuário e renderização da interface.

**Componentes:**
- app.py: Aplicação principal Streamlit
- Componentes de UI no módulo src/presentation/

**Responsabilidades:**
- Renderização da interface de usuário
- Gerenciamento de estado da sessão (st.session_state)
- Captura de eventos e interações do usuário
- Coordenação entre serviços de aplicação

#### 2. Application Layer (Serviços)

Contém a lógica de aplicação e casos de uso.

**Serviços Principais:**
- AssuntoClassifierService: Classificação de categorias principais
- SubAssuntoClassifierService: Classificação de subcategorias
- DashboardService: Geração de visualizações e gráficos
- CacheService: Gerenciamento de cache e persistência
- DataValidatorService: Validação de integridade dos dados
- FuzzyColumnMapper: Mapeamento fuzzy de nomes de colunas

#### 3. Domain Layer (Domínio)

Define as entidades de negócio e contratos de interface.

**Entidades:**
- Complaint: Representação de uma reclamação
- ClassificationResult: Resultado de classificação ML
- ValidationResult: Resultado de validação de dados

**Interfaces:**
- IAssuntoClassifier: Contrato para classificadores
- IEmbeddingService: Contrato para geração de embeddings
- IExcelReader, IReportExporter: Contratos para I/O

#### 4. Infrastructure Layer (Infraestrutura)

Implementações concretas de serviços externos e acesso a dados.

**Serviços:**
- EmbeddingService: Geração de embeddings com Sentence Transformers
- SQLiteCacheService: Persistência em banco SQLite
- ExcelReaderService: Leitura de arquivos Excel/CSV
- ReportExporterService: Exportação de relatórios

---

## Fluxo de Execução

### 1. Processamento de Upload de Dados

**Sequência de Execução:**

1.1. **Recepção do Arquivo**
   - Usuário faz upload de um ou múltiplos arquivos via interface Streamlit
   - Arquivos aceitos: .xlsx, .xls, .csv
   - Sistema armazena temporariamente em memória (UploadedFile)

1.2. **Leitura e Parsing**
   - ExcelReaderService recebe o arquivo
   - Detecta o formato (Excel ou CSV)
   - Para CSV: detecta encoding automaticamente (UTF-8, Latin-1, CP1252)
   - Converte para pandas DataFrame
   - Trata erros de parsing e retorna mensagens descritivas

1.3. **Mapeamento de Colunas**
   - FuzzyColumnMapper analisa os nomes das colunas
   - Executa fuzzy matching para mapear colunas com nomes variados
   - Normaliza para nomes padrão esperados pelo sistema
   - Tolerância: aceita variações de case, acentos, espaços extras

1.4. **Validação de Dados**
   - DataValidatorService verifica:
     - Presença de colunas obrigatórias (NU_REGISTRO, DS_ASSUNTO, etc.)
     - Unicidade de NU_REGISTRO (chave primária)
     - Formato válido de datas em DT_REGISTRO_ATENDIMENTO
     - Valores não nulos em campos críticos
   - Retorna ValidationResult com lista de erros encontrados

1.5. **Concatenação e Deduplicação**
   - Se múltiplos arquivos: concatena DataFrames
   - Remove duplicatas baseando-se em NU_REGISTRO
   - Mantém primeira ocorrência em caso de duplicata
   - Gera estatísticas: total de registros, duplicatas removidas

1.6. **Armazenamento**
   - Atualiza st.session_state['df'] com DataFrame processado
   - SQLiteCacheService armazena histórico de upload
   - Registra log com timestamp, nome do arquivo, quantidade de registros
   - Persiste metadados para auditoria

### 2. Classificação Automática de Categorias

**Sequência de Execução:**

2.1. **Identificação de Registros Não Classificados**
   - AssuntoClassifierService varre o DataFrame
   - Identifica registros onde:
     - DS_ASSUNTO está vazio (null ou string vazia), OU
     - DS_ASSUNTO contém valores genéricos ("OUTROS", "OUTRO", etc.)
   - Conta total de registros para classificação

2.2. **Construção de Texto para Classificação**
   - TextBuilderService processa cada registro:
     - Concatena DS_OBSERVACAO (campo principal)
     - Adiciona SUB_ASSUNTO se disponível (contexto adicional)
     - Remove caracteres especiais e normaliza espaços
     - Limita tamanho máximo do texto (512 tokens)
   - Resultado: texto limpo e contextualizado para embedding

2.3. **Geração de Embeddings**
   - EmbeddingService carrega modelo Sentence Transformer
   - Modelo: paraphrase-multilingual-MiniLM-L12-v2 (384 dimensões)
   - Processamento em mini-batches de 500 registros
   - Para cada batch:
     - Tokeniza textos
     - Gera embeddings normalizados (L2 norm = 1)
     - Armazena em array NumPy
   - Normalização permite uso de similaridade coseno eficiente

2.4. **Classificação K-NN**
   - Carrega embeddings de referência de data/ml/assunto_embeddings.npy
   - Para cada embedding não classificado:
     - Calcula similaridade coseno com todos embeddings de referência
     - Identifica K=5 vizinhos mais próximos
     - Executa votação ponderada:
       - Peso de cada vizinho = similaridade coseno
       - Categoria vencedora = maior soma de pesos
     - Calcula confiança = peso da categoria vencedora / soma total

2.5. **Aplicação de Threshold**
   - Threshold definido: 0.45
   - Se confiança >= 0.45:
     - Atribui categoria automaticamente
     - Marca como classificado automaticamente (flag interno)
   - Se confiança < 0.45:
     - Mantém campo vazio para revisão manual
     - Registra em log para análise posterior

2.6. **Atualização do DataFrame**
   - Atualiza coluna DS_ASSUNTO no DataFrame
   - Incrementa contadores de estatísticas
   - Atualiza st.session_state['df']
   - Executa st.rerun() para atualizar interface

### 3. Geração de Dashboards

**Sequência de Execução:**

3.1. **Recebimento do DataFrame Filtrado**
   - DashboardService recebe DataFrame após aplicação de filtros
   - Valida presença de colunas necessárias para cada gráfico

3.2. **Análise Temporal**
   - Processamento de gráficos semanais:
     - Converte DT_REGISTRO_ATENDIMENTO para datetime
     - Agrupa por semana (período: segunda a domingo)
     - Calcula contagem de reclamações por semana
     - Identifica picos (valores acima da média + 1 desvio padrão)
     - Gera hover tooltips com Top 5 categorias da semana
   - Processamento de gráficos mensais:
     - Agrupa por mês
     - Segue lógica similar à análise semanal

3.3. **Análise Categórica**
   - Processamento de categorias principais (DS_ASSUNTO):
     - Executa value_counts() no DataFrame
     - Ordena por frequência decrescente
     - Remove categoria "OUTROS" se especificado
     - Gera gráfico de barras horizontais
   - Processamento de subcategorias (SUB_ASSUNTO):
     - Lógica análoga às categorias principais
     - Considera contexto de categorização hierárquica

3.4. **Análise de Rankings**
   - Ranking de Filiais:
     - Agrupa por DS_FILIAL
     - Ordena por quantidade de reclamações
     - Identifica filiais com maior volume
   - Ranking de Operadoras (se disponível):
     - Verifica existência da coluna OPERADORA
     - Executa agregação similar

3.5. **Geração de Gráficos Plotly**
   - Para cada tipo de gráfico:
     - Define schema de cores (paleta personalizada)
     - Configura eixos, títulos, legendas
     - Adiciona interatividade:
       - Hover tooltips informativos
       - Zoom e pan habilitados
       - Download como PNG em alta resolução
     - Aplica tema dark mode
     - Retorna objeto Figure do Plotly

3.6. **Renderização na Interface**
   - Streamlit renderiza cada gráfico com st.plotly_chart()
   - Gráficos organizados em layout responsivo (colunas)
   - Botões de download de imagem PNG associados

### 4. Exportação de Dados

**Sequência de Execução:**

4.1. **Seleção de Formato**
   - Usuário escolhe formato via interface: Excel, CSV ou JSON
   - Sistema captura seleção e DataFrame filtrado atual

4.2. **Processamento pelo ReportExporterService**
   - Recebe DataFrame e formato selecionado
   - Aplica formatações específicas do formato

4.3. **Geração de Arquivo Excel**
   - Cria arquivo .xlsx em memória (BytesIO)
   - Estrutura:
     - Aba "Dados": DataFrame completo com formatação condicional
     - Aba "Resumo": Estatísticas agregadas (contagens por categoria, filial, etc.)
   - Formatação aplicada:
     - Cabeçalhos em negrito
     - Colunas de data formatadas (DD/MM/AAAA)
     - Largura automática de colunas
     - Bordas e cores alternadas em linhas

4.4. **Geração de Arquivo CSV**
   - Exporta DataFrame para CSV
   - Configurações:
     - Encoding: UTF-8 with BOM (compatibilidade Excel)
     - Separador: vírgula
     - Quote character: aspas duplas
     - index=False (remove índice do pandas)

4.5. **Geração de Arquivo JSON**
   - Exporta DataFrame para JSON
   - Configurações:
     - Orientação: records (lista de objetos)
     - force_ascii=False (preserva caracteres especiais)
     - indent=2 (pretty-print para leitura humana)

4.6. **Download**
   - Sistema gera nome do arquivo: relatorio_nip_YYYYMMDD_HHMMSS.[ext]
   - Streamlit apresenta botão de download
   - Navegador executa download automático ao clicar

---

## Estrutura de Diretórios

```
backup_v5/
│
├── app.py                                    # Aplicação principal Streamlit
├── requirements.txt                          # Dependências Python
├── README.md                                 # Documentação técnica
├── CONTEXTO.md                               # Contexto do projeto
├── CLAUDE.md                                 # Instruções para Claude AI
│
├── data/                                     # Dados e modelos treinados
│   └── ml/
│       ├── assunto_embeddings.npy            # Embeddings pré-calculados (Categorias)
│       ├── assunto_reference.pkl             # Dados de referência (Categorias)
│       ├── subassunto_embeddings.npy         # Embeddings pré-calculados (SubCategorias)
│       └── subassunto_reference.pkl          # Dados de referência (SubCategorias)
│
├── output_data/                              # Dados processados e exportados
├── planilhas_parcialmente_classificada/      # Resultados parciais
├── planilhas_total_classificada/             # Resultados completos
│
├── scripts/                                  # Scripts auxiliares
│   ├── analysis/                             # Scripts de análise
│   ├── batch/                                # Processamento em lote
│   └── training/                             # Treinamento de modelos
│
└── src/                                      # Código-fonte principal
    ├── interfaces/                           # Contratos (Interfaces)
    ├── models/                               # Entidades de domínio
    ├── services/                             # Serviços da aplicação
    ├── infrastructure/                       # Infraestrutura
    ├── presentation/                         # Camada de apresentação
    └── core/                                 # Núcleo da aplicação
```

---

## Componentes Principais

### ExcelReaderService

**Responsabilidade:** Leitura e parsing de arquivos Excel/CSV

**Funcionalidades:**
- Suporta múltiplos formatos (.xlsx, .xls, .csv)
- Detecção automática de encoding (UTF-8, Latin-1, CP1252)
- Tratamento robusto de erros de parsing
- Conversão para pandas DataFrame

**Interface:** IExcelReader

### FuzzyColumnMapper

**Responsabilidade:** Mapeamento inteligente de colunas

**Tecnologia:** RapidFuzz para fuzzy string matching

**Funcionalidades:**
- Fuzzy matching para nomes de colunas variados
- Tolerância a variações (espaços, acentos, case)
- Score de similaridade configurável (threshold: 80)
- Mapeamento de colunas obrigatórias

**Interface:** IColumnMapper

### AssuntoClassifierService

**Responsabilidade:** Classificação automática de categorias (DS_ASSUNTO)

**Tecnologia:** K-NN com embeddings semânticos

**Parâmetros:**
- Threshold de confiança: 0.45
- K-neighbors: 5
- Batch size: 500 registros
- Modelo: paraphrase-multilingual-MiniLM-L12-v2 (384 dimensões)

**Métricas de Performance:**
- Acurácia: 85-90%
- Precision: ~88%
- Recall: ~82%
- Velocidade: ~500 registros/batch

**Interface:** IAssuntoClassifier

### SubAssuntoClassifierService

**Responsabilidade:** Classificação automática de subcategorias (SUB_ASSUNTO)

**Tecnologia:** Similar ao AssuntoClassifierService

**Diferencial:** Considera contexto da categoria principal (DS_ASSUNTO) como feature adicional

**Interface:** IAssuntoClassifier

### EmbeddingService

**Responsabilidade:** Geração de embeddings semânticos

**Modelo:** Sentence Transformers - paraphrase-multilingual-MiniLM-L12-v2

**Características:**
- Embeddings de 384 dimensões
- Normalização L2 (cosine similarity)
- Processamento em batches configuráveis
- Cache de embeddings para otimização

**Método Principal:**
```
generate_embeddings(texts: List[str], batch_size: int = 500) -> np.ndarray
```

**Interface:** IEmbeddingService

### DashboardService

**Responsabilidade:** Geração de visualizações interativas

**Gráficos Disponíveis:**
- Temporal: Análise semanal e mensal
- Categórico: Distribuição de assuntos e sub-assuntos
- Ranking: Filiais e operadoras
- Comparativo: Antes/Depois da classificação

**Tecnologia:** Plotly para gráficos interativos

**Interface:** IDashboardGenerator

### SQLiteCacheService

**Responsabilidade:** Persistência de dados e cache

**Funcionalidades:**
- Histórico de uploads com timestamp
- Estatísticas de uso do sistema
- Cache de queries frequentes
- Limpeza automática de dados expirados (TTL: 30 dias)

**Esquema de Banco:**
- Tabela: upload_history
  - id: INTEGER PRIMARY KEY
  - filename: TEXT
  - upload_date: TIMESTAMP
  - record_count: INTEGER
  - user_session: TEXT

**Interface:** ICacheService

### ReportExporterService

**Responsabilidade:** Exportação de relatórios em múltiplos formatos

**Formatos Suportados:**
- Excel (.xlsx): Com formatação, múltiplas abas, estilos
- CSV (UTF-8 with BOM): Compatível com Excel
- JSON (pretty-printed): Estruturado para APIs

**Funcionalidades Específicas:**
- Excel: Abas separadas para dados e resumo estatístico
- CSV: Encoding com BOM para caracteres especiais
- JSON: Indentação para legibilidade

**Interface:** IReportExporter

---

## Sistema de Classificação

### Arquitetura do Classificador

O sistema de classificação utiliza técnica de K-Nearest Neighbors (K-NN) combinada com embeddings semânticos para classificação automática.

**Componentes:**

1. **Sentence Transformer Model**
   - Modelo: paraphrase-multilingual-MiniLM-L12-v2
   - Dimensões: 384
   - Multilingue: suporta português, inglês, espanhol

2. **Embedding Generation**
   - Entrada: Texto concatenado (DS_OBSERVACAO + contexto)
   - Saída: Vetor de 384 dimensões normalizado (L2 norm = 1)

3. **Cosine Similarity**
   - Métrica de distância: Similaridade coseno
   - Fórmula: similarity(A,B) = (A · B) / (||A|| × ||B||)
   - Range: [0, 1] onde 1 = idêntico, 0 = ortogonal

4. **K-NN Classification**
   - K = 5 (5 vizinhos mais próximos)
   - Votação ponderada por similaridade
   - Categoria vencedora: maior soma de pesos

5. **Threshold Application**
   - Threshold: 0.45
   - Confiança >= 0.45: classificação automática
   - Confiança < 0.45: marcação para revisão manual

### Processo de Treinamento

**Etapa 1: Preparação de Dados de Referência**

1.1. Coleta de reclamações já classificadas manualmente
1.2. Extração de texto completo + categoria conhecida
1.3. Limpeza e normalização de textos

**Etapa 2: Geração de Embeddings de Referência**

Executar script de treinamento:
```bash
python scripts/training/train_assunto_classifier_fast.py
```

Processo:
- Carrega dataset de referência
- Gera embeddings para todos os textos
- Salva embeddings em data/ml/assunto_embeddings.npy
- Salva metadados em data/ml/assunto_reference.pkl

**Etapa 3: Validação de Acurácia**

Executar script de teste:
```bash
python scripts/training/test_accuracy_fast.py
```

Métricas avaliadas:
- Acurácia geral
- Precision por categoria
- Recall por categoria
- F1-Score
- Matriz de confusão

### Otimizações Implementadas

1. **Batch Processing**
   - Embeddings gerados em batches de 500 registros
   - Reduz overhead de GPU/CPU
   - Memória eficiente

2. **Caching**
   - Embeddings de referência carregados uma vez
   - Reutilizados para todas as classificações
   - Salvo em arquivos .npy para acesso rápido

3. **Normalização**
   - Embeddings normalizados (L2 norm)
   - Permite uso de dot product ao invés de cálculo completo de coseno
   - Speedup significativo

---

## Stack Tecnológica

### Frontend

**Streamlit 1.28+**
- Framework web Python
- Renderização reativa de componentes
- Gerenciamento de estado via session_state

**Plotly 5.17+**
- Visualizações interativas
- Suporte a zoom, pan, hover tooltips
- Export de gráficos em alta resolução

**CSS3**
- Animações e transições
- Dark mode customizado
- Layout responsivo

### Backend

**Python 3.9+**
- Linguagem core do sistema
- Type hints para segurança de tipos

**Pandas 2.1+**
- Manipulação e análise de dados tabulares
- Operações vetorizadas de alto desempenho

**NumPy 1.24+**
- Computação numérica
- Arrays multidimensionais
- Operações matemáticas otimizadas

### Machine Learning

**Sentence Transformers 2.2+**
- Geração de embeddings semânticos
- Modelos pré-treinados multilingues

**Scikit-learn 1.3+**
- Algoritmos de ML (K-NN)
- Métricas de avaliação
- Normalização e pré-processamento

**Transformers 4.35+**
- Biblioteca HuggingFace
- Modelos de linguagem state-of-the-art

### Persistência

**SQLite3**
- Banco de dados relacional embarcado
- Cache e histórico de uploads
- Zero configuração

**Pickle**
- Serialização de objetos Python
- Armazenamento de metadados e modelos

**NumPy .npy**
- Formato binário eficiente para arrays
- Armazenamento de embeddings

### Utilidades

**RapidFuzz**
- Fuzzy string matching otimizado
- Algoritmo Levenshtein Distance

**OpenPyXL**
- Leitura e escrita de arquivos Excel (.xlsx)
- Suporte a formatação avançada

**Python-dateutil**
- Parsing flexível de datas
- Manipulação de timezones

---

## Instalação e Configuração

### Pré-requisitos

- Python 3.9 ou superior
- pip 21.0+ (gerenciador de pacotes Python)
- 4GB RAM mínimo (recomendado: 8GB para datasets grandes)
- Espaço em disco: 2GB (modelos + dependências)

### Instalação Passo a Passo

**1. Preparação do Ambiente**

```bash
cd backup_v5
```

**2. Instalação de Dependências**

```bash
pip install -r requirements.txt
```

Dependências principais instaladas:
- streamlit>=1.28.0
- pandas>=2.0.0
- plotly>=5.17.0
- sentence-transformers>=2.2.0
- scikit-learn>=1.3.0
- openpyxl>=3.1.0

**3. Treinamento de Modelos (Opcional)**

Se não existirem modelos pré-treinados em data/ml/:

```bash
python scripts/training/train_assunto_classifier_fast.py
python scripts/training/train_subassunto_classifier_fast.py
```

Tempo estimado: 10-30 minutos dependendo do tamanho do dataset de referência.

**4. Execução da Aplicação**

```bash
streamlit run app.py
```

Aplicação será iniciada em: http://localhost:8501

### Configuração Avançada

**Variáveis de Ambiente**

```bash
export STREAMLIT_SERVER_PORT=8080
export STREAMLIT_THEME_BASE="dark"
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
```

**Ajuste de Performance**

Editar parâmetros em src/services/AssuntoClassifierService.py:

```python
BATCH_SIZE = 500
THRESHOLD = 0.45
K_NEIGHBORS = 5
```

**Customização de Modelos**

Alterar modelo de embedding em src/services/EmbeddingService.py:

```python
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
```

Modelos alternativos:
- paraphrase-multilingual-mpnet-base-v2 (mais preciso, mais lento)
- distiluse-base-multilingual-cased-v2 (mais rápido, menor dimensão)

---

## Uso do Sistema

### 1. Upload de Dados

**Procedimento:**

1. Acessar aplicação via navegador (http://localhost:8501)
2. Expandir seção "Upload de Dados" na sidebar
3. Clicar em "Browse files" ou arrastar arquivo(s)
4. Aguardar processamento (barra de progresso exibida)
5. Confirmar mensagem de sucesso com estatísticas de upload

**Colunas Obrigatórias:**

- NU_REGISTRO: Identificador único (número ou string)
- DS_ASSUNTO: Categoria principal
- SUB_ASSUNTO: Subcategoria
- DS_OBSERVACAO: Texto descritivo da reclamação
- DT_REGISTRO_ATENDIMENTO: Data de registro (formato: DD/MM/AAAA ou AAAA-MM-DD)
- DS_FILIAL: Nome da filial

**Validações Executadas:**

- Unicidade de NU_REGISTRO
- Presença de todas as colunas obrigatórias
- Formato válido de datas
- Remoção de duplicatas automática

### 2. Classificação Automática

**Procedimento:**

1. Expandir seção "Classificação" na sidebar
2. Verificar estatísticas de registros não classificados
3. Clicar em "Classificar Categorias" (DS_ASSUNTO)
4. Aguardar processamento com barra de progresso
5. Conferir estatísticas pós-classificação
6. (Opcional) Clicar em "Classificar SubCategorias" (SUB_ASSUNTO)

**Comportamento:**

- Registros com confiança >= 0.45: classificados automaticamente
- Registros com confiança < 0.45: mantidos vazios para revisão manual
- Estatísticas exibidas: total classificado, pendente, taxa de sucesso

### 3. Visualização de Dados

**Abas Disponíveis:**

- Painel: Visão consolidada com todos os gráficos
- Análise Temporal: Gráficos semanais e mensais com detecção de picos
- Categorização: Distribuição de DS_ASSUNTO e SUB_ASSUNTO
- Filiais: Ranking de filiais por volume de reclamações
- Operadoras: Análise por operadora (se coluna OPERADORA existir)
- Auditoria: Histórico de uploads e estatísticas de uso

**Interatividade:**

- Hover sobre pontos: exibe tooltips com detalhes
- Zoom: clicar e arrastar para zoom em área específica
- Pan: shift + arrastar para navegação
- Download: botão para salvar gráfico como PNG HD

### 4. Aplicação de Filtros

**Procedimento:**

1. Expandir seção "Filtros" na sidebar
2. Selecionar intervalo de datas:
   - Data Início: seletor de calendário
   - Data Fim: seletor de calendário
3. (Opcional) Selecionar categorias específicas
4. Clicar em "Aplicar Filtros"
5. Gráficos serão atualizados automaticamente

**Filtros Disponíveis:**

- Temporal: Intervalo de datas em DT_REGISTRO_ATENDIMENTO
- Categórico: Seleção múltipla de DS_ASSUNTO
- Filial: Seleção múltipla de DS_FILIAL

### 5. Exportação de Relatórios

**Formatos Disponíveis:**

**Excel (.xlsx)**
- Estrutura multi-abas:
  - Aba "Dados": DataFrame completo formatado
  - Aba "Resumo": Estatísticas agregadas
- Formatação aplicada: cabeçalhos em negrito, colunas auto-ajustadas

**CSV (UTF-8 with BOM)**
- Compatível com Excel
- Separador: vírgula
- Preserva caracteres especiais

**JSON (pretty-printed)**
- Estrutura: array de objetos
- Indentação: 2 espaços
- UTF-8 sem escaping de caracteres

**Procedimento:**

1. Navegar até seção "Exportar"
2. Selecionar formato desejado
3. Clicar no botão correspondente
4. Arquivo será baixado automaticamente

---

## Notas Técnicas

### Validações Críticas

1. **Duplicatas:** NU_REGISTRO deve ser único. Sistema remove duplicatas automaticamente mantendo primeira ocorrência.

2. **Colunas Obrigatórias:** Sistema valida presença antes de processar. Upload rejeitado se alguma coluna estiver ausente.

3. **Formatos de Data:** Sistema tenta múltiplos formatos comuns. Falha se nenhum formato for reconhecido.

4. **Campos Vazios:** Campos críticos (NU_REGISTRO, DT_REGISTRO_ATENDIMENTO) não podem estar vazios.

### Performance

**Benchmarks (hardware médio: i5, 8GB RAM):**

- Upload de arquivo 10k registros: ~5 segundos
- Classificação de 10k registros: ~2-3 minutos
- Geração de gráficos: <1 segundo
- Exportação Excel 50k registros: ~10 segundos

**Otimizações Aplicadas:**

- Processamento em batches para ML
- Cache de embeddings de referência
- Queries otimizadas em pandas (operações vetorizadas)
- Lazy loading de componentes

### Troubleshooting

**Erro: "Modelo não treinado"**

Causa: Arquivos de modelo não encontrados em data/ml/

Solução:
```bash
python scripts/training/train_assunto_classifier_fast.py
```

**Erro: "Memória insuficiente"**

Causa: Dataset muito grande para RAM disponível

Solução:
- Reduzir BATCH_SIZE em AssuntoClassifierService.py
- Processar dados em partes menores
- Aumentar RAM do sistema

**Gráficos não atualizam após filtro**

Causa: Cache do Streamlit retendo estado anterior

Solução:
- Clicar em "Aplicar Filtros" novamente
- Ou usar botão "Limpar Cache/Memória" na sidebar

**Upload rejeitado: "Coluna X não encontrada"**

Causa: Nome de coluna no arquivo difere do esperado

Solução:
- Verificar nomes de colunas obrigatórias
- FuzzyMapper tem tolerância limitada (score >= 80)
- Renomear colunas para match exato se necessário

---

## Desenvolvimento

### Arquitetura de Testes

**Scripts de Teste Disponíveis:**

```bash
python scripts/training/test_accuracy_fast.py
```

Executa:
- Validação cruzada 5-fold
- Cálculo de métricas (accuracy, precision, recall, F1)
- Geração de matriz de confusão

```bash
python scripts/analysis/analyze_results.py
```

Executa:
- Análise estatística de resultados de classificação
- Distribuição de confiança
- Identificação de categorias problemáticas

### Estrutura de Logs

Sistema gera logs em múltiplos níveis:

- INFO: Operações normais (upload, classificação)
- WARNING: Situações atípicas (baixa confiança, colunas faltantes)
- ERROR: Erros de processamento

Logs armazenados em:
- Console (stdout)
- Arquivo: logs/app.log (se configurado)

### Contribuindo

Projeto interno. Para sugestões ou melhorias, contatar:
**Cauê Magela [AI-SPEC/CS]**

---

## Licença

Uso interno - Todos os direitos reservados

---

**Última Atualização:** 07/11/2025
**Versão da Documentação:** 5.0
