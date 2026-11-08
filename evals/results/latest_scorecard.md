# DocuMind Enterprise Comprehensive 2D Evaluation Scorecard

**Evaluated At**: 20261108_172632 UTC  
**Total Test Cases Evaluated**: 49 across 49 Matrix Cells  
**Target Tenant ID**: `558c93f9-1135-40ba-86f7-cd6676525ef1`  

## 1. Executive Summary: All 7 Enterprise Metrics

| Metric | Stage | Formula / Definition | Corpus Overall Score |
| :--- | :---: | :--- | :---: |
| **Recall@3** | Retrieval | Ground-truth chunk present in top 3 results | **0.78** |
| **Precision@3** | Retrieval | Fraction of top 3 chunks containing ground-truth evidence | **0.60** |
| **MRR** | Retrieval | Mean Reciprocal Rank ($1/\text{rank}$ of first relevant chunk) | **0.72** |
| **NDCG@3** | Retrieval | Normalized Discounted Cumulative Gain with logarithmic rank discount | **0.72** |
| **Faithfulness** | Generation | Ratio of answer claims strictly entailed by context (Hallucination metric) | **1.00** |
| **Answer Relevancy** | Generation | Semantic alignment and completeness of answer to user query | **0.95** |
| **Refusal Accuracy** | Generation | Rejection of unanswerable queries without false refusal of answerable ones | **0.80** |

## 2. Retrieval Evaluation Matrices

### Metric: Recall@3 (Top-3 Retrieval Coverage)

| Document Category | Factual | Semantic | Numerical | Comparative | Multi-hop | Table-based | Unanswerable | Row Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Text-heavy** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | **0.86** |
| **Structured** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | **0.86** |
| **Technical** | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | **0.71** |
| **Long-form** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | **0.71** |
| **Visual** | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | **0.71** |
| **Complex** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | **0.86** |
| **Mixed** | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | **0.71** |
| **Col Avg** | **0.86** | **0.86** | **1.00** | **1.00** | **0.86** | **0.86** | **0.00** | **0.78** |

### Metric: Precision@3 (Top-3 Chunk Relevance Density)

| Document Category | Factual | Semantic | Numerical | Comparative | Multi-hop | Table-based | Unanswerable | Row Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Text-heavy** | 0.67 | 0.33 | 1.00 | 0.67 | 1.00 | 0.33 | 0.00 | **0.57** |
| **Structured** | 1.00 | 0.33 | 1.00 | 1.00 | 1.00 | 0.33 | 0.00 | **0.67** |
| **Technical** | 1.00 | 0.33 | 1.00 | 0.33 | 0.00 | 0.67 | 0.00 | **0.48** |
| **Long-form** | 0.67 | 0.67 | 0.67 | 1.00 | 1.00 | 0.00 | 0.00 | **0.57** |
| **Visual** | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 | 1.00 | 0.00 | **0.67** |
| **Complex** | 1.00 | 1.00 | 1.00 | 0.67 | 0.67 | 0.33 | 0.00 | **0.67** |
| **Mixed** | 0.00 | 0.67 | 0.67 | 1.00 | 1.00 | 0.67 | 0.00 | **0.57** |
| **Col Avg** | **0.76** | **0.48** | **0.86** | **0.81** | **0.81** | **0.48** | **0.00** | **0.60** |

### Metric: MRR (Mean Reciprocal Rank)

| Document Category | Factual | Semantic | Numerical | Comparative | Multi-hop | Table-based | Unanswerable | Row Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Text-heavy** | 0.50 | 0.33 | 1.00 | 1.00 | 1.00 | 0.33 | 0.00 | **0.59** |
| **Structured** | 1.00 | 0.33 | 1.00 | 1.00 | 1.00 | 0.50 | 0.00 | **0.69** |
| **Technical** | 1.00 | 1.00 | 1.00 | 1.00 | 0.17 | 1.00 | 0.00 | **0.74** |
| **Long-form** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.00 | **0.75** |
| **Visual** | 1.00 | 0.20 | 0.50 | 1.00 | 1.00 | 1.00 | 0.00 | **0.67** |
| **Complex** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | **0.86** |
| **Mixed** | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | **0.71** |
| **Col Avg** | **0.79** | **0.69** | **0.93** | **1.00** | **0.88** | **0.73** | **0.00** | **0.72** |

### Metric: NDCG@3 (Normalized Discounted Cumulative Gain)

| Document Category | Factual | Semantic | Numerical | Comparative | Multi-hop | Table-based | Unanswerable | Row Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Text-heavy** | 0.69 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 0.00 | **0.67** |
| **Structured** | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 0.63 | 0.00 | **0.73** |
| **Technical** | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | **0.71** |
| **Long-form** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | **0.71** |
| **Visual** | 1.00 | 0.00 | 0.69 | 1.00 | 1.00 | 1.00 | 0.00 | **0.67** |
| **Complex** | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 0.00 | **0.85** |
| **Mixed** | 0.00 | 0.92 | 1.00 | 1.00 | 1.00 | 0.92 | 0.00 | **0.69** |
| **Col Avg** | **0.81** | **0.70** | **0.96** | **0.99** | **0.86** | **0.72** | **0.00** | **0.72** |

## 3. Generation & LLM-as-a-Judge Matrices

### Metric: Faithfulness (Hallucination Resistance)

| Document Category | Factual | Semantic | Numerical | Comparative | Multi-hop | Table-based | Unanswerable | Row Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Text-heavy** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| **Structured** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| **Technical** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| **Long-form** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| **Visual** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| **Complex** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| **Mixed** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| **Col Avg** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |

### Metric: Answer Relevancy

| Document Category | Factual | Semantic | Numerical | Comparative | Multi-hop | Table-based | Unanswerable | Row Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Text-heavy** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| **Structured** | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | **0.93** |
| **Technical** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| **Long-form** | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.97** |
| **Visual** | 1.00 | 1.00 | 0.80 | 1.00 | 0.20 | 1.00 | 1.00 | **0.86** |
| **Complex** | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | **0.93** |
| **Mixed** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| **Col Avg** | **1.00** | **0.97** | **0.97** | **1.00** | **0.74** | **1.00** | **1.00** | **0.95** |

### Metric: Refusal Accuracy

| Document Category | Factual | Semantic | Numerical | Comparative | Multi-hop | Table-based | Unanswerable | Row Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Text-heavy** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| **Structured** | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | **0.57** |
| **Technical** | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.86** |
| **Long-form** | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.86** |
| **Visual** | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | **0.57** |
| **Complex** | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | **0.86** |
| **Mixed** | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.86** |
| **Col Avg** | **0.57** | **0.71** | **0.86** | **1.00** | **0.57** | **0.86** | **1.00** | **0.80** |

## 4. Composite Quality Matrix

### Metric: Composite Quality Score

| Document Category | Factual | Semantic | Numerical | Comparative | Multi-hop | Table-based | Unanswerable | Row Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Text-heavy** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | **0.95** |
| **Structured** | 1.00 | 1.00 | 1.00 | 1.00 | 0.83 | 1.00 | 0.67 | **0.93** |
| **Technical** | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 0.67 | **0.91** |
| **Long-form** | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | 0.67 | 0.67 | **0.90** |
| **Visual** | 1.00 | 0.67 | 0.93 | 1.00 | 0.73 | 1.00 | 0.67 | **0.86** |
| **Complex** | 1.00 | 1.00 | 1.00 | 1.00 | 0.83 | 1.00 | 0.67 | **0.93** |
| **Mixed** | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | **0.91** |
| **Col Avg** | **0.95** | **0.94** | **0.99** | **1.00** | **0.87** | **0.95** | **0.67** | **0.91** |

