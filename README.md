# Ontology-aware KG & KG-RAG

지식그래프 기반 산업 규격 문서 분석 · RAG 프레임워크


## Overview

**Ontology-aware KG-RAG**는 산업 규격(SPEC) 문서의
복잡한 규칙·조건·예외 조항을 **Ontology + Knowledge Graph(KG)** 기반으로 자동 분석·구조화하고,
이를 **RAG(Retrieval-Augmented Generation)** 모델에서 활용할 수 있도록 구성한 프레임워크다.

본 저장소는 두 개의 핵심 모듈로 구성된다.

1. **Ontology-KG**

   * 문서 구조 분석
   * 삼중항(triple) 추출
   * 동의어 정제
   * Neo4j 기반 KG 구축

2. **Ontology-aware KG-RAG**

   * KG 기반 검색(KG Retriever)
   * MindMap 기반 엔티티 중심 QA
   * 임베딩 기반 title/entity 매칭 및 증거 추출

---

## Repository Structure

```
ontology_based_kg_paper
│
├── ontology_kg/
│   ├── ontology/          # 문서 구조 분석 및 전처리
│   ├── triple/            # 표/텍스트 삼중항 추출
│   └── synonym_pruning/   # 동의어 사전 구축 + KG 정제
│
├── ontology_kg_rag/
│   ├── kg_retriever/      # title embedding + KG 기반 RAG
│   └── mindmap/           # entity embedding + Neo4j 기반 QA
│
└── 각 파일마다 확인/
    ├── triplet/           # 추출된 triple
    ├── synomy_output/           # 동의어 사전
    ├── pruning_ouput/           # Neo4j 넣는 최종 KG
    ├── kg_rag/            # title embedding
    ├── QA_data/                # QA dataset
    └── output/            # RAG/MindMap 결과
```

---

## 1. Ontology-KG Module

### 주요 기능

* **문서 전처리 & 섹션 구조 자동 인식**
* **텍스트/표 기반 Triple Extraction (LLM 기반)**
* **동의어 기반 entity/relation 정규화**
* **그래프 정제 & Neo4j 적재**

### 실행 순서

```
1. python ontology.py                # 문서 전처리 및 구조 라벨링
2. python triple_table.py            # 표 기반 triple 추출
3. python triple_text.py             # 텍스트 triple 추출
4. python synonym.py                 # 동의어 사전 구축
5. python pruning.py                 # KG 정제
6. python neo4j.py                   # KG → Neo4j 적재
```

---

## 2. Ontology-aware KG-RAG Module

### KG Retriever

* title 임베딩 기반 섹션 후보군 선택
* local/global triple 수집 후 ranking
* LLM 기반 답변 생성

### MindMap

* 질문 → 엔티티 추출
* 엔티티 임베딩 기반 KG 매칭
* Neo4j에서 근거(section) 검색
* 근거를 바탕으로 LLM이 최종 답변 생성

### 실행 순서

```
# KG Retriever
1. python title_embedding.py
2. python main.py

# MindMap
1. python entity_embedding.py
2. python MindMap.py
```

---

## Environment

* Python 3.10+
* Neo4j Aura / Local Neo4j
* 주요 패키지: pandas, sentence-transformers, openai, neo4j, langchain

---

## Project Goal

* 산업 규격 문서의 **조건·규칙·예외 조항 자동 구조화**
* 문서 전체를 **온톨로지 기반 지식그래프**로 변환
* KG를 활용한 **정확한 근거 기반 QA / 고신뢰 RAG 시스템** 구축

---

## Contact

* 한양대학교 IDSL
* [jiinpark@hanyang.ac.kr](mailto:jiinpark@hanyang.ac.kr)
* [hyunajeon@hanyang.ac.kr](mailto:hyunajeon@hanyang.ac.kr)
* [yoonseolee@hanyang.ac.kr](mailto:yoonseolee@hanyang.ac.kr)

