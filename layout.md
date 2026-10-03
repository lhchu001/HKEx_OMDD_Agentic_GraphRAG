hkex-omdd-evidence-rag/
├── README.md
├── .gitignore
├── .env
├── requirements.txt
├── LoadLLM.py
├── LLM/                    # Provider adapters imported by LoadLLM.py
├── indexing.py
├── graph_indexing.py
├── graph_store.py
├── retrieving.py
├── evaluation/
│   ├── test_cases.json
│   └── evaluate.py
├── docs/                   # Local source PDFs; excluded by default
└── data/                   # Generated indexes and graph; excluded