# Legislative Definitions Agent

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![Docker](https://img.shields.io/badge/docker-enabled-blue?logo=docker)](https://www.docker.com/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-driven system that leverages Large Language Models (LLMs) for extracting, indexing, retrieving and generating legal definitions from XML-encoded legislation across different jurisdictions, languages and time periods. It is designed to assist legal professionals in efficiently accessing and utilizing legislative definitions, keeping in mind the complexities of legal texts and their evolution over time.

The system functions as a conversational agent, enabling natural language queries tailored to different end-user types, such as lawyers, legislators, and judges. It employs a hybrid approach for the retrieval of legal definitions, integrating dense semantic search with sparse keyword-based methods, and incorporates jurisdiction-aware and point-in-time filtering to ensure jurisdictional and temporal accuracy. In cases where an already established definition is not found, the system leverages Retrieval-Augmented Generation (RAG) techniques to generate a novel one that is grounded in and consistent with in-force legislative documents.

The system is evaluated using automatic quantitative metrics and qualitative assessments from legal experts. The findings demonstrate strong retrieval capabilities but highlight limitations in generating definitions that fully comply with legal standards, underscoring the need for human oversight in legal applications of AI.

------

## Architecture

```mermaid
flowchart LR
    classDef service fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef client fill:#e1f5fe,stroke:#0277bd,stroke-width:2px;
    classDef llm fill:#fff3e0,stroke:#ff6f00,stroke-width:2px;
    classDef user fill:#ffffff,stroke:#333,stroke-width:2px;
    classDef storage fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef input fill:#ffffff,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;
    classDef invisible fill:transparent,stroke:none;

    subgraph UserContext [ ]
        direction TB
        User((User)):::user
        UserQuery[/"Query:<br/>What's the definition<br/>of 'Vessel' in the EU<br/>Legislation? "/]:::input

    end
    style UserContext fill:transparent,stroke:none

    subgraph Frontend [Frontend Interface]
        direction TB
        Streamlit[Streamlit App]:::client
        AgentClient[Agent Client]:::client

        Streamlit <--> AgentClient
    end

    subgraph Backend [Backend Services]
        direction TB
        FastAPI[FastAPI Service]:::service
        
        subgraph LangGraph [LangGraph Agent]
            direction TB
            Start(Start) --> AIAgent[AI Agent]
            AIAgent -- Queries --> DefTool[Definition Retrieval<br>Pipeline]
            DefTool -- Results --> AIAgent
        end
    end


    subgraph Data [Data Layer]
        direction TB
        
        subgraph Milvus [MilvusDB Vector Store]
            direction TB
            MilvusEmb[(Hybrid<br/>Embeddings)]:::storage
        end

        subgraph Exist [eXist-db XML Repo]
            direction TB
            ExistEU[(EurLex<br/>Corpus)]:::storage
            ExistNo[(Normattiva<br/>Corpus)]:::storage
            ExistPdl[(PDL<br/>Corpus)]:::storage
        end
    end

    LLM(LLM Provider):::llm


    UserQuery --> Streamlit
    
    AgentClient <--> FastAPI
    FastAPI <--> LangGraph
    
    AIAgent <--> LLM
    

    ExistEU ---> MilvusEmb
    ExistNo ---> MilvusEmb
    ExistPdl ---> MilvusEmb

    DefTool <--->|hybrid search| MilvusEmb

    DefTool <---> ExistEU
    DefTool <---> ExistNo
    DefTool <---> ExistPdl
```

### Definition Retrieval Pipeline

```mermaid
graph LR
    A([User Query]) --> B(Hybrid Retrieval<br/>Dense + Sparse):::process
    B --> C{Jurisdiction<br>Specified?}:::decision
    C -- Yes --> D(Jurisdiction<br>Filter):::process
    C -- No --> E(Timeline<br>Construction):::process
    D --> F{Found any<br>definitions?}:::decision
    F -- Yes --> E
    F -- No --> G(Definition<br>Generation):::process
    E --> H{Time Frame<br>Specified?}:::decision
    H -- Yes --> I(Temporal<br>Filter):::process
    H -- No --> J(Semantic<br>Reranking):::process
    I --> K{Definitions<br>remaining?}:::decision
    K -- Yes --> J
    K -- No --> G
    J --> L{Confidence<br>threshold met?}:::decision
    L -- Yes --> M([Final Agent<br>Response]):::dashed
    L -- No --> G
    G --> M
```


### Components
- **Backend**: A FastAPI service hosting the LangGraph agent.
- **Frontend**: A Streamlit application providing a chat interface for users, capable of visualizing the inputs and outputs of each node within the retrieval pipeline tool for transparent execution tracing.
- **Client module**: Facilitates communication between the frontend and backend services.
- **Data Layer**:
    - **eXist-db**: XML database for hosting and querying Akoma Ntoso legislative documents.
    - **MilvusDB**: Stores vector embeddings for hybrid semantic search.
- **Orchestration**: LangChain and LangGraph manage the agent's reasoning loop and tool execution.


## Installation

#### Local setup:

```bash
git clone https://github.com/leonardozilli/LegalDefAgent
cd LegalDefAgent
uv sync
source .venv/bin/activate
cp .env-example .env
# Edit .env to add API keys and DB credentials
```

## Usage
The system has a CLI for common operations. You can invoke it with:

```sh
legaldefagent [COMMAND] <args>
```

Available commands:

- `extract-definitions` : Extract definitions from local XML files or eXistDB collections.
- `embed-definitions`   : Compute embeddings for extracted definitions.
- `populate-vectorstore`: Populate the vector store with the generated embeddings and metadata.
- `run-service`         : Start the backend agent service.
- `run-app`             : Start the Streamlit frontend app.

Example workflow:

```sh
# 1. Extract definitions
legaldefagent extract-definitions -s exist

# 2. Compute embeddings
legaldefagent embed-definitions -i data/definitions_corpus/definitions.csv

# 3. Populate the vector store
legaldefagent populate-vectorstore -d data/definitions_corpus/definitions.csv -e data/embeddings/defs_embeddings_hybrid.pkl

# 4. Start the FastAPI server
legaldefagent run-service

# 5. Launch the Streamlit app
legaldefagent run-app
```

The services will be available at:
   - Backend: `http://localhost:8000`
   - Frontend: `http://localhost:3000`

### Docker
To run with Docker Compose:

```bash
# 1. Build the base image
docker build -t legaldefagent-base:latest -f docker/Dockerfile.base .

# 2. Start services
docker-compose up --build
```



---

## Project Structure

```text
├── data/                           # Data storage (corpora, embeddings)
├── docker/                         # Dockerfiles and container configurations
├── evaluation/                     # Evaluation utilities and artefacts
├── src/
│   └── legaldefagent/          
│       ├── api/                    # FastAPI service definitions
│       ├── core/                   # Core logic and tool implementations
│       │   ├── agents/             # LangGraph agent definitions
│       │   ├── db/                 # Database interaction modules
│       │   │   ├── existdb/        # XQuery handlers for eXist-db
│       │   │   └── vectorstore/    # MilvusDB integration
│       │   ├── schema/             # Data schemas and models
│       │   ├── tools/              # Agent Retrieval tools
│       │   └── llm.py              # LLM provider integration
│       ├── frontend/               # Streamlit frontend application
│       ├── ingestion/              # Data ingestion modules
│       ├── cli.py                  # CLI entry point
│       ├── settings.py             # Configuration management
│       └── utils.py                # Utility functions
├── pyproject.toml                  # Project dependencies and metadata
├── config.yaml                     # System configuration
└── README.md                       # This file
```
