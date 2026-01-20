---
config:
  look: neo
  theme: mc
---
flowchart BT 
 subgraph ENV["Google Colab Pro"]
        PY["Python 3.10"]
        RAM["System RAM 53 GB"]
        GPU["GPU RAM 22.5 GB"]
        DISK["Disk 235.7 GB"]
  end
 subgraph MODELS["LLMs and Models"]
        GENMA["Genma 3N (2B) by Unsloth"]
        CLIP["OpenAI CLIP"]
  end
 subgraph LOGGING["Logging and Tracking"]
        WNB["Weights and Biases"]
  end
    ENV --> MODELS & LOGGING
    MODELS --> FND
    LOGGING --> FND
