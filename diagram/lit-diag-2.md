---
config:
  theme: neo
---
flowchart LR
 subgraph 1_Feature_Engineering_Era["1_Feature_Engineering_Era"]
    direction TB
        FE["1. Feature Engineering Era"]
        FE_M["Method: Handcrafted Features/ML"]
        FE_C["Focus: Linguistic/Structural Cues"]
        FE_O["Outcome: Interpretable but Fragile"]
  end
 subgraph 2_Deep_Learning_NLP_Focus["2_Deep_Learning_NLP_Focus"]
    direction TB
        DL["2. Deep Learning/NLP Focus"]
        DL_M["Method: Embeddings LSTMs/Transformers"]
        DL_C["Focus: Semantic Representation/BERT"]
        DL_O["Outcome: Fine-tunable Captures Semantics"]
  end
 subgraph 3_Contextual_Multimodal_Integration["3_Contextual_Multimodal_Integration"]
    direction TB
        CM["3. Contextual & Multimodal"]
        CM_M["Method: GNNs/Cross-modal Alignment"]
        CM_C["Focus: Social Context/Image-Text Inconsistency"]
        CM_O["Outcome: Holistic Detection/Early Warning"]
  end
 subgraph 4_LLMs_Reasoning_Frameworks["4_LLMs_Reasoning_Frameworks"]
    direction TB
        LLM["4. LLMs & Reasoning"]
        LLM_M["Method: LLMs Zero-shot/Modular Design"]
        LLM_C["Focus: Verification Rationales/Debate"]
        LLM_O["Outcome: Deep Fusion/Verifiable Claims"]
  end
    FE --> FE_M
    FE_M --> FE_C
    FE_C --> FE_O
    DL --> DL_M
    DL_M --> DL_C
    DL_C --> DL_O
    CM --> CM_M
    CM_M --> CM_C
    CM_C --> CM_O
    LLM --> LLM_M
    LLM_M --> LLM_C
    LLM_C --> LLM_O
    A(("Fake News Detection Evolution")) --> 1_Feature_Engineering_Era
    1_Feature_Engineering_Era --> 2_Deep_Learning_NLP_Focus
    2_Deep_Learning_NLP_Focus --> 3_Contextual_Multimodal_Integration
    3_Contextual_Multimodal_Integration --> 4_LLMs_Reasoning_Frameworks
     A:::era
     FE:::era
     FE_M:::method
     FE_C:::concept
     FE_O:::outcome
     DL:::era
     DL_M:::method
     DL_C:::concept
     DL_O:::outcome
     CM:::era
     CM_M:::method
     CM_C:::concept
     CM_O:::outcome
     LLM:::era
     LLM_M:::method
     LLM_C:::concept
     LLM_O:::outcome
    classDef era fill:#263238,stroke:#546E7A,color:#ffffff,stroke-width:2px
    classDef method fill:#80DEEA,stroke:#00BCD4,color:#000000,stroke-width:1px
    classDef concept fill:#D3D3D3,stroke:#555,stroke-width:1px
    classDef outcome fill:#B0C4DE,stroke:#546E7A,color:#000000
