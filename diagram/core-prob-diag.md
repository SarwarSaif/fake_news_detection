flowchart TD
    classDef stage fill:#263238,stroke:#546E7A,color:#ffffff,stroke-width:2px;
    classDef problem fill:#FF6347,stroke:#C0392B,color:#ffffff;
    classDef component fill:#4DD0E1,stroke:#00BCD4,color:#000000,stroke-width:1px;
    classDef output fill:#90EE90,stroke:#3C993C;
    classDef question fill:#FFB74D,stroke:#FF9800,color:#000000,stroke-width:1px;

    subgraph Core_Problem_Statement
        P1[Current LLM Approaches &#40;CoT, Vectors&#41;]:::problem
        P2[Problem: Operate as Black Box]:::problem
        P3[Problem: Lack Core OSINT Principles]:::problem
        
        P1 --> P2
        P1 --> P3
    end
    
    P3 --> RQ_Start[Central Research Question: How to build an effective, interpretable framework?]:::question
    
    %% The Proposed Solution Framework
    subgraph Proposed_LLM_OSINT_Framework
        direction LR
        
        S1[LLM-Enhanced Multimodal Core]:::component
        
        subgraph Key_Integration_Modules
            direction TB
            S2_1[Cross-Modal Dissimilarity Detection]:::component
            S2_2[Deepfake Identification Module]:::component
            S2_3[External Knowledge Reasoning - OSINT Principles]:::component
        end
        
        S1 --> S2_1
        S1 --> S2_2
        S1 --> S2_3
        
        S2_1 & S2_2 & S2_3 --> S3[Final Output: Structured Misinformation Verdict]:::output
    end
    
    RQ_Start --> Proposed_LLM_OSINT_Framework
    
    %% Evaluation Section based on Sub-Questions
    subgraph Evaluation_and_Research_Outcomes
        direction TB
        E1{Sub-Q: Detect misinformation across modalities?}:::question
        E2{Sub-Q: Role of Dissimilarity/Deepfake in Accuracy?}:::question
        E3{Sub-Q: External Knowledge vs. Factual Verification?}:::question
        
        S3 --> E1
        S3 --> E2
        S3 --> E3

        E1 --> Final_Outcomes[Outcomes: Compare Accuracy, Precision, Recall, and Interpretability]:::output
        E2 --> Final_Outcomes
        E3 --> Final_Outcomes
    end