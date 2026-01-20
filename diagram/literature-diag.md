flowchart TD
    classDef era fill:#263238,stroke:#546E7A,color:#ffffff,stroke-width:2px;
    classDef method fill:#80DEEA,stroke:#00BCD4,color:#000000,stroke-width:1px;
    classDef concept fill:#D3D3D3,stroke:#555,stroke-width:1px;
    classDef author fill:#FFB74D,stroke:#FF9800,color:#000000;
    classDef outcome fill:#B0C4DE,stroke:#546E7A,color:#000000;
    
    A((Fake News Detection Research Evolution)):::era
    
    %% Phase 1: Feature Engineering Era
    subgraph 1_Feature_Engineering_Era
        direction TB
        B[Early Approaches - Feature Engineering]:::method
        B --> B1[Linguistic and Structural Cues]:::concept
        B --> B2[Pre-trained ML Classifiers]:::concept
        
        %% Syntax changed to user's request
        B1 --> B1a[Castillo et al #40;2011#41; - Twitter Features]:::author
        B1 --> B1b[Gupta et al #40;2013#41; - Multimedia Signals]:::author
        B1 --> B1c[Biyani et al #40;2016#41; and Sun et al #40;2013#41; - Stylistic Markers]:::author
        B2 --> B2a[Result: Easy to Interpret but Fragile]:::outcome
    end
    
    A --> 1_Feature_Engineering_Era
    
    %% Phase 2: Deep Learning / NLP Focus
    subgraph 2_Deep_Learning_NLP_Focus
        direction LR
        C[NLP and Text Representations]:::method
        C --> C1[Traditional: BoW TF-IDF]:::concept
        C1 --> C2[Word Embeddings #40;Wang 2017 and Karimi 2018#41;]:::author
        C2 --> C3[Contextualized Embeddings LSTMs #40;Rashkin 2017#41;]:::author
        C3 --> C4[Transformers: BERT RoBERTA #40;Kaliyar 2021 and Mridha 2021#41;]:::author
        C4 --> C4a[Result: Fine-tunable captures Semantics]:::outcome
    end
    
    1_Feature_Engineering_Era --> 2_Deep_Learning_NLP_Focus
    
    %% Phase 3: Contextual & Multimodal Integration
    subgraph 3_Contextual_Multimodal_Integration
        direction LR
        
        subgraph Social_Context_Methods
            direction TB
            D[Social Context Approaches]:::method
            D --> D1[Features: Credibility Engagement]:::concept
            D1 --> D1a[Vosoughi et al #40;2018#41; - Spreading Speed]:::author
            D --> D2[Graph-based Methods #40;Bian 2020 and Mehta 2022#41;]:::author
            D2 --> D2a[Goal: Early Warning - Scalability Challenge]:::outcome
        end
        
        subgraph Multimodal_Evolution
            direction TB
            E[Multimodal Approaches]:::method
            E --> E1[Text and Images #40;Jin 2017 and Wang 2018#41;]:::author
            E1 --> E2[Cross-modal Semantic Alignment #40;Boididou 2018 and Qian 2021#41;]:::author
            E2 --> E3[Vision-Language Models: CLIP #40;Wang 2024#41;]:::author
            E3 --> E3a[Result: Detect Image-Text Inconsistencies]:::outcome
        end
        
        Social_Context_Methods --> Link_Node[(Integration)]:::concept
        Multimodal_Evolution --> Link_Node
        
    end
    
    2_Deep_Learning_NLP_Focus --> 3_Contextual_Multimodal_Integration
    
    %% Phase 4: Large Language Models (LLMs) & Reasoning
    subgraph 4_LLMs_Reasoning_Frameworks
        direction TB
        F[Large Language Models - LLMs]:::method
        F --> F1[Detection: Zero-shot Few-shot]:::concept
        F1 --> F1a[Rationales Hu et al #40;2024#41; and Man et al #40;2025#41;]:::author
        F --> F2[Advanced Reasoning Frameworks]:::concept
        F2 --> F2a[Courtroom Debate Jin et al #40;2025#41;]:::author
        F2 --> F2b[Multimodal Reasoning Wang et al #40;2024#41;]:::author
        F2 --> F2c[Structured Modular Design Martirano etil #40;2025#41;]:::author
        F2c --> F2c1[Goal: Deep Fusion Cross-modal Relationships]:::outcome
    end
    
    3_Contextual_Multimodal_Integration --> 4_LLMs_Reasoning_Frameworks




    flowchart TD
    classDef era fill:#263238,stroke:#546E7A,color:#ffffff,stroke-width:2px;
    classDef method fill:#80DEEA,stroke:#00BCD4,color:#000000,stroke-width:1px;
    classDef author fill:#FFB74D,stroke:#FF9800,color:#000000;
    classDef outcome fill:#B0C4DE,stroke:#546E7A,color:#000000;

    subgraph 1_Feature_Engineering_Era
        direction TB
        Title1[Feature Engineering Era: Early Approaches]:::era
        B[Method: Handcrafted Features]:::method
        
        B --> B1[Concept: Linguistic and Structural Cues]:::method
        B --> B2[Concept: Pre-trained ML Classifiers]:::method
        
        B1 --> B1a[Castillo et al #40;2011#41; - Twitter Features]:::author
        B1 --> B1b[Gupta et al #40;2013#41; - Multimedia Signals]:::author
        B1 --> B1c[Biyani et al #40;2016#41; and Sun et al #40;2013#41; - Stylistic Markers]:::author
        B2 --> B2a[Outcome: Easy to Interpret but Fragile]:::outcome
    end




flowchart TD
    classDef era fill:#263238,stroke:#546E7A,color:#ffffff,stroke-width:2px;
    classDef method fill:#80DEEA,stroke:#00BCD4,color:#000000,stroke-width:1px;
    classDef author fill:#FFB74D,stroke:#FF9800,color:#000000;
    classDef outcome fill:#B0C4DE,stroke:#546E7A,color:#000000;

    subgraph 2_Deep_Learning_NLP_Focus
        direction TB
        Title2[Deep Learning Era: NLP and Semantic Representations]:::era
        C[Method: Advanced Text Representations]:::method

        C --> C1[Concept: Traditional BoW TF-IDF]:::method
        C1 --> C2[Word Embeddings #40;Wang 2017 and Karimi 2018#41;]:::author
        C2 --> C3[Contextualized Embeddings LSTMs #40;Rashkin 2017#41;]:::author
        C3 --> C4[Transformers: BERT RoBERTA #40;Kaliyar 2021 and Mridha 2021#41;]:::author
        C4 --> C4a[Outcome: Fine-tunable captures Semantics]:::outcome
    end


flowchart TD
    classDef era fill:#263238,stroke:#546E7A,color:#ffffff,stroke-width:2px;
    classDef method fill:#80DEEA,stroke:#00BCD4,color:#000000,stroke-width:1px;
    classDef author fill:#FFB74D,stroke:#FF9800,color:#000000;
    classDef outcome fill:#B0C4DE,stroke:#546E7A,color:#000000;
    classDef concept fill:#D3D3D3,stroke:#555,stroke-width:1px;

    subgraph 3_Contextual_Multimodal_Integration
        direction LR
        Title3[Contextual & Multimodal Integration]:::era
        
        subgraph Social_Context_Methods
            direction TB
            D[Method: Social Context Approaches]:::method
            D --> D1[Concept: Credibility and Engagement Features]:::concept
            D --> D2[Concept: Graph-based Methods]:::concept
            
            D1 --> D1a[Vosoughi et al #40;2018#41; - Spreading Speed]:::author
            D2 --> D2a[Graph Methods #40;Bian 2020 and Mehta 2022#41;]:::author
            D2a --> D2b[Goal: Early Warning / Scalability Challenge]:::outcome
        end
        
        subgraph Multimodal_Evolution
            direction TB
            E[Method: Cross-Media Approaches]:::method
            E --> E1[Concept: Text and Images Fusion]:::concept
            E1 --> E2[Concept: Cross-modal Semantic Alignment]:::concept
            E2 --> E3[Vision-Language Models: CLIP]:::concept
            
            E1 --> E1a[Text-Image Fusion #40;Jin 2017 and Wang 2018#41;]:::author
            E2 --> E2a[Semantic Alignment #40;Boididou 2018 and Qian 2021#41;]:::author
            E3 --> E3a[CLIP #40;Wang 2024#41;]:::author
            E3a --> E3b[Result: Detect Image-Text Inconsistencies]:::outcome
        end
        
        Title3 --> Social_Context_Methods
        Title3 --> Multimodal_Evolution
        
        Social_Context_Methods --> Link_Node[(Integration Point)]:::concept
        Multimodal_Evolution --> Link_Node
    end


flowchart TD
    classDef era fill:#263238,stroke:#546E7A,color:#ffffff,stroke-width:2px;
    classDef method fill:#80DEEA,stroke:#00BCD4,color:#000000,stroke-width:1px;
    classDef author fill:#FFB74D,stroke:#FF9800,color:#000000;
    classDef outcome fill:#B0C4DE,stroke:#546E7A,color:#000000;

    subgraph 4_LLMs_Reasoning_Frameworks
        direction TB
        Title4[Current & Future: LLMs and Reasoning Frameworks]:::era
        F[Method: Large Language Models - LLMs]:::method
        
        F --> F1[Concept: Zero-shot / Few-shot Detection]:::method
        F --> F2[Concept: Advanced Reasoning Frameworks]:::method
        
        F1 --> F1a[Rationales Hu et al #40;2024#41; and Man et al #40;2025#41;]:::author
        F2 --> F2a[Courtroom Debate Jin et al #40;2025#41;]:::author
        F2 --> F2b[Multimodal Reasoning Wang et al #40;2024#41;]:::author
        F2 --> F2c[Structured Modular Design Martirano etil #40;2025#41;]:::author
        F2c --> F2c1[Goal: Deep Fusion Cross-modal Relationships]:::outcome
    end