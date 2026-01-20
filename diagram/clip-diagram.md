---
config:
  layout: elk
  theme: neo-dark
---
flowchart TB
 subgraph 1_Initialization_and_Setup["1_Initialization_and_Setup"]
    direction TB
        A["Dataset Samples"]
        B{"Load CLIPEncoder"}
        C["Log Dataset Statistics"]
  end
 subgraph 2_Score_Calculation_Loop["2_Score_Calculation_Loop"]
    direction LR
        E["Outer Loop: Sim Type / Cosine-Euclidean"]
        D["Start Similarity Evaluation"]
        F["Inner Loop: Label / Fake-Real"]
        G1["Read / Load Cache"]
        G2["Select Samples & Batch"]
        H["Batch Processing: Size 128"]
        I1["Load / Preprocess Images"]
        I2["Compute Similarity / CLIP"]
        J["Similarity Scores Batch"]
        K["Update Cache & Local Score List"]
  end
 subgraph 3_Analysis_and_Reporting["3_Analysis_and_Reporting"]
    direction TB
        L["All Scores Collected"]
        M["Generate Box Plot / Matplotlib"]
        N["Log Image to W&B"]
        P["End: Finalize Tracker"]
  end
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F & L & M
    F --> G1 & G2 & E
    G1 --> H
    G2 --> H
    H --> I1
    I1 --> I2
    I2 --> J
    J --> K
    K --> F
    L --> M
    M --> N
    N --> P
    3_Analysis_and_Reporting --> P
     A:::data
     A:::stage
     B:::component
     B:::stage
     C:::component
     C:::stage
     E:::stage
     D:::stage
     F:::stage
     G1:::cache
     G2:::component
     H:::stage
     I1:::component
     I2:::component
     J:::data
     K:::component
     L:::data
     M:::component
     N:::external
     P:::stage
    classDef stage fill:#263238,stroke:#546E7A,color:#ffffff,stroke-width:2px
    classDef component fill:#4DD0E1,stroke:#00BCD4,color:#000000,stroke-width:1px
    classDef external fill:#FFB74D,stroke:#FF9800,color:#000000,stroke-width:1px
    classDef data fill:#FFFFFF,stroke:#B0BEC5,color:#000000,stroke-width:1px
    classDef cache fill:#FFE0B2,stroke:#FFCC80,color:#000000,stroke-width:1px
