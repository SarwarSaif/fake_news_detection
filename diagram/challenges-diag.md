---
config:
  theme: mc
---
flowchart LR
 subgraph 1_Data_and_Benchmarking["1_Data_and_Benchmarking"]
    direction TB
        B1["Dataset and Evaluation Challenges"]
        C1["Deciding on Multi-Modal Benchmarking Datasets"]
        C2["Difficulty Handling Heterogeneous Data Sources"]
  end
 subgraph 2_Technical_and_Resource["2_Technical_and_Resource"]
    direction TB
        B2["Technical and Resource Constraints"]
        C3["Limited Access to High-Performance Training Resources"]
        C4["Image Size Reduction and Parallel Processing Issues"]
        C5["Managing Maximum Token Limits for Text"]
  end
 subgraph 3_Development_and_Operational["3_Development_and_Operational"]
    direction TB
        B3["Development and Operational"]
        C6["Steep Learning Curve for LLM Model Usage and Adaptation"]
  end
    B1 --> C1 & C2
    B2 --> C3 & C4 & C5
    B3 --> C6
    A["Core Research Challenges"] --> 1_Data_and_Benchmarking & 2_Technical_and_Resource & 3_Development_and_Operational
     A:::main
     B1:::category
     C1:::obstacle
     C2:::obstacle
     B2:::category
     C3:::obstacle
     C4:::obstacle
     C5:::obstacle
     B3:::category
     C6:::obstacle
    classDef main fill:#CC0000,stroke:#8B0000,color:#ffffff,stroke-width:2px
    classDef category fill:#FFD700,stroke:#B8860B,color:#000000
    classDef obstacle fill:#F08080,stroke:#CD5C5C,color:#000000,stroke-width:1px
