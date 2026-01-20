---
config:
  layout: elk
  theme: neo-dark
---
flowchart TB
 subgraph 1_System_Initialization_Data_Acquisition["1_System_Initialization_Data_Acquisition"]
        A["Dataset Samples"]
        B{"Load Gemma-3N Model"}
  end
 subgraph 2_Core_Fact_Checking_Process["2_Core_Fact_Checking_Process"]
        D1["Image Preprocessing"]
        C["Loop: Process One Sample"]
        D2{"System Prompt: Forensic Analyst Persona"}
        E["Input Builder"]
        F["Gemma-3N LLM"]
        G["Raw Output: JSON/Text"]
  end
 subgraph 3_Result_Interpretation_Tracking["3_Result_Interpretation_Tracking"]
        I["Structured Result"]
        H["Output Parser"]
        J1["Save Local Files &ﬂ°°40¶ßRaw & Parsed&ﬂ°°41¶ß"]
        J2["Log to W&B Dashboard"]
  end
    A --> B
    B --> C
    C --> D1 & D2 & K["End: Aggregate Final Summary Report"]
    D1 --> E
    D2 --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J1 & J2
    J1 --> C
    J2 --> C
     A:::data
     A:::stage
     B:::component
     B:::stage
     D1:::component
     C:::stage
     C:::stage
     D2:::component
     E:::component
     F:::component
     G:::data
     I:::data
     H:::component
     J1:::component
     J2:::external
     K:::stage
    classDef stage fill:#263238,stroke:#546E7A,color:#ffffff,stroke-width:2px
    classDef component fill:#4DD0E1,stroke:#00BCD4,color:#000000,stroke-width:1px
    classDef external fill:#FFB74D,stroke:#FF9800,color:#000000,stroke-width:1px
    classDef data fill:#FFFFFF,stroke:#B0BEC5,color:#000000,stroke-width:1px
