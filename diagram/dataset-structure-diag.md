---
config:
  theme: mc
  layout: dagre
---
flowchart TD
 subgraph MMFakeBench_Dataset_Structure["MMFakeBench_Dataset_Structure"]
    direction TB
        A["MMFakeBench Dataset"]
        B["root_dir: .../MMFakeBench_val"]
        D["source/MMFakeBench_val.json"]
        C1["real/"]
        C2["fake/"]
        E1["Entry Fields: text, image_path, label"]
  end
 subgraph GossipCop_Dataset_Structure["GossipCop_Dataset_Structure"]
    direction TB
        F["GossipCop Dataset"]
        G["root_dir: .../gossipcop"]
        H["real/"]
        I["fake/"]
        H1["real/img/"]
        H2["real/text/"]
        I1["fake/img/"]
        I2["fake/text/"]
        T1{"text"}
  end
    A --> B
    B --> D & C1 & C2
    D -- Reads entries --> E1
    E1 -- image_path points to --> C1 & C2
    F --> G
    G --> H & I
    H --> H1 & H2
    I --> I1 & I2
    H2 -- JSON files contain --> T1
    I2 -- JSON files contain --> T1
    H1 -- Image files --> G
    I1 -- Image files --> G
     A:::dataset
     B:::directory
     D:::file
     C1:::directory
     C2:::directory
     E1:::data
     F:::dataset
     G:::directory
     H:::directory
     I:::directory
     H1:::directory
     H2:::directory
     I1:::directory
     I2:::directory
     T1:::file
    classDef dataset fill:#263238,stroke:#546E7A,color:#ffffff,stroke-width:2px
    classDef directory fill:#D3D3D3,stroke:#555,stroke-width:1px
    classDef file fill:#FFFFFF,stroke:#546E7A,stroke-width:1px
    classDef link fill:#B0C4DE,stroke:#546E7A
