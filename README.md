# 🏟️ Intelligent Stadium Microclimate & Canopy Control System

An IoT and AI-driven smart stadium prototype combining **ESP32 local sensors**, **Open-Meteo cloud auxiliary data**, a **Random Forest precipitation classifier**, and a **Mamdani Fuzzy Inference System (FIS)** for autonomous retractable canopy control.

## 🧠 System Architecture & 5-Level Priority Flowchart

```mermaid
flowchart TD
    Start(["⏱️ <b>SYSTEM LOOP START</b><br/>(Real-Time Control Cycle)"]) --> ModeSelect

    ModeSelect{"<b>Input Mode Selection</b><br/>(Plan A vs. Plan B Fallback)"}
    ModeSelect -- "Plan A (Connected)" --> Local & Cloud
    ModeSelect -- "Plan B (No Hardware)" --> Sim["🎛️ <b>Dashboard Simulator / CSV Replay</b><br/>Manual Sliders or Historical Dataset"]
    Sim --> Local & Cloud

    subgraph SensingLayer ["📡 <b>1. Hybrid Edge-Cloud Sensing Layer</b>"]
        direction LR
        Local["<b>Local ESP32 Sensors</b><br/>• DHT11: Temp (°C) & RH (%)<br/>• Photosensitive LDR: Light (0-100%)<br/>• Rain Sensor: Active Rain (0/1)<br/>• PIR Sensor: Occupancy (0/1)"]
        Cloud["<b>Open-Meteo Auxiliary Data</b><br/>• Wind Speed (km/h)<br/>• Cloud Cover (%)<br/>• UV Index (0-12)<br/>• Temporal & 1h Lag Deltas"]
    end

    Local & Cloud --> RF["🌲 <b>2. Stage-1 AI: Random Forest Classifier</b><br/>Fuses Local Sensors + Cloud Aux + 1h Trends<br/><b>Output: Next-1h Rain Probability (rf_prob %)</b>"]

    RF --> P1{"<b>[Priority 1: Hardware Rain Interrupt]</b><br/>Is Rain Sensor == 1?<br/>(Active Rainfall Detected)"}

    P1 -- "YES" --> Act1["🚨 <b>Emergency Rain Override</b><br/>Protect Stadium Pitch Immediately<br/><b>Canopy Closure = 100%</b><br/>(Servo 1: 180°, Servo 2: 0°)"]
    P1 -- "NO" --> P2{"<b>[Priority 2: Proactive AI Rain Shield]</b><br/>Is rf_prob >= 65%?<br/>(Heavy Rain Imminent in 1h)"}

    P2 -- "YES" --> Act2["🌧️ <b>Proactive Pre-Closure Mode</b><br/>Close Roof Before Downpour Starts<br/><b>Canopy Closure = 90%</b><br/>(Servo 1: 162°, Servo 2: 18°)"]
    P2 -- "NO" --> P3{"<b>[Priority 3: Wind Structural Safety]</b><br/>Is Wind Speed > 18 km/h?<br/>(Strong Wind Warning)"}

    P3 -- "YES" --> Act3["💨 <b>Wind Safety Retraction Mode</b><br/>Retract Canopy to Prevent Wind Damage<br/><b>Canopy Closure = 10%</b><br/>(Servo 1: 18°, Servo 2: 162°)"]
    P3 -- "NO" --> P4{"<b>[Priority 4: PIR Occupancy Check]</b><br/>Is PIR Sensor == 0?<br/>(Stadium Empty / Unoccupied)"}

    P4 -- "YES (Empty)" --> Act4["🌱 <b>Eco Standby & Turf Sunlight Mode</b><br/>Open Roof for Grass Photosynthesis & Save Motor Power<br/><b>Canopy Closure = 0%</b><br/>(Servo 1: 0°, Servo 2: 180°)"]
    P4 -- "NO (Occupied)" --> Act5["🧠 <b>[Priority 5: Mamdani Fuzzy Inference System]</b><br/>Spectators Present & Safe Weather ➔ Optimize Thermal Comfort<br/><b>Inputs:</b> Temp + Humidity + Light/UV + rf_prob<br/><b>Defuzzified Output:</b> Canopy Closure = X% (0% - 100%)"]

    Act1 & Act2 & Act3 & Act4 & Act5 --> OutputLayer

    subgraph OutputLayer ["⚙️ <b>3. Dual-Track Execution & Visualization Layer</b>"]
        direction LR
        Dash["🖥️ <b>2D/3D Digital Twin Dashboard</b><br/>Displays AI Decision Mode, Sensor Metrics,<br/>& Animated Retractable Canopy"]
        HW["🤖 <b>Physical Actuation (When Connected)</b><br/>ESP32 I2C (SDA=21, SCL=22) ➔ PCA9685<br/>Drives <b>Servo 1 (Angle°) & Servo 2 (180-Angle°)</b>"]
    end

    OutputLayer --> EndLoop(["🔄 <b>WAIT FOR NEXT CYCLE</b>"])

    style Start fill:#2d3436,stroke:#000,color:#fff
    style EndLoop fill:#2d3436,stroke:#000,color:#fff
    style ModeSelect fill:#00cec9,stroke:#000,color:#000
    style RF fill:#0984e3,stroke:#000,color:#fff
    style Act1 fill:#d63031,stroke:#000,color:#fff
    style Act2 fill:#e17055,stroke:#000,color:#fff
    style Act3 fill:#fdcb6e,stroke:#000,color:#000
    style Act4 fill:#00b894,stroke:#000,color:#fff
    style Act5 fill:#6c5ce7,stroke:#000,color:#fff
```
