# 🦥 LazyLock AI
### Gemma 4 Multimodal Hardware Engineering Studio & Telegram Bot

> **"From Idea, Photo, Voice, or Datasheet to Breadboard in 15 Seconds."**
> *Built for rapid prototyping, AI productivity, and creative hardware engineering using open-source & cloud models.*

---

## 🚀 Overview
**LazyLock** is a multi-agent AI hardware engineering assistant that bridges the gap between raw hardware ideas and physical prototyping. Instead of spending hours reading pinout datasheets, calculating voltage thresholds, and writing boilerplate C++ code, makers can simply send a **Text prompt, Photo, Voice Note, or PDF Datasheet** via Telegram, and instantly receive a complete rapid-prototyping package.

Powered by **Google Gemma 4** via a flexible **Dual-Engine Architecture**, LazyLock lets you seamlessly switch between **Google AI Studio Cloud API** (maximum speed and power) and **Local Open-Source Ollama** (absolute data privacy and offline edge execution), coordinating **4 specialized AI agents** to handle architecture, safety audits, firmware generation, and hands-free voice coaching.

---

## ✨ Key Features & Hackathon Pillars

1. **🧠 100% Gemma 4 Multimodal Intelligence**
   - Leverages Google's Gemma 4 model family (`gemma-4-26b-a4b-it`, `gemma-4-31b-it`, `gemma-4-12b-it`) natively supporting system instructions, structured JSON, and multimodal inputs.
2. **🔀 Dual-Engine Open Source Switcher (Cloud vs. Local Edge)**
   - **Google Cloud API Mode (`/mode google`):** Uses Google AI Studio for fast cloud-based multimodal agent pipelines.
   - **Local Open-Source Mode (`/mode ollama`):** Connects directly to local Ollama instances running open-weight Gemma models (`http://localhost:11434`), proving complete offline execution and enterprise data privacy.
3. **🎙️ 4-Modality Universal Intake (Agent 0)**
   - **Text:** Describe any IoT or automation project.
   - **Photos 📷:** Snap a picture of loose components or a messy breadboard; Gemma 4 Vision identifies parts and auto-designs a build.
   - **Voice Notes 🎙️:** Speak your hardware idea hands-free while your hands are busy on the workbench.
   - **PDF Datasheets 📄:** Upload a sensor spec sheet, schematic, or project rubric; Agent 0 parses it instantly.
4. **🛡️ 4-Agent Autonomous Pipeline**
   - **Agent 0:** Multimodal Vision/Audio/PDF Intake & Specification Builder.
   - **Agent 1:** Schematic Architect (Generates Mermaid circuit diagrams & pinout tables).
   - **Agent 2:** Safety & BOM Auditor (Enforces strict JSON to check 3.3V/5V logic level compatibility and prevent "magic smoke").
   - **Agent 3:** Firmware & Voice Coach (Writes complete `.ino` C++ code + conversational wiring script).
5. **📲 Telegram-First Control & Live Web Dashboard Sync**
   - Use Telegram as your remote control while your laptop acts as a live Web Monitor streaming real-time agent logs, Mermaid graphs, active engine badges, and hardware vaults.

---

## 📐 System Architecture Flow

```text
[ Telegram: Text / Photo / Voice / PDF ] 
                  │
                  ▼
         [ Agent 0: Intake ] ──► (Gemma 4 Vision / Audio / PDF Parser)
                  │
                  ▼
     [ Dual-Engine LLM Router ] 
     ├──► Cloud API (Google AI Studio Gemma 4)
     └──► Local Edge (Open-Source Ollama)
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
   [ Agent 1 ] [ Agent 2 ] [ Agent 3 ]
   Schematic    Safety JSON   C++ .ino & Audio
       │          │          │
       └──────────┼──────────┘
                  │
                  ▼
   [ Output Delivery & Web Sync ] 
   ├──► Telegram (Summary + .ino File + Voice Note)
   └──► Web Dashboard (Live Terminal Logs & Visual Vault)