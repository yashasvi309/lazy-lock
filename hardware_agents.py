import os
import io
import re
import json
import time
import base64
import socket
import logging
from urllib.parse import quote
from collections import deque
from typing import Dict, Any, Optional, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3.util.connection as urllib3_cn
from pymongo import MongoClient
from dotenv import load_dotenv

# =====================================================================
# 0. CONFIGURABLE LOGGING SYSTEM (TERMINAL + WEB UI OBSERVABILITY)
# =====================================================================
LOG_BUFFER: deque = deque(maxlen=120)

class UIMemoryLogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        LOG_BUFFER.append(msg)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LazyLockAI")

ui_handler = UIMemoryLogHandler()
ui_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(ui_handler)

def get_recent_logs() -> List[str]:
    return list(LOG_BUFFER)

# Force IPv4 socket binding to prevent regional ISP timeouts
def _allowed_gai_family():
    return socket.AF_INET

urllib3_cn.allowed_gai_family = _allowed_gai_family

# =====================================================================
# 1. DUAL-ENGINE CONFIGURATION (GOOGLE CLOUD GEMMA 4 + LOCAL OLLAMA)
# =====================================================================
load_dotenv(override=True)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").strip().lower()  # "google" or "ollama"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

raw_model = os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it").strip()
VALID_GEMMA4_MODELS = {
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
    "gemma-4-12b-it",
    "gemma-4-e4b-it",
    "gemma-4-e2b-it",
}
GEMINI_MODEL = raw_model if raw_model in VALID_GEMMA4_MODELS else "gemma-4-26b-a4b-it"
FALLBACK_GEMMA4_MODELS = ["gemma-4-26b-a4b-it", "gemma-4-31b-it"]

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b").strip()

MONGODB_URI = os.getenv("MONGODB_ATLAS_URI", "").strip()

# Live Observability State for the Web Dashboard
LATEST_AUDIO_BYTES: Optional[bytes] = None
LATEST_PIPELINE_RESULT: Optional[Dict[str, Any]] = None
PIPELINE_STATUS: Dict[str, Any] = {
    "is_running": False,
    "current_stage": "Idle — Waiting for Telegram Input",
    "provider": LLM_PROVIDER,
    "active_model": OLLAMA_MODEL if LLM_PROVIDER == "ollama" else GEMINI_MODEL,
    "last_updated": time.time()
}

def set_pipeline_status(is_running: bool, stage: str):
    PIPELINE_STATUS["is_running"] = is_running
    PIPELINE_STATUS["current_stage"] = stage
    PIPELINE_STATUS["provider"] = LLM_PROVIDER
    PIPELINE_STATUS["active_model"] = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else GEMINI_MODEL
    PIPELINE_STATUS["last_updated"] = time.time()

def switch_engine_mode(new_provider: str, custom_model: Optional[str] = None) -> str:
    global LLM_PROVIDER, GEMINI_MODEL, OLLAMA_MODEL
    new_provider = new_provider.strip().lower()
    if new_provider in ("google", "cloud", "api"):
        LLM_PROVIDER = "google"
        if custom_model and custom_model in VALID_GEMMA4_MODELS:
            GEMINI_MODEL = custom_model
    elif new_provider in ("ollama", "local", "edge"):
        LLM_PROVIDER = "ollama"
        if custom_model:
            OLLAMA_MODEL = custom_model
    set_pipeline_status(False, f"Switched engine to {LLM_PROVIDER.upper()}")
    active = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else GEMINI_MODEL
    logger.info(f"🔄 Engine switched -> Provider: {LLM_PROVIDER.upper()} | Model: {active}")
    return f"Provider: {LLM_PROVIDER.upper()} | Active Model: {active}"

# Optional MongoDB Atlas Connection for Hardware Vault
db_client = None
projects_collection = None
if MONGODB_URI and "cluster.mongodb.net" not in MONGODB_URI:
    try:
        db_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
        projects_collection = db_client["lazylock"]["hardware_vault"]
        logger.info("Connected to MongoDB Atlas Hardware Vault successfully.")
    except Exception as e:
        logger.warning(f"MongoDB Atlas connection skipped: {e}")


def _create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=5, pool_maxsize=5)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

HTTP_SESSION = _create_session()


def validate_environment() -> bool:
    if LLM_PROVIDER == "ollama":
        return True
    if not GEMINI_API_KEY or GEMINI_API_KEY == "AIzaSyYourActualApiKeyHere":
        logger.error("GEMINI_API_KEY is missing or set to placeholder in .env.")
        return False
    return True


def _clean_gemma4_output(raw_text: str) -> str:
    """Strips internal Gemma 4 reasoning blocks (<|channel>thought...<channel|>) from output."""
    if not raw_text:
        return ""
    cleaned = re.sub(r"<\|channel>thought.*?<channel\|>", "", raw_text, flags=re.DOTALL)
    cleaned = re.sub(r"<thought>.*?</thought>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def extract_pdf_text_fallback(pdf_bytes: bytes) -> str:
    """Extracts text locally from PDF bytes (used for Ollama or hybrid PDF inspection)."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for idx, page in enumerate(reader.pages[:12]):
            txt = page.extract_text() or ""
            if txt.strip():
                pages_text.append(f"--- Page {idx + 1} ---\n{txt.strip()}")
        combined = "\n\n".join(pages_text)
        if combined:
            logger.info(f"📄 Extracted {len(combined)} chars from PDF ({len(reader.pages)} pages).")
            return combined[:8000]
    except Exception as e:
        logger.warning(f"Local pypdf extraction skipped/unavailable ({e}); relying on native multimodal PDF bytes.")
    return ""


# =====================================================================
# 2. SMART DUAL-ENGINE LLM ROUTER (GOOGLE GEMMA 4 API + LOCAL OLLAMA)
# =====================================================================
def _query_ollama(
    agent_tag: str,
    system_prompt: str,
    user_prompt: str,
    temp: float = 0.5,
    force_json: bool = False,
    media_bytes: Optional[bytes] = None,
    mime_type: Optional[str] = None
) -> Optional[str]:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    full_prompt = f"{user_prompt}"

    payload: Dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": temp
        }
    }
    if force_json:
        payload["format"] = "json"

    # Attach image bytes for local Ollama vision models (e.g., gemma4 / llava)
    if media_bytes and mime_type and mime_type.startswith("image/"):
        payload["images"] = [base64.b64encode(media_bytes).decode("utf-8")]

    logger.info(f"[{agent_tag}] 🖥️ Dispatching to LOCAL OLLAMA ({OLLAMA_MODEL}) at {OLLAMA_BASE_URL}...")
    try:
        res = HTTP_SESSION.post(url, json=payload, timeout=(10, 90))
        res.raise_for_status()
        data = res.json()
        text = _clean_gemma4_output(data.get("response", ""))
        if text:
            logger.info(f"[{agent_tag}] 🖥️ Local Ollama ({OLLAMA_MODEL}) returned {len(text)} chars.")
            return text
    except Exception as e:
        logger.warning(f"[{agent_tag}] Local Ollama failed ({e}). Checking cloud Gemma 4 fallback...")
    return None


def _query_google_gemma4(
    agent_tag: str,
    system_prompt: str,
    user_prompt: str,
    temp: float = 0.5,
    force_json: bool = False,
    max_tokens: int = 1500,
    media_bytes: Optional[bytes] = None,
    mime_type: Optional[str] = None
) -> Optional[str]:
    models_to_try = [GEMINI_MODEL] + [m for m in FALLBACK_GEMMA4_MODELS if m != GEMINI_MODEL]
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    config: Dict[str, Any] = {
        "temperature": temp,
        "topP": 0.95,
        "maxOutputTokens": max_tokens,
    }
    if force_json:
        config["responseMimeType"] = "application/json"

    user_parts: List[Dict[str, Any]] = []
    if media_bytes and mime_type:
        b64_data = base64.b64encode(media_bytes).decode("utf-8")
        user_parts.append({
            "inline_data": {
                "mime_type": mime_type,
                "data": b64_data
            }
        })
    user_parts.append({"text": user_prompt})

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": user_parts}],
        "generationConfig": config
    }

    for model_name in models_to_try:
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        logger.info(f"[{agent_tag}] ☁️ Dispatching to Google Cloud Gemma 4 ({model_name}) | media={mime_type or 'text'}...")
        try:
            res = HTTP_SESSION.post(api_url, headers=headers, json=payload, timeout=(10, 60))
            if res.status_code == 400 and "responseMimeType" in str(res.text):
                payload["generationConfig"].pop("responseMimeType", None)
                res = HTTP_SESSION.post(api_url, headers=headers, json=payload, timeout=(10, 60))

            res.raise_for_status()
            data = res.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "\n".join([p.get("text", "") for p in parts if "text" in p and not p.get("thought", False)]).strip()
                text = _clean_gemma4_output(text)
                if text:
                    logger.info(f"[{agent_tag}] ☁️ Gemma 4 ({model_name}) returned {len(text)} chars.")
                    return text
        except Exception as e:
            logger.warning(f"[{agent_tag}] Cloud attempt on {model_name} failed ({e}).")
    return None


def query_llm(
    agent_tag: str,
    system_prompt: str,
    user_prompt: str,
    temp: float = 0.5,
    force_json: bool = False,
    max_tokens: int = 1500,
    media_bytes: Optional[bytes] = None,
    mime_type: Optional[str] = None
) -> Optional[str]:
    """
    Unified Dual-Engine Dispatcher:
    Routes to Local Ollama or Google Cloud Gemma 4 based on LLM_PROVIDER,
    with automatic failover if one engine is unreachable during live demo.
    """
    if LLM_PROVIDER == "ollama":
        # Note: If audio or raw PDF without local text is sent in Ollama mode,
        # try Ollama first, or fall back to Google Cloud Gemma 4 if GEMINI_API_KEY is available
        res = _query_ollama(agent_tag, system_prompt, user_prompt, temp, force_json, media_bytes, mime_type)
        if res:
            return res
        if GEMINI_API_KEY:
            logger.info(f"[{agent_tag}] Auto-failing over from Ollama to Cloud Gemma 4...")
            return _query_google_gemma4(agent_tag, system_prompt, user_prompt, temp, force_json, max_tokens, media_bytes, mime_type)
        return None
    else:
        res = _query_google_gemma4(agent_tag, system_prompt, user_prompt, temp, force_json, max_tokens, media_bytes, mime_type)
        if res:
            return res
        logger.info(f"[{agent_tag}] Checking Local Ollama fallback...")
        return _query_ollama(agent_tag, system_prompt, user_prompt, temp, force_json, media_bytes, mime_type)


def parse_llm_json(agent_tag: str, raw_output: str) -> Dict[str, Any]:
    logger.info(f"[{agent_tag}] Parsing JSON safety & BOM payload...")
    cleaned = _clean_gemma4_output(raw_output)

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx : end_idx + 1]

    try:
        parsed = json.loads(cleaned)
        logger.info(f"[{agent_tag}] JSON verified! Keys: {list(parsed.keys())}")
        return parsed
    except Exception as e:
        logger.error(f"[{agent_tag}] JSON parse fallback triggered: {e}")
        return {
            "safety_verdict": "CAUTION — Manual Review Recommended",
            "voltage_warnings": "Verify 3.3V and 5V logic levels before powering rails.",
            "bill_of_materials": raw_output[:600] if raw_output else "Standard microcontroller + jumper wires",
            "power_requirements": "Use a current-limited 5V USB or bench supply.",
            "lazy_troubleshooting": "1. Verify common GND. 2. Check 3.3V vs 5V pins. 3. Reseat breadboard jumpers."
        }


def generate_voice_audio(text_script: str) -> Optional[bytes]:
    """
    Zero-key built-in TTS MP3 generator so Telegram & the Web UI always receive
    a playable hands-free voice note without needing ElevenLabs keys.
    """
    global LATEST_AUDIO_BYTES
    clean_script = re.sub(r"[#*`_~]", "", text_script).strip()
    if not clean_script:
        return None

    try:
        logger.info("🎙️ [Built-in TTS] Synthesizing hands-free MP3 voice note...")
        short_text = clean_script[:195]
        tts_url = f"[https://translate.google.com/translate_tts?ie=UTF-8&q=](https://translate.google.com/translate_tts?ie=UTF-8&q=){quote(short_text)}&tl=en&client=tw-ob"
        res = HTTP_SESSION.get(tts_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if res.status_code == 200 and len(res.content) > 200:
            LATEST_AUDIO_BYTES = res.content
            logger.info(f"🎙️ [Built-in TTS] Audio synthesized ({len(res.content)} bytes).")
            return res.content
    except Exception as e:
        logger.warning(f"[Built-in TTS] Audio synthesis skipped: {e}")

    LATEST_AUDIO_BYTES = None
    return None


# =====================================================================
# 3. THE 4 GEMMA 4 SPECIALIZED HARDWARE AGENTS (TEXT + PHOTO + VOICE + PDF)
# =====================================================================

def agent_gemma4_multimodal_intake(
    user_text: Optional[str] = None,
    media_bytes: Optional[bytes] = None,
    mime_type: Optional[str] = None,
    board_type: str = "ESP32 DevKit V1"
) -> Dict[str, str]:
    """
    AGENT 0 (Multimodal Intake & Creative Prototyper):
    Processes Photos (image/*), Voice Notes (audio/*), PDF Datasheets/Specs (application/pdf),
    or Text into a unified hardware engineering prompt.
    """
    if not media_bytes or not mime_type:
        return {
            "synthesized_prompt": user_text or "Design an interactive smart sensor and actuator rapid prototype.",
            "media_summary": "Direct Text Specification",
            "detected_board": board_type
        }

    # --- MODALITY 1: PHOTO / VISION ---
    if mime_type.startswith("image/"):
        set_pipeline_status(True, "Agent 0: Gemma 4 Vision Inspecting Hardware Photo...")
        logger.info("📷 AGENT 0 ACTIVATED: Gemma 4 Multimodal Vision Hardware Inspector")
        system_prompt = (
            "You are Agent 0: The Gemma 4 Multimodal Hardware Vision Inspector & Creative Prototyper. "
            "Inspect the uploaded photo (loose components, breadboard, microcontroller, or circuit sketch). "
            "Output:\n"
            "1. **🔍 Detected Hardware & Pins:** Every microcontroller, sensor, module, or actuator visible.\n"
            "2. **⚡ Wiring / Breadboard Audit:** Spot any loose jumpers, missing ground rails, or voltage risks.\n"
            "3. **🚀 Rapid Prototype Specification:** Formulate a clear hardware build specification using these exact parts."
        )
        goal_str = f"User Caption/Goal: {user_text}" if user_text else "Identify the parts in the image and propose a creative rapid prototype!"
        user_msg = f"Target Board: {board_type}\n{goal_str}"
        vision_report = query_llm(
            "Agent-0-Vision",
            system_prompt,
            user_msg,
            temp=0.35,
            max_tokens=700,
            media_bytes=media_bytes,
            mime_type=mime_type
        ) or (user_text or "Hardware circuit from uploaded photo.")

        return {
            "synthesized_prompt": f"User Goal: {user_text or 'Prototype from photo'}\n\n[Gemma 4 Vision Report]:\n{vision_report}",
            "media_summary": vision_report,
            "detected_board": board_type
        }

    # --- MODALITY 2: VOICE / AUDIO ---
    if mime_type.startswith("audio/"):
        set_pipeline_status(True, "Agent 0: Gemma 4 Decoding Voice Command...")
        logger.info("🎙️ AGENT 0 ACTIVATED: Gemma 4 Multimodal Voice Command Decoder")
        system_prompt = (
            "You are Agent 0: The Gemma 4 Voice-to-Hardware Rapid Prototyping Assistant. "
            "Listen to the maker's spoken audio command and output:\n"
            "1. **🗣️ Voice Transcript:** Exact transcription of what the user asked to build.\n"
            "2. **🚀 Hardware Build Specification:** Clear technical specification listing the board, sensors, actuators, and logic."
        )
        user_msg = f"Target Board: {board_type}\nAdditional Note: {user_text or 'Telegram Voice Note'}\nTranscribe and extract the hardware specification:"
        audio_report = query_llm(
            "Agent-0-Audio",
            system_prompt,
            user_msg,
            temp=0.25,
            max_tokens=500,
            media_bytes=media_bytes,
            mime_type=mime_type
        ) or f"**🗣️ Voice Command Captured:** {user_text or f'Build a smart automated sensor & relay system on {board_type}.'}"

        return {
            "synthesized_prompt": f"{user_text or ''}\n\n[Gemma 4 Voice Command Analysis]:\n{audio_report}".strip(),
            "media_summary": audio_report,
            "detected_board": board_type
        }

    # --- MODALITY 3: PDF DATASHEET / RUBRIC / SCHEMATIC ---
    if mime_type == "application/pdf":
        set_pipeline_status(True, "Agent 0: Gemma 4 Parsing PDF Datasheet / Document...")
        logger.info("📄 AGENT 0 ACTIVATED: Gemma 4 PDF Datasheet & Specification Extractor")
        extracted_text = extract_pdf_text_fallback(media_bytes)

        system_prompt = (
            "You are Agent 0: The Gemma 4 Hardware Datasheet & PDF Specification Analyst. "
            "Analyze the uploaded PDF document (which may be a sensor datasheet, pinout guide, problem statement, or schematic PDF). "
            "Output:\n"
            "1. **📄 PDF Document Summary:** Key components, operating voltages (3.3V/5V), and communication protocols (I2C/SPI/UART/GPIO) found in the PDF.\n"
            "2. **📌 Critical Pinout / Ratings:** Important pins or electrical limits mentioned.\n"
            "3. **🚀 Rapid Prototype Specification:** A complete hardware build specification ready for schematic & firmware generation."
        )
        pdf_context = f"\n\n[Extracted PDF Text]:\n{extracted_text}" if extracted_text else ""
        user_msg = (
            f"Target Board: {board_type}\n"
            f"User Goal/Caption: {user_text or 'Extract the hardware circuit & firmware specification from this PDF.'}"
            f"{pdf_context}"
        )
        # If local text was extracted and we're in Ollama mode, we don't need to send raw PDF inline_data
        send_bytes = media_bytes if LLM_PROVIDER == "google" else None
        send_mime = "application/pdf" if LLM_PROVIDER == "google" else None

        pdf_report = query_llm(
            "Agent-0-PDF",
            system_prompt,
            user_msg,
            temp=0.3,
            max_tokens=750,
            media_bytes=send_bytes,
            mime_type=send_mime
        ) or (extracted_text[:600] if extracted_text else "PDF hardware specification received.")

        return {
            "synthesized_prompt": f"User Goal: {user_text or 'Hardware build from PDF'}\n\n[Gemma 4 PDF Datasheet Analysis]:\n{pdf_report}",
            "media_summary": pdf_report,
            "detected_board": board_type
        }

    return {
        "synthesized_prompt": user_text or "Rapid hardware prototype",
        "media_summary": f"Processed attachment ({mime_type})",
        "detected_board": board_type
    }


def agent_circuit_architect(project_prompt: str, board_type: str) -> str:
    set_pipeline_status(True, "Agent 1: Designing Mermaid Schematic & Pinout Table...")
    logger.info("🔌 AGENT 1 ACTIVATED: Gemma 4 Schematic & Pinout Architect")
    system_prompt = (
        "You are Agent 1: An expert Embedded Systems & Rapid Prototyping Schematic Architect powered by Gemma 4. "
        "Given the hardware specification and target microcontroller board, output:\n\n"
        "1. **📐 Mermaid Circuit Block Diagram:** Include a valid ```mermaid graph LR ... ``` code block showing how the microcontroller, sensors, actuators, and power rails connect.\n"
        "2. **📌 Exact Pin-to-Pin Wiring Table:** A clear Markdown list/table showing exact connections (e.g., `ESP32 GPIO 4 -> Relay IN`, `3.3V -> VCC`, `GND -> Common GND`) and any required pull-up/current-limiting resistors.\n"
        "3. **🦥 5-Minute Rapid Breadboard Shortcut:** 2 creative, practical hacks to prototype this in under 5 minutes without soldering."
    )
    user_msg = f"Target Board: {board_type}\nProject Specification:\n{project_prompt}\nGenerate the schematic and pinout guide:"
    return query_llm("Agent-1-Schematic", system_prompt, user_msg, temp=0.45) or "Agent 1 failed to generate schematic."


def agent_safety_bom_audit(project_prompt: str, board_type: str) -> Dict[str, Any]:
    set_pipeline_status(True, "Agent 2: Running Magic-Smoke Voltage & BOM Audit (JSON)...")
    logger.info("🛡️ AGENT 2 ACTIVATED: Gemma 4 Magic-Smoke Safety & BOM Auditor")
    system_prompt = (
        "You are Agent 2: A Senior Electrical Safety & Rapid Prototyping Auditor powered by Gemma 4. "
        "Analyze the electronics project for voltage mismatches, overcurrent risks, flyback diode requirements, and component costs.\n"
        "Respond strictly in valid JSON with these exact keys (all values must be strings):\n"
        "{\n"
        '  "safety_verdict": "SAFE / CAUTION status with a 1-sentence summary",\n'
        '  "voltage_warnings": "Explicit checks for 3.3V vs 5V logic level conflicts, relay isolation, or pin current limits",\n'
        '  "bill_of_materials": "Itemized list of required hardware parts with estimated prices",\n'
        '  "power_requirements": "Estimated total current draw (mA) and recommended power supply",\n'
        '  "lazy_troubleshooting": "Top 3 things to check first if the circuit does not power on"\n'
        "}\nReturn ONLY the raw JSON object."
    )
    user_msg = f"Target Board: {board_type}\nProject Specification:\n{project_prompt}\nOutput the JSON safety & BOM audit:"
    raw = query_llm("Agent-2-Auditor", system_prompt, user_msg, temp=0.2, force_json=True, max_tokens=900)
    if not raw:
        return {"error": "Agent 2 failed to get a response from Gemma 4."}
    return parse_llm_json("Agent-2-Auditor", raw)


def agent_firmware_and_voice(project_prompt: str, board_type: str) -> Dict[str, Any]:
    set_pipeline_status(True, "Agent 3: Writing C++ (.ino) Firmware & Voice Walkthrough...")
    logger.info("💻 AGENT 3 ACTIVATED: Gemma 4 Zero-Effort Firmware & Hands-Free Voice Coach")
    system_prompt = (
        "You are Agent 3: An Embedded Firmware Engineer & Voice Coach powered by Gemma 4. Generate:\n"
        "1. **Complete, Ready-to-Flash Firmware Code:** Enclosed in a ```cpp code block (Arduino .ino / C++) with exact pin definitions matching the project and Serial telemetry for rapid debugging.\n"
        "2. **## Hands-Free Voice Wiring Script:** At the very end, under the exact heading `## Hands-Free Voice Wiring Script`, write a punchy, conversational 35-second voiceover walkthrough (3-4 sentences) telling the maker step-by-step which jumper wires to plug in while their hands are holding the breadboard."
    )
    user_msg = f"Target Board: {board_type}\nProject Specification:\n{project_prompt}\nGenerate the firmware and voice script:"
    result = query_llm("Agent-3-Firmware", system_prompt, user_msg, temp=0.4, max_tokens=1600) or ""

    voice_script = (
        f"Here is your LazyLock Gemma 4 wiring briefing for your {board_type}. "
        f"Connect all ground pins to a common ground rail first, verify your 3.3 volt and 5 volt power lines before plugging in USB, and flash the generated sketch."
    )
    if "## Hands-Free Voice Wiring Script" in result:
        parts = result.split("## Hands-Free Voice Wiring Script")
        if len(parts) > 1:
            extracted = parts[1].replace("#", "").replace("*", "").strip()
            if extracted:
                voice_script = extracted

    raw_ino = ""
    if "```cpp" in result:
        raw_ino = result.split("```cpp")[1].split("```")[0].strip()
    elif "```c" in result:
        raw_ino = result.split("```c")[1].split("```")[0].strip()

    audio_bytes = generate_voice_audio(voice_script)

    return {
        "firmware_markdown": result,
        "raw_ino_code": raw_ino,
        "voice_script": voice_script,
        "audio_bytes": audio_bytes
    }


# =====================================================================
# 4. CENTRAL MULTIMODAL PIPELINE ORCHESTRATOR
# =====================================================================
def run_hardware_pipeline(
    user_idea: str = "",
    board_type: str = "ESP32 DevKit V1",
    media_bytes: Optional[bytes] = None,
    mime_type: Optional[str] = None,
    source: str = "telegram-text"
) -> Dict[str, Any]:
    global LATEST_PIPELINE_RESULT
    start_time = time.time()
    modality = mime_type if mime_type else "text/plain"
    active_model_name = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else GEMINI_MODEL

    logger.info("=" * 70)
    logger.info(
        f"🦥 LAZYLOCK PIPELINE STARTED | Engine: {LLM_PROVIDER.upper()} ({active_model_name}) | "
        f"Source: {source.upper()} | Modality: {modality}"
    )
    logger.info("=" * 70)

    if not validate_environment():
        set_pipeline_status(False, "Error: GEMINI_API_KEY missing in .env")
        raise RuntimeError("GEMINI_API_KEY is missing in your .env file.")

    try:
        # Stage 0: Multimodal Intake (Photo / Audio / PDF / Text)
        intake = agent_gemma4_multimodal_intake(
            user_text=user_idea,
            media_bytes=media_bytes,
            mime_type=mime_type,
            board_type=board_type
        )
        effective_prompt = intake["synthesized_prompt"]
        media_summary = intake["media_summary"]

        if media_bytes and LLM_PROVIDER == "google":
            time.sleep(0.6)

        # Stage 1: Circuit Architect
        schematic = agent_circuit_architect(effective_prompt, board_type)
        if LLM_PROVIDER == "google":
            time.sleep(0.8)

        # Stage 2: Safety & BOM Auditor
        audit = agent_safety_bom_audit(effective_prompt, board_type)
        if LLM_PROVIDER == "google":
            time.sleep(0.8)

        # Stage 3: Firmware & Voice Coach
        firmware_data = agent_firmware_and_voice(effective_prompt, board_type)

        elapsed = round(time.time() - start_time, 2)
        set_pipeline_status(False, f"Completed in {elapsed}s via {LLM_PROVIDER.upper()} ({active_model_name})")
        logger.info(f"✨ LAZYLOCK PIPELINE COMPLETED IN {elapsed}s!")

        default_title = "Text Hardware Build"
        if mime_type:
            if "image" in mime_type:
                default_title = "📷 Photo Hardware Inspection & Build"
            elif "audio" in mime_type:
                default_title = "🎙️ Voice Note Hardware Build"
            elif "pdf" in mime_type:
                default_title = "📄 PDF Datasheet Hardware Build"

        result_payload = {
            "user_idea": user_idea or default_title,
            "source": source,
            "modality": modality,
            "provider": LLM_PROVIDER,
            "model_used": active_model_name,
            "board_type": board_type,
            "media_summary": media_summary,
            "schematic": schematic,
            "safety_audit": audit,
            "firmware": firmware_data["firmware_markdown"],
            "raw_ino_code": firmware_data["raw_ino_code"],
            "voice_script": firmware_data["voice_script"],
            "audio_bytes": firmware_data.get("audio_bytes"),
            "has_audio": firmware_data.get("audio_bytes") is not None,
            "elapsed_seconds": elapsed,
            "timestamp": time.time(),
            "logs": get_recent_logs()
        }

        LATEST_PIPELINE_RESULT = result_payload

        if projects_collection is not None:
            try:
                projects_collection.insert_one({
                    "user_idea": result_payload["user_idea"],
                    "source": source,
                    "modality": modality,
                    "provider": LLM_PROVIDER,
                    "model_used": active_model_name,
                    "board_type": board_type,
                    "media_summary": media_summary,
                    "schematic": schematic,
                    "safety_audit": audit,
                    "firmware": firmware_data["firmware_markdown"],
                    "timestamp": result_payload["timestamp"]
                })
                logger.info("🗄️ Project archived to MongoDB Atlas Hardware Vault.")
            except Exception as e:
                logger.warning(f"MongoDB Atlas insert skipped: {e}")

        return result_payload

    except Exception as e:
        set_pipeline_status(False, f"Pipeline Error: {str(e)[:80]}")
        raise

    