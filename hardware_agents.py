import os
import json
import time
import socket
import logging
from typing import Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3.util.connection as urllib3_cn
from pymongo import MongoClient
from dotenv import load_dotenv

# =====================================================================
# 0. CONFIGURABLE LOGGING SYSTEM (AI OBSERVABILITY)
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SlothLabAI")

# Force IPv4 socket binding to prevent regional ISP timeouts
def _allowed_gai_family():
    return socket.AF_INET
urllib3_cn.allowed_gai_family = _allowed_gai_family

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
MONGODB_URI = os.getenv("MONGODB_ATLAS_URI", "").strip()

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Optional MongoDB Connection for Hardware Vault
db_client = None
projects_collection = None
if MONGODB_URI:
    try:
        db_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
        projects_collection = db_client["slothlab"]["hardware_vault"]
        logger.info("Connected to MongoDB Atlas successfully.")
    except Exception as e:
        logger.warning(f"MongoDB Atlas connection skipped: {e}")


def _create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=5, pool_maxsize=5)
    session.mount("https://", adapter)
    return session

HTTP_SESSION = _create_session()


def query_gemini(agent_tag: str, system_prompt: str, user_prompt: str, temp: float=0.5, force_json: bool=False) -> Optional[str]:
    logger.info(f"[{agent_tag}] Sending payload to Gemini ({GEMINI_MODEL}, temp={temp}, json={force_json})...")
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    
    config: Dict[str, Any] = {"temperature": temp, "topP": 0.95, "maxOutputTokens": 1500}
    if force_json:
        config["responseMimeType"] = "application/json"

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": config
    }

    try:
        res = HTTP_SESSION.post(GEMINI_API_URL, headers=headers, json=payload, timeout=(10, 40))
        res.raise_for_status()
        data = res.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "\n".join([p.get("text", "") for p in parts if "text" in p]).strip()
            logger.info(f"[{agent_tag}] Response received ({len(text)} chars).")
            return text
        return None
    except Exception as e:
        logger.error(f"[{agent_tag}] Gemini request failed: {e}")
        return None


def generate_elevenlabs_audio(text_script: str) -> Optional[bytes]:
    """
    Sends the wiring guide script to ElevenLabs to generate a natural voice audio file.
    """
    if not ELEVENLABS_API_KEY:
        logger.warning("ELEVENLABS_API_KEY not configured. Skipping voice generation.")
        return None

    logger.info("🎙️ Generating ElevenLabs hands-free audio briefing...")
    url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM" # Rachel default voice ID
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    payload = {
        "text": text_script[:800], # Keep concise for audio notes
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        if res.status_code == 200:
            logger.info("ElevenLabs audio generation successful.")
            return res.content
        logger.error(f"ElevenLabs error {res.status_code}: {res.text}")
        return None
    except Exception as e:
        logger.error(f"ElevenLabs connection error: {e}")
        return None


# =====================================================================
# 3-AGENT HARDWARE PIPELINE
# =====================================================================

def agent_circuit_architect(project_prompt: str) -> str:
    logger.info("🔌 AGENT 1: Schematic & Pinout Architect Activated")
    system_prompt = (
        "You are an expert embedded systems hardware architect. Given a project idea, "
        "generate a detailed pin-to-pin wiring guide and an ASCII / markdown block diagram. "
        "List exact components needed (microcontroller, sensors, actuators, resistors, power supplies)."
    )
    return query_gemini("Agent-1-Schematic", system_prompt, project_prompt, temp=0.5) or "Failed to generate schematic."


def agent_safety_bom_audit(project_prompt: str) -> Dict[str, Any]:
    logger.info("🛡️ AGENT 2: Magic-Smoke Safety & BOM Auditor Activated")
    system_prompt = (
        "You are a hardware safety auditor. Analyze the electronics project for voltage mismatches, "
        "current limits, and fire/short risks. Output strictly in JSON format with exact keys:\n"
        "{\n"
        '  "safety_verdict": "SAFE / CAUTION with a 1-sentence assessment",\n'
        '  "voltage_warnings": "Explicit checks for 3.3V vs 5V logic level conflicts",\n'
        '  "bill_of_materials": "Itemized list of hardware parts with estimated component costs",\n'
        '  "power_requirements": "Total current draw and recommended power supply method"\n'
        "}\nReturn ONLY valid raw JSON."
    )
    raw = query_gemini("Agent-2-Auditor", system_prompt, project_prompt, temp=0.2, force_json=True)
    if not raw:
        return {"error": "Safety audit failed."}
    
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.splitlines()[1:-1]
        cleaned = "\n".join(cleaned)
    try:
        return json.loads(cleaned)
    except:
        return {"safety_verdict": "CAUTION", "bill_of_materials": raw}


def agent_firmware_and_voice(project_prompt: str) -> Dict[str, Any]:
    logger.info("💻 AGENT 3: Zero-Effort Firmware & Voice Guide Activated")
    system_prompt = (
        "You are an embedded firmware engineer. Write complete, clean Arduino C++ code (.ino format) "
        "for the project. Also, write a short, friendly 45-second script explaining how to plug in the wires "
        "so it can be read aloud as an audio briefing.\n"
        "Format your output with clear markdown blocks: ```cpp for code, and a section titled '## Audio Guide Script'."
    )
    result = query_gemini("Agent-3-Firmware", system_prompt, project_prompt, temp=0.4) or ""
    
    # Extract audio script snippet for ElevenLabs
    audio_script = "Plug in your microcontroller, connect power rails, and upload your firmware."
    if "## Audio Guide Script" in result:
        parts = result.split("## Audio Guide Script")
        if len(parts) > 1:
            audio_script = parts[1].split("```")[0].strip()

    audio_bytes = generate_elevenlabs_audio(audio_script)

    return {
        "firmware_markdown": result,
        "audio_bytes": audio_bytes
    }


def run_hardware_pipeline(user_idea: str) -> Dict[str, Any]:
    logger.info(f"🚀 RUNNING SLOTHLAB HARDWARE PIPELINE FOR: '{user_idea}'")
    
    schematic = agent_circuit_architect(user_idea)
    time.sleep(1)
    audit = agent_safety_bom_audit(user_idea)
    time.sleep(1)
    firmware_data = agent_firmware_and_voice(user_idea)

    package = {
        "user_idea": user_idea,
        "schematic": schematic,
        "safety_audit": audit,
        "firmware": firmware_data["firmware_markdown"],
        "timestamp": time.time()
    }

    # Save to MongoDB Atlas Hardware Vault if configured
    if projects_collection is not None:
        try:
            projects_collection.insert_one(package)
            logger.info("Saved project package to MongoDB Atlas.")
        except Exception as e:
            logger.error(f"Failed to save to MongoDB: {e}")

    return {
        "schematic": schematic,
        "safety_audit": audit,
        "firmware": firmware_data["firmware_markdown"],
        "audio_bytes": firmware_data.get("audio_bytes")
    }