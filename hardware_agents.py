import os
import json
import time
import socket
import logging
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
LOG_BUFFER: deque = deque(maxlen=80)

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
# 1. ENVIRONMENT & MULTI-SPONSOR CONFIGURATION
# =====================================================================
load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
raw_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

VALID_MODELS = {
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemma-3-27b-it",
}
GEMINI_MODEL = raw_model if raw_model in VALID_MODELS else "gemini-3.8-flash"

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
MONGODB_URI = os.getenv("MONGODB_ATLAS_URI", "").strip()

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Store latest ElevenLabs audio in memory for the web audio player
LATEST_AUDIO_BYTES: Optional[bytes] = None

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
        total=4,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=5, pool_maxsize=5)
    session.mount("https://", adapter)
    return session

HTTP_SESSION = _create_session()


def validate_environment() -> bool:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "AIzaSyYourActualApiKeyHere":
        logger.error("GEMINI_API_KEY is missing or set to placeholder in .env.")
        return False
    return True


def query_gemini(
    agent_tag: str,
    system_prompt: str,
    user_prompt: str,
    temp: float = 0.5,
    force_json: bool = False,
    max_tokens: int = 1400
) -> Optional[str]:
    logger.info(f"[{agent_tag}] Dispatching prompt to {GEMINI_MODEL} (temp={temp}, json={force_json})...")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    config: Dict[str, Any] = {
        "temperature": temp,
        "topP": 0.95,
        "maxOutputTokens": max_tokens
    }
    if force_json and GEMINI_MODEL.startswith("gemini"):
        config["responseMimeType"] = "application/json"

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": config
    }

    try:
        res = HTTP_SESSION.post(GEMINI_API_URL, headers=headers, json=payload, timeout=(10, 45))
        res.raise_for_status()
        data = res.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "\n".join([p.get("text", "") for p in parts if "text" in p]).strip()
            logger.info(f"[{agent_tag}] Received {len(text)} chars from {GEMINI_MODEL}.")
            return text
        logger.warning(f"[{agent_tag}] Empty candidate response: {data}")
        return None
    except Exception as e:
        logger.error(f"[{agent_tag}] Gemini API request failed: {e}")
        return None


def parse_llm_json(agent_tag: str, raw_output: str) -> Dict[str, Any]:
    logger.info(f"[{agent_tag}] Parsing JSON safety & BOM payload...")
    cleaned = raw_output.strip()

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
            "bill_of_materials": raw_output,
            "power_requirements": "Use a current-limited USB or bench supply."
        }


def generate_elevenlabs_audio(text_script: str) -> Optional[bytes]:
    """
    Sends the hands-free wiring walkthrough to ElevenLabs TTS API.
    """
    global LATEST_AUDIO_BYTES
    if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "your_elevenlabs_api_key_here":
        logger.info("[ElevenLabs] No API key configured in .env; using browser Web Speech synthesis fallback.")
        LATEST_AUDIO_BYTES = None
        return None

    logger.info("🎙️ [ElevenLabs] Synthesizing hands-free voice wiring note...")
    url = "[https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM](https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM)"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    payload = {
        "text": text_script[:750],
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.55, "similarity_boost": 0.75}
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        if res.status_code == 200:
            LATEST_AUDIO_BYTES = res.content
            logger.info(f"🎙️ [ElevenLabs] Audio synthesized ({len(res.content)} bytes).")
            return res.content
        logger.warning(f"[ElevenLabs] HTTP {res.status_code}: {res.text[:120]}")
        return None
    except Exception as e:
        logger.error(f"[ElevenLabs] TTS request error: {e}")
        return None


# =====================================================================
# 2. THE 3 SPECIALIZED HARDWARE AGENTS
# =====================================================================

def agent_circuit_architect(project_prompt: str, board_type: str) -> str:
    logger.info("🔌 AGENT 1 ACTIVATED: Schematic & Pinout Architect")
    system_prompt = (
        "You are an expert Embedded Systems & Hardware Schematic Architect. "
        "Given the user's hardware idea and target microcontroller board, output:\n\n"
        "1. **📐 Mermaid Circuit Block Diagram:** Include a valid ```mermaid graph LR ... ``` code block showing how the microcontroller, sensors, actuators, and power rails connect.\n"
        "2. **📌 Exact Pin-to-Pin Wiring Table:** A clear Markdown list/table showing exact connections (e.g., `ESP32 GPIO 4 -> Relay IN`, `3.3V -> VCC`, `GND -> Common GND`) and any required pull-up/current-limiting resistors.\n"
        "3. **🦥 Lazy Breadboard Shortcut:** 2 quick tips to wire this in under 5 minutes without soldering."
    )
    user_msg = f"Target Board: {board_type}\nProject Idea: {project_prompt}\nGenerate the schematic and pinout guide:"
    return query_gemini("Agent-1-Schematic", system_prompt, user_msg, temp=0.5) or "Agent 1 failed to generate schematic."


def agent_safety_bom_audit(project_prompt: str, board_type: str) -> Dict[str, Any]:
    logger.info("🛡️ AGENT 2 ACTIVATED: Magic-Smoke Safety & BOM Auditor")
    system_prompt = (
        "You are a Senior Electrical Safety Auditor. Analyze the electronics project for voltage mismatches, "
        "overcurrent risks, flyback diode requirements, and component costs.\n"
        "Respond strictly in valid JSON with these exact keys (all values must be strings):\n"
        "{\n"
        '  "safety_verdict": "SAFE / CAUTION status with a 1-sentence summary",\n'
        '  "voltage_warnings": "Explicit checks for 3.3V vs 5V logic level conflicts, relay isolation, or pin current limits",\n'
        '  "bill_of_materials": "Itemized list of required hardware parts with estimated prices",\n'
        '  "power_requirements": "Estimated total current draw (mA) and recommended power supply",\n'
        '  "lazy_troubleshooting": "Top 3 things to check first if the circuit does not power on"\n'
        "}\nReturn ONLY the raw JSON object."
    )
    user_msg = f"Target Board: {board_type}\nProject Idea: {project_prompt}\nOutput the JSON safety & BOM audit:"
    raw = query_gemini("Agent-2-Auditor", system_prompt, user_msg, temp=0.25, force_json=True, max_tokens=900)
    if not raw:
        return {"error": "Agent 2 failed to get a response from Gemini."}
    return parse_llm_json("Agent-2-Auditor", raw)


def agent_firmware_and_voice(project_prompt: str, board_type: str) -> Dict[str, Any]:
    logger.info("💻 AGENT 3 ACTIVATED: Zero-Effort Firmware & Hands-Free Voice Coach")
    system_prompt = (
        "You are an Embedded Firmware Engineer. Generate:\n"
        "1. **Complete, Ready-to-Flash Firmware Code:** Enclosed in a ```cpp code block (Arduino .ino / C++) with pin definitions matching the project.\n"
        "2. **## Hands-Free Voice Wiring Script:** At the very end, under the exact heading `## Hands-Free Voice Wiring Script`, write a conversational 45-second voiceover walkthrough (3-4 sentences) that tells the maker step-by-step which jumper wires to plug in while their hands are holding the breadboard."
    )
    user_msg = f"Target Board: {board_type}\nProject Idea: {project_prompt}\nGenerate the firmware and voice script:"
    result = query_gemini("Agent-3-Firmware", system_prompt, user_msg, temp=0.45, max_tokens=1500) or ""

    # Extract voice script for ElevenLabs & Browser TTS
    voice_script = f"Here is your LazyLock wiring briefing for your {board_type}. Connect all ground pins to a common ground rail first, verify your 3.3 volt and 5 volt power lines before plugging in USB, and flash the generated sketch."
    if "## Hands-Free Voice Wiring Script" in result:
        parts = result.split("## Hands-Free Voice Wiring Script")
        if len(parts) > 1:
            extracted = parts[1].replace("#", "").replace("*", "").strip()
            if extracted:
                voice_script = extracted

    # Extract raw .ino code for 1-click file download
    raw_ino = ""
    if "```cpp" in result:
        raw_ino = result.split("```cpp")[1].split("```")[0].strip()
    elif "```c" in result:
        raw_ino = result.split("```c")[1].split("```")[0].strip()

    audio_bytes = generate_elevenlabs_audio(voice_script)

    return {
        "firmware_markdown": result,
        "raw_ino_code": raw_ino,
        "voice_script": voice_script,
        "audio_bytes": audio_bytes
    }


# =====================================================================
# 3. CENTRAL PIPELINE ORCHESTRATOR (SEQUENTIAL ANTI-429 MODE)
# =====================================================================
def run_hardware_pipeline(user_idea: str, board_type: str = "ESP32 DevKit V1") -> Dict[str, Any]:
    start_time = time.time()
    logger.info("=" * 64)
    logger.info(f"🦥 LAZYLOCK PIPELINE STARTED | Board: {board_type} | Idea: {user_idea}")
    logger.info("=" * 64)

    if not validate_environment():
        raise RuntimeError("GEMINI_API_KEY is missing or invalid in your .env file.")

    schematic = agent_circuit_architect(user_idea, board_type)
    time.sleep(1.2)

    audit = agent_safety_bom_audit(user_idea, board_type)
    time.sleep(1.2)

    firmware_data = agent_firmware_and_voice(user_idea, board_type)

    # Save to MongoDB Atlas Hardware Vault if configured
    if projects_collection is not None:
        try:
            projects_collection.insert_one({
                "user_idea": user_idea,
                "board_type": board_type,
                "schematic": schematic,
                "safety_audit": audit,
                "firmware": firmware_data["firmware_markdown"],
                "timestamp": time.time()
            })
            logger.info("🗄️ Project archived to MongoDB Atlas Hardware Vault.")
        except Exception as e:
            logger.warning(f"MongoDB Atlas insert skipped: {e}")

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"✨ LAZYLOCK PIPELINE COMPLETED IN {elapsed}s!")

    return {
        "board_type": board_type,
        "schematic": schematic,
        "safety_audit": audit,
        "firmware": firmware_data["firmware_markdown"],
        "raw_ino_code": firmware_data["raw_ino_code"],
        "voice_script": firmware_data["voice_script"],
        "audio_bytes": firmware_data.get("audio_bytes"),
        "logs": get_recent_logs()
    }