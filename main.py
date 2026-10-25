import os
import io
import threading
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn

import hardware_agents as agents

app = FastAPI(title="LazyLock | Gemma 4 Multimodal Telegram Controller & Live Web Monitor")
templates = Jinja2Templates(directory="templates")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


class HardwareRequest(BaseModel):
    idea: str
    board_type: str = "ESP32 DevKit V1"


class ModeSwitchRequest(BaseModel):
    provider: str  # "google" or "ollama"
    model: str = ""


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/status")
def get_system_status():
    """Returns real-time engine mode (Google vs Ollama), active agent stage, and logs."""
    return {
        "status": agents.PIPELINE_STATUS,
        "has_result": agents.LATEST_PIPELINE_RESULT is not None,
        "latest_timestamp": agents.LATEST_PIPELINE_RESULT["timestamp"] if agents.LATEST_PIPELINE_RESULT else None,
        "logs": agents.get_recent_logs()
    }


@app.get("/api/latest-result")
def get_latest_result():
    """
    Polled by the Web Dashboard to automatically render every Telegram build
    (Text, Photo, Voice, or PDF) on the big screen in real time!
    """
    if not agents.LATEST_PIPELINE_RESULT:
        return {"has_result": False, "status": agents.PIPELINE_STATUS}

    res = agents.LATEST_PIPELINE_RESULT
    return {
        "has_result": True,
        "status": agents.PIPELINE_STATUS,
        "provider": res["provider"],
        "model": res["model_used"],
        "timestamp": res["timestamp"],
        "data": {
            "user_idea": res["user_idea"],
            "source": res["source"],
            "modality": res["modality"],
            "provider": res["provider"],
            "model_used": res["model_used"],
            "board_type": res["board_type"],
            "media_summary": res["media_summary"],
            "schematic": res["schematic"],
            "safety_audit": res["safety_audit"],
            "firmware": res["firmware"],
            "raw_ino_code": res["raw_ino_code"],
            "voice_script": res["voice_script"],
            "has_audio": res["has_audio"],
            "elapsed_seconds": res["elapsed_seconds"],
            "logs": agents.get_recent_logs()
        }
    }


@app.post("/api/switch-mode")
def api_switch_mode(req: ModeSwitchRequest):
    """Allows switching between Google Cloud Gemma 4 and Local Ollama."""
    msg = agents.switch_engine_mode(req.provider, req.model if req.model else None)
    return {"status": "success", "message": msg, "pipeline_status": agents.PIPELINE_STATUS}


@app.post("/api/build-hardware")
def api_build_hardware(req: HardwareRequest):
    """Fallback endpoint if triggered from Web UI directly."""
    try:
        result = agents.run_hardware_pipeline(
            user_idea=req.idea,
            board_type=req.board_type,
            source="web-Trigger"
        )
        return {
            "status": "success",
            "model": result["model_used"],
            "provider": result["provider"],
            "data": {
                "user_idea": result["user_idea"],
                "source": result["source"],
                "modality": result["modality"],
                "board_type": result["board_type"],
                "media_summary": result["media_summary"],
                "schematic": result["schematic"],
                "safety_audit": result["safety_audit"],
                "firmware": result["firmware"],
                "raw_ino_code": result["raw_ino_code"],
                "voice_script": result["voice_script"],
                "has_audio": result["has_audio"],
                "elapsed_seconds": result["elapsed_seconds"],
                "logs": result["logs"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audio-stream")
def get_latest_audio():
    """Streams the latest synthesized MP3 voice briefing to the browser."""
    if agents.LATEST_AUDIO_BYTES:
        return Response(content=agents.LATEST_AUDIO_BYTES, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="No audio generated yet.")


@app.get("/api/logs")
def get_logs():
    """Returns the real-time AI observability log buffer."""
    return {"logs": agents.get_recent_logs(), "status": agents.PIPELINE_STATUS}


# =====================================================================
# TELEGRAM BOT CONTROLLER (TEXT + PHOTO + VOICE + PDF + /mode SWITCH)
# =====================================================================
def run_telegram_bot():
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "your_telegram_bot_token_here":
        agents.logger.info("Telegram Bot token not set in .env — waiting for token.")
        return

    try:
        from telegram import Update
        from telegram.ext import (
            ApplicationBuilder,
            CommandHandler,
            ContextTypes,
            MessageHandler,
            filters,
        )
    except ImportError:
        agents.logger.warning("python-telegram-bot not installed; skipping Telegram listener.")
        return

    async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        active_model = agents.OLLAMA_MODEL if agents.LLM_PROVIDER == "ollama" else agents.GEMINI_MODEL
        welcome = (
            f"🦥 LazyLock Multimodal Hardware Studio (Gemma 4)\n"
            f"⚡ Active Engine: {agents.LLM_PROVIDER.upper()} ({active_model})\n\n"
            "Send me ANY of these 4 inputs to build hardware:\n"
            "1️⃣ TEXT: Describe a circuit or IoT idea.\n"
            "2️⃣ PHOTO 📷: Snap a photo of components, a breadboard, or a sketch.\n"
            "3️⃣ VOICE NOTE 🎙️: Speak your hardware request hands-free.\n"
            "4️⃣ PDF 📄: Upload a sensor datasheet, pinout PDF, or project rubric.\n\n"
            "🔧 Commands:\n"
            "• /mode google — Use Google Cloud Gemma 4 API\n"
            "• /mode ollama — Use Local Offline Ollama Gemma\n"
            "• /status — Check active engine & model"
        )
        await update.message.reply_text(welcome)

    async def mode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args
        if not args:
            active_model = agents.OLLAMA_MODEL if agents.LLM_PROVIDER == "ollama" else agents.GEMINI_MODEL
            await update.message.reply_text(
                f"⚡ Current Engine: {agents.LLM_PROVIDER.upper()} ({active_model})\n"
                "Usage: `/mode google` or `/mode ollama [model_name]`"
            )
            return
        provider = args[0]
        custom_model = args[1] if len(args) > 1 else None
        msg = agents.switch_engine_mode(provider, custom_model)
        await update.message.reply_text(f"🔄 Engine Updated!\n{msg}")

    async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        st = agents.PIPELINE_STATUS
        await update.message.reply_text(
            f"📊 LazyLock System Status:\n"
            f"• Engine: {st['provider'].upper()}\n"
            f"• Model: {st['active_model']}\n"
            f"• Stage: {st['current_stage']}"
        )

    async def send_pipeline_reply(update: Update, result: dict):
        verdict = result["safety_audit"].get("safety_verdict", "CLEAR")
        voltage_warn = result["safety_audit"].get("voltage_warnings", "Verify 3.3V/5V rails.")
        bom = result["safety_audit"].get("bill_of_materials", "See Web Dashboard")

        media_section = ""
        if result.get("media_summary") and result["modality"] != "text/plain":
            media_section = f"\n🔍 Agent 0 Multimodal Analysis:\n{result['media_summary'][:550]}\n"

        summary = (
            f"✅ LazyLock Prototype Ready ({result['elapsed_seconds']}s)\n"
            f"⚙️ Engine: {result['provider'].upper()} ({result['model_used']})\n"
            f"🎯 Target Board: {result['board_type']} | Input: {result['modality']}\n"
            f"{media_section}\n"
            f"🛡️ Safety Verdict: {verdict}\n"
            f"⚡ Voltage Check: {voltage_warn[:280]}\n"
            f"🧰 BOM Preview: {str(bom)[:250]}\n\n"
            f"🎙️ Hands-Free Wiring Guide:\n\"{result['voice_script']}\"\n\n"
            f"🖥️ Full Mermaid circuit diagram, pinout table & live logs are live on the Web Monitor!"
        )
        await update.message.reply_text(summary[:4000])

        # 1. Attach downloadable .ino firmware file
        if result.get("raw_ino_code"):
            ino_bytes = result["raw_ino_code"].encode("utf-8")
            ino_file = io.BytesIO(ino_bytes)
            ino_file.name = "lazylock_prototype.ino"
            await update.message.reply_document(
                document=ino_file,
                filename="lazylock_prototype.ino",
                caption=f"💻 Ready-to-flash C++ Firmware (.ino) | Generated by {result['model_used']}"
            )

        # 2. Attach playable Hands-Free Voice Note
        if result.get("audio_bytes"):
            audio_stream = io.BytesIO(result["audio_bytes"])
            audio_stream.name = "wiring_guide.mp3"
            await update.message.reply_voice(
                voice=audio_stream,
                caption="🎧 Hands-Free Breadboard Wiring Walkthrough"
            )

    async def handle_telegram_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_text = update.message.text
        if not user_text:
            return

        active_model = agents.OLLAMA_MODEL if agents.LLM_PROVIDER == "ollama" else agents.GEMINI_MODEL
        await update.message.reply_text(
            f"🦥 [{agents.LLM_PROVIDER.upper()}: {active_model}] Running 4-Agent Hardware Pipeline for your text prompt..."
        )
        try:
            result = await asyncio.to_thread(
                agents.run_hardware_pipeline,
                user_text,
                "ESP32 DevKit V1",
                None,
                None,
                "telegram-text"
            )
            await send_pipeline_reply(update, result)
        except Exception as e:
            await update.message.reply_text(f"❌ Error generating hardware plan: {str(e)}")

    async def handle_telegram_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
        caption = update.message.caption or ""
        await update.message.reply_text(
            f"📷 [{agents.LLM_PROVIDER.upper()}] Agent 0 Vision is inspecting your hardware photo & launching Agents 1-3..."
        )
        try:
            photo_obj = update.message.photo[-1]
            tg_file = await photo_obj.get_file()
            buf = io.BytesIO()
            await tg_file.download_to_memory(out=buf)
            image_bytes = buf.getvalue()

            result = await asyncio.to_thread(
                agents.run_hardware_pipeline,
                caption,
                "ESP32 DevKit V1",
                image_bytes,
                "image/jpeg",
                "telegram-photo"
            )
            await send_pipeline_reply(update, result)
        except Exception as e:
            await update.message.reply_text(f"❌ Error analyzing hardware photo: {str(e)}")

    async def handle_telegram_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
        caption = update.message.caption or ""
        await update.message.reply_text(
            f"🎙️ [{agents.LLM_PROVIDER.upper()}] Agent 0 Audio is decoding your voice command & launching Agents 1-3..."
        )
        try:
            media_obj = update.message.voice or update.message.audio
            tg_file = await media_obj.get_file()
            buf = io.BytesIO()
            await tg_file.download_to_memory(out=buf)
            audio_bytes = buf.getvalue()
            mime_type = getattr(media_obj, "mime_type", None) or "audio/ogg"

            result = await asyncio.to_thread(
                agents.run_hardware_pipeline,
                caption,
                "ESP32 DevKit V1",
                audio_bytes,
                mime_type,
                "telegram-voice"
            )
            await send_pipeline_reply(update, result)
        except Exception as e:
            await update.message.reply_text(f"❌ Error processing voice command: {str(e)}")

    async def handle_telegram_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
        doc = update.message.document
        if not doc:
            return
        file_name = doc.file_name or "document.pdf"
        mime_type = doc.mime_type or ""
        if not (mime_type == "application/pdf" or file_name.lower().endswith(".pdf")):
            await update.message.reply_text("⚠️ Please send a PDF document (.pdf), Photo, Voice Note, or Text.")
            return

        caption = update.message.caption or f"Analyze {file_name} and design the hardware circuit & firmware."
        await update.message.reply_text(
            f"📄 [{agents.LLM_PROVIDER.upper()}] Agent 0 is parsing PDF '{file_name}' & launching Agents 1-3..."
        )
        try:
            tg_file = await doc.get_file()
            buf = io.BytesIO()
            await tg_file.download_to_memory(out=buf)
            pdf_bytes = buf.getvalue()

            result = await asyncio.to_thread(
                agents.run_hardware_pipeline,
                caption,
                "ESP32 DevKit V1",
                pdf_bytes,
                "application/pdf",
                "telegram-pdf"
            )
            await send_pipeline_reply(update, result)
        except Exception as e:
            await update.message.reply_text(f"❌ Error processing PDF datasheet: {str(e)}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("mode", mode_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_telegram_text))
    application.add_handler(MessageHandler(filters.PHOTO, handle_telegram_photo))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_telegram_voice))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_telegram_pdf))

    agents.logger.info("🤖 LazyLock Multimodal Telegram Controller (Text + Photo + Voice + PDF) started!")
    application.run_polling(stop_signals=None)


@app.on_event("startup")
def startup_event():
    if TELEGRAM_TOKEN and TELEGRAM_TOKEN != "your_telegram_bot_token_here":
        t = threading.Thread(target=run_telegram_bot, daemon=True)
        t.start()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)