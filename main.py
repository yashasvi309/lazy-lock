import os
import threading
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn

import hardware_agents as agents

app = FastAPI(title="LazyLock | 3-Agent AI Hardware Engineer & Telegram Bot")
templates = Jinja2Templates(directory="templates")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


class HardwareRequest(BaseModel):
    idea: str
    board_type: str = "ESP32 DevKit V1"


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/build-hardware")
def api_build_hardware(req: HardwareRequest):
    try:
        if not agents.validate_environment():
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY is missing in your .env file. Add your Google AI Studio key."
            )

        result = agents.run_hardware_pipeline(req.idea, req.board_type)
        response_data = {
            "board_type": result["board_type"],
            "schematic": result["schematic"],
            "safety_audit": result["safety_audit"],
            "firmware": result["firmware"],
            "raw_ino_code": result["raw_ino_code"],
            "voice_script": result["voice_script"],
            "has_audio": result["audio_bytes"] is not None,
            "logs": result["logs"]
        }
        return {
            "status": "success",
            "model": agents.GEMINI_MODEL,
            "data": response_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audio-stream")
def get_latest_audio():
    """Streams the latest ElevenLabs MP3 voice briefing to the browser."""
    if agents.LATEST_AUDIO_BYTES:
        return Response(content=agents.LATEST_AUDIO_BYTES, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="No ElevenLabs audio generated yet.")


@app.get("/api/logs")
def get_logs():
    """Returns the real-time AI observability log buffer."""
    return {"logs": agents.get_recent_logs()}


# =====================================================================
# TELEGRAM BOT LISTENER (OPTIONAL BACKGROUND THREAD)
# =====================================================================
def run_telegram_bot():
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "your_telegram_bot_token_here":
        agents.logger.info("Telegram Bot token not set in .env — Web Studio running standalone.")
        return

    try:
        from telegram import Update
        from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
    except ImportError:
        agents.logger.warning("python-telegram-bot not installed; skipping Telegram listener.")
        return

    async def handle_telegram_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_text = update.message.text
        if not user_text:
            return

        await update.message.reply_text(
            "🦥 *LazyLock AI* is designing your schematic, auditing voltage rails, and writing firmware...",
            parse_mode="Markdown"
        )

        try:
            result = agents.run_hardware_pipeline(user_text, "ESP32 DevKit V1")
            verdict = result["safety_audit"].get("safety_verdict", "CLEAR")
            summary = (
                f"✅ *Circuit & Firmware Ready!*\n\n"
                f"🛡️ *Safety Verdict:* `{verdict}`\n\n"
                f"🎙️ *Hands-Free Wiring Guide:*\n_{result['voice_script']}_\n\n"
                f"💻 Full pinout table & `.ino` firmware updated on your LazyLock Web Studio!"
            )
            await update.message.reply_text(summary, parse_mode="Markdown")

            if result.get("audio_bytes"):
                await update.message.reply_voice(voice=result["audio_bytes"])
        except Exception as e:
            await update.message.reply_text(f"❌ Error generating hardware plan: {str(e)}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_telegram_message))
    agents.logger.info("🤖 Telegram Bot background polling started!")
    application.run_polling()


@app.on_event("startup")
def startup_event():
    if TELEGRAM_TOKEN and TELEGRAM_TOKEN != "your_telegram_bot_token_here":
        t = threading.Thread(target=run_telegram_bot, daemon=True)
        t.start()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

    