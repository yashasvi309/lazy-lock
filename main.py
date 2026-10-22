import os
import threading
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response
from pydantic import BaseModel
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import uvicorn

import hardware_agents as agents

app = FastAPI(title="SlothLab | AI Hardware Engineer & Telegram Bot")
templates = Jinja2Templates(directory="templates")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

class HardwareRequest(BaseModel):
    idea: str

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/build-hardware")
def api_build_hardware(req: HardwareRequest):
    try:
        result = agents.run_hardware_pipeline(req.idea)
        # Exclude raw bytes from JSON web response
        response_data = {
            "schematic": result["schematic"],
            "safety_audit": result["safety_audit"],
            "firmware": result["firmware"],
            "has_audio": result["audio_bytes"] is not None
        }
        return {"status": "success", "data": response_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/audio-stream")
def get_latest_audio():
    # Helper endpoint to stream audio if generated
    return Response(content=b"", media_type="audio/mpeg")


# =====================================================================
# TELEGRAM BOT LISTENER (BACKGROUND THREAD)
# =====================================================================
async def handle_telegram_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not user_text:
        return
    
    await update.message.reply_text("🦥 SlothLab AI is analyzing your hardware request and drafting schematics...")
    
    # Run pipeline in background thread
    try:
        result = agents.run_hardware_pipeline(user_text)
        
        # Send text response summary
        summary = (
            f"✅ **Circuit Designed!**\n\n"
            f"🛡️ **Safety Verdict:** {result['safety_audit'].get('safety_verdict', 'CLEAR')}\n\n"
            f"Check the web dashboard for full pinouts, BOM, and Arduino firmware."
        )
        await update.message.reply_text(summary, parse_mode="Markdown")

        # Send ElevenLabs Voice Note if available
        if result.get("audio_bytes"):
            await update.message.reply_voice(voice=result["audio_bytes"])

    except Exception as e:
        await update.message.reply_text(f"❌ Hardware generation failed: {str(e)}")

def run_telegram_bot():
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "your_telegram_bot_token_here":
        print("ℹ️ Telegram bot token not provided. Bot polling skipped.")
        return
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_telegram_message))
    
    print("🤖 Telegram Bot polling started successfully...")
    application.run_polling()


@app.on_event("startup")
def startup_event():
    if TELEGRAM_TOKEN and TELEGRAM_TOKEN != "your_telegram_bot_token_here":
        t = threading.Thread(target=run_telegram_bot, daemon=True)
        t.start()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)