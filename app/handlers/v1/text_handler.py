import json

from telegram import Update
from telegram.ext import ContextTypes


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE, gen_model):
    # await update.message.reply_text(f"Ты написал: {update.message.text}")
    message = update.message.text
    result = gen_model.generate(message)
    await update.message.reply_text(json.dumps(result, ensure_ascii=False, indent=2))
