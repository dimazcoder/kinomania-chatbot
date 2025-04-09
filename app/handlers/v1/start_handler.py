from telegram import Update
from telegram.ext import ContextTypes


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет, я бот киномании!")
