from functools import partial

from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from app.core.config import config
from app.handlers.v1.start_handler import handle_start
from app.handlers.v1.text_handler import handle_text
from app.helpers.logger import logger


def start_bot(gen_model):
    bot = ApplicationBuilder().token(config.telegram_bot_token).build()

    bot.add_handler(CommandHandler("start", handle_start))
    # bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    bot.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        partial(handle_text, gen_model=gen_model)
    ))

    logger.info("Bot started")

    bot.run_polling()