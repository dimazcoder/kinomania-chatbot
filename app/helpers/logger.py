import logging
import sys

logger = logging.getLogger("asyncio_logger")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.propagate = False
