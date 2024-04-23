from config import TOKEN
import asyncio
from aiogram import Bot
from aiogram.dispatcher import Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import logging
from handlers import register_handlers_learner, register_handlers_common, register_handlers_rep



async def main():
    bot = Bot(TOKEN, parse_mode="HTML")
    dp = Dispatcher(bot, storage=MemoryStorage())
    register_handlers_learner(dp)
    register_handlers_common(dp)
    register_handlers_rep(dp)
    await dp.start_polling()


if __name__ == "__main__":
    asyncio.run(main())
