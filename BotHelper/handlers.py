from config import admin_id
from messages import MESSAGES
from aiogram.types import Message, ReplyKeyboardRemove
from aiogram.types.message import ContentType
from aiogram.utils.markdown import text, bold, italic, code, pre
from aiogram.types import ParseMode
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext, Dispatcher
import keyboards as kb
from myutils import MainStates


async def process_start_command(message: Message):
    await MainStates.start_state.set()
    await message.answer(MESSAGES['start'], reply_markup=kb.start_kb)

async def process_cancel_command(message: Message):
    await MainStates.start_state.set()
    await message.answer(MESSAGES['start'],reply_markup=kb.start_kb)

#линейка анкетирования ученика
async def find_rep_pointed(message: Message):
    await MainStates.waiting_for_faculty.set()
    await message.answer(MESSAGES['choise_faculty'], reply_markup=kb.faculties_kb)


async def faculty_pointed(message: Message, state: FSMContext):
    if (message.text not in kb.FACULTIES):
        await message.answer(MESSAGES['error_chоise'])
        return

    if (message.text=="На главную"):
        await process_cancel_command(message)
        return

    await state.update_data(faculty_learner=message.text)
    await MainStates.waiting_for_theme.set()
    await message.answer(MESSAGES['choise_theme'],reply_markup=kb.themes_kb)

async def theme_pointed(message: Message, state: FSMContext):
    if message.text!="Пропустить":
        state.update_data(info_learner="")
    else:
        state.update_data(info_learner=message.text)
    if (message.text=="На главную"):
        await process_cancel_command(message)
        return
    
    #ДОБАВИТЬ ЗАПИСИ В БАЗУ ДАННЫХ
    await MainStates.cycle_of_search.set()
    await message.answer(MESSAGES['find_rep'], reply_markup=kb.search_kb)




def register_handlers_learner(dp: Dispatcher):
    dp.register_message_handler(find_rep_pointed, Text(equals="Найти ментора", ignore_case=True), state=MainStates.start_state)
    dp.register_message_handler(faculty_pointed, state=MainStates.waiting_for_faculty)
    dp.register_message_handler(theme_pointed, state=MainStates.waiting_for_theme)
#конец линейки анкетирования ученика

#линейка анкетирования репетитора
async def become_rep_pointed(message: Message):
    await message.answer(MESSAGES['print_name'], reply_markup=kb.print_name_kb)
    await MainStates.waiting_for_name_rep.set()

async def name_rep_pointed(message: Message, state: FSMContext):
    if (message.text==""):
        await message.answer("Ну хоть один символ:с")
        return

    if (message.text=="На главную"):
        await process_cancel_command(message)
        return

    await state.update_data(name_rep=message.text)
    await MainStates.waiting_for_theme_rep.set()
    await message.answer(MESSAGES['choise_theme_rep'], reply_markup=kb.themes_kb)

async def theme_rep_pointed(message: Message, state: FSMContext):
    if (message.text not in kb.THEMES):
        await message.answer(MESSAGES['error_chоise'])
        return
    
    if (message.text=="На главную"):
        await process_cancel_command(message)
        return
    #СФОРМИРОВАТЬ СПИСОК ВЫБРАННЫХ ПРЕДМЕТОВ И ЗАГРУЗИТЬ В theme_rep
    await state.update_data(theme_rep=message.text)
    await MainStates.waiting_for_photo_rep.set()
    await message.answer(MESSAGES['load_photo'], reply_markup=kb.print_name_kb)

async def photo_rep_got(message: Message, state: FSMContext):

    if (message.text=="На главную"):
        await process_cancel_command(message)
        return
    #загрузить фото и добавить id в бд  
    await MainStates.waiting_for_number_rep.set()
    await message.answer(MESSAGES['get_number'],reply_markup=kb.number_kb)

async def number_rep_got(message: Message, state: FSMContext):  
      await message.answer(MESSAGES['final_rep'],reply_markup=kb.number_kb)

def register_handlers_rep(dp: Dispatcher):
    dp.register_message_handler(become_rep_pointed, Text(equals="Стать ментором", ignore_case=True), state=MainStates.start_state)
    dp.register_message_handler(name_rep_pointed, state=MainStates.waiting_for_name_rep)
    dp.register_message_handler(theme_rep_pointed, state=MainStates.waiting_for_theme_rep)
    dp.register_message_handler(photo_rep_got, content_types=["photo"], state=MainStates.waiting_for_photo_rep)
    dp.register_message_handler(number_rep_got, state=MainStates.waiting_for_number_rep)



#конец линейки анкетирования репетитора





def register_handlers_common(dp: Dispatcher):
    dp.register_message_handler(process_start_command, commands="start", state="*")
    dp.register_message_handler(process_cancel_command, Text(equals="На главную", ignore_case=True), state="*")





