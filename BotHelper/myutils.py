from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext


class MainStates(StatesGroup):
    start_state = State()
    waiting_for_name_rep = State()
    waiting_for_faculty = State()
    waiting_for_theme = State()
    waiting_for_theme_rep = State()
    waiting_for_info_rep = State()
    waiting_for_photo_rep = State()
    cycle_of_search = State()
    waiting_for_number_rep = State()

    