from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton

cancel_button = KeyboardButton("На главную")

start_kb = ReplyKeyboardMarkup(
    resize_keyboard=True
).row(KeyboardButton("Найти ментора"), \
    KeyboardButton("Стать ментором"))

print_name_kb = ReplyKeyboardMarkup(resize_keyboard=True).add(cancel_button)

FACULTIES=["ММФ","ЭФ"] #СДЕЛАТЬ БАЗУ
faculties_kb = ReplyKeyboardMarkup(resize_keyboard=True)
for fac in FACULTIES:
    faculties_kb.add(KeyboardButton(fac))
faculties_kb.add(cancel_button)

THEMES = ["Математический анализ","Высшая алгебра","Дискретная математика"] #СДЕЛАТЬ БАЗУ С УПРОРЯДОЧИВАНИЕМ ПО АЛФАВИТУ
themes_kb = ReplyKeyboardMarkup(resize_keyboard=True)
for th in THEMES:
    themes_kb.add(KeyboardButton(th))
themes_kb.add(cancel_button)


info_kb = ReplyKeyboardMarkup(resize_keyboard=True).add(KeyboardButton("Пропустить")).add(cancel_button)


number_kb = ReplyKeyboardMarkup(resize_keyboard=True).add(KeyboardButton(text="Отправить номер", request_contact=True)).add(cancel_button)
search_kb = ReplyKeyboardMarkup(resize_keyboard=True).add(KeyboardButton("Получить контакт")).add(KeyboardButton("Далее")).add(cancel_button)