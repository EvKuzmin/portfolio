from aiogram.utils.emoji import emojize

start_message='Будем знакомы: я - котенок Helper! *мур-р*\n\
    Найти героя, который поможет тебе в обучении?\n\
    Тогда жми "Найти ментора"\n\
    Или же ты сам хочешь стать спасителем для наших пользователей?\n\
    Тогда жми "Стать ментором" '

for_any_message = 'Что ты там мурчишь? Прости, не понимаю тебя('
print_name_message = 'Сейчас мы будем составлять твою анкету. Для начала, введи свое имя.'
error_choise_message = 'Ну ты чего? Я же предложил варианты...\n Давай-ка ты ограничишься ими :3'
choise_faculty_message = "С какого ты факультета?"
choise_theme_message = "Какие предметы интересует?"
choise_theme_rep_message = "Какие предметы - твоя сильная сторона?"
load_photo_message = "Загрузи фотокарточку, которая будет в твоей анкете. Это не обязательно, можешь пропустить."
about_learner_message = "(PASS)"
about_rep_message = "Расскажи немного о себе. Это информация будет отображаться в твоей анкете и поможет потенциальному клиенту выбрать именно тебя с:\n \
    А если ещё укажешь цену своим услугам, будет вобще замечательно :3"
final_rep_message = "Твоя анкета будет предоставлена нуждающимся ребяткам"
get_number_message = "Нам нужен твой номер, чтобы отправить его ученику, после чего он с тобой свяжется с:(временное решение)"
find_rep_message = "Сейчас кого-нибудь подберем"

MESSAGES = {
    'start': start_message,
    'any': for_any_message,
    'print_name': print_name_message,
    'error_chоise': error_choise_message,
    'choise_faculty': choise_faculty_message,
    'choise_theme': choise_theme_message,
    'choise_theme_rep': choise_theme_rep_message,
    'load_photo': load_photo_message,
    'about_learner': about_learner_message,
    'about_rep': about_rep_message,
    'final_rep': final_rep_message,
    'get_number': get_number_message,
    'find_rep': find_rep_message,
}