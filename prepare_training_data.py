import pandas as pd
from sentence_transformers import InputExample
import json
import random

# Загрузка данных с предпочтениями
df = pd.read_excel("Направления с предпочтениями.xlsx", sheet_name="Лист1", engine="openpyxl").fillna("не указано")

# Создаем индекс предпочтений: предпочтение -> [направления]
preference_index = {}
for _, row in df.iterrows():
    if "Интересы (предпочтения)" in row:
        interests = str(row["Интересы (предпочтения)"])
        if interests != "не указано" and interests.strip():
            for interest in interests.split(","):
                clean_interest = interest.strip()
                if clean_interest:
                    if clean_interest not in preference_index:
                        preference_index[clean_interest] = []
                    preference_index[clean_interest].append(row["Направление"])

# Собираем все уникальные предпочтения
all_interests = set()
for interests in df["Интересы (предпочтения)"]:
    if isinstance(interests, str) and interests != "не указано":
        for interest in interests.split(","):
            clean_interest = interest.strip()
            if clean_interest:
                all_interests.add(clean_interest)

QUESTION_TEMPLATES = {
    "прох_балл_бюджет": [
        "Какой проходной балл на бюджет по направлению {dir}?",
        "Сколько баллов нужно для поступления на бюджет {dir}?",
        "Проходной балл ЕГЭ (бюджет) для {dir}?",
        "Какой минимальный балл для поступления на бюджет {dir}?"
    ],
    "мин_балл_информатика": [
        "Какой минимальный балл по информатике нужен для {dir}?",
        "Сколько нужно набрать по информатике для поступления на {dir}?",
        "Минимальный проходной балл по информатике для {dir}?"
    ],
    "мин_балл_физика": [
        "Какой минимальный балл по физике нужен для {dir}?",
        "Сколько нужно набрать по физике для поступления на {dir}?",
        "Минимальный проходной балл по физике для {dir}?"
    ],
    "бюджетных_мест": [
        "Сколько бюджетных мест на направлении {dir}?",
        "Количество бюджетных мест для {dir}?",
        "Сколько мест на бюджет по {dir}?"
    ],
    "контрактных_мест": [
        "Сколько контрактных мест на направлении {dir}?",
        "Количество контрактных мест для {dir}?",
        "Сколько мест на контракт по {dir}?"
    ],
    "баллы": [
        "Какой проходной балл на {dir}?", "Сколько баллов нужно для {dir}?",
        "Проходной балл ЕГЭ для {dir}?", "Какой минимум баллов нужен на {dir}?",
        "С каким баллом поступают на {dir}?", "Баллы для поступления на {dir}?",
        "Сколько нужно набрать, чтобы попасть на {dir}?",
    ],
    "предметы": [
        "Какие предметы нужны для поступления на {dir}?", "Какие ЕГЭ нужно сдавать для {dir}?",
        "Предметы для поступления на {dir}?", "Что сдавать на ЕГЭ для {dir}?",
        "Какие дисциплины требуются для {dir}?", "Что входит в список предметов на {dir}?",
        "Без каких предметов не поступить на {dir}?",
    ],
    "профессии": [
        "Кем можно работать после {dir}?", "Какие профессии после {dir}?",
        "На кого учат в направлении {dir}?", "Какие карьерные перспективы у {dir}?",
        "Какие специальности связаны с {dir}?", "Где можно работать после окончания {dir}?",
        "Куда идут выпускники {dir}?",
    ],
    "специализация": [
        "Что изучают на {dir}?", "Какая специализация у {dir}?",
        "Основные темы направления {dir}?", "На чем специализируется {dir}?",
        "Чему обучают на {dir}?", "Какие предметы изучают студенты {dir}?",
        "Что входит в программу {dir}?",
    ],
    "предпочтения": [
        "Какие интересы нужны для направления {dir}?",
        "Какие предпочтения подходят для {dir}?",
        "Какие интересы учитываются на {dir}?",
        "Что должно нравиться для {dir}?",
        "Какие увлечения подходят для направления {dir}?",
        "Подойдет ли направление {dir} если мои интересы: {interest}?",
        "Смогу ли я учиться на {dir} если мне нравится {interest}?",
        "Какие направления подходят для интереса {interest}?",
        "Где изучают {interest}?",
        "На каком направлении есть {interest}?",
        "Куда поступить, если мне нравится {interest}?",
        "Какие направления связаны с {interest}?",
    ]
}

def random_prefix():
    return random.choice(["", "Интересно,", "Скажи, пожалуйста,", "Подскажи,", "Можно узнать,", "Напомни,", "Если не сложно,"])

def random_suffix():
    return random.choice(["", "Спасибо!", "Это важно для меня.", "Очень нужно знать.", "Планирую поступать туда.", "Рассматриваю этот вариант."])

def generate_answer(fact_type, fact_value, direction=None, interest=None):
    templates = {
        "прох_балл_бюджет": [
            f"Проходной балл на бюджет по направлению «{direction}» составлял {fact_value}.",
            f"Для поступления на бюджет в направлении «{direction}» нужно было набрать около {fact_value} баллов.",
            f"Минимальный балл для бюджетного поступления на «{direction}» — {fact_value}."
        ],
        "мин_балл_информатика": [
            f"Минимальный балл по информатике для направления «{direction}» — {fact_value}.",
            f"Для поступления на «{direction}» с информатикой нужно набрать минимум {fact_value} баллов.",
            f"По информатике для «{direction}» требуется как минимум {fact_value} баллов."
        ],
        "мин_балл_физика": [
            f"Минимальный балл по физике для направления «{direction}» — {fact_value}.",
            f"Для поступления на «{direction}» с физикой нужно набрать минимум {fact_value} баллов.",
            f"По физике для «{direction}» требуется как минимум {fact_value} баллов."
        ],
        "бюджетных_мест": [
            f"Количество бюджетных мест на направлении «{direction}» — {fact_value}.",
            f"Для «{direction}» выделено {fact_value} бюджетных мест.",
            f"Бюджетных мест на «{direction}»: {fact_value}."
        ],
        "контрактных_мест": [
            f"Количество контрактных мест на направлении «{direction}» — {fact_value}.",
            f"Для «{direction}» выделено {fact_value} контрактных мест.",
            f"Контрактных мест на «{direction}»: {fact_value}."
        ],
        "баллы": [
            f"На направление «{direction}» проходной балл в прошлом году был {fact_value}.",
            f"Чтобы поступить на «{direction}», нужно было набрать около {fact_value} баллов.",
            f"Проходной балл на «{direction}» составлял примерно {fact_value}.",
            f"Минимальный балл для поступления на «{direction}» — {fact_value}.",
            f"Обычно поступают на «{direction}» с результатом около {fact_value}."
        ],
        "предметы": [
            f"Для поступления на «{direction}» нужно сдавать следующие предметы: {fact_value}.",
            f"ЕГЭ по направлениям: {fact_value} — это требование для «{direction}».",
            f"Абитуриенту на «{direction}» понадобятся предметы: {fact_value}.",
            f"Список ЕГЭ для направления «{direction}»: {fact_value}.",
            f"Поступающие на «{direction}» выбирают: {fact_value}."
        ],
        "профессии": [
            f"После «{direction}» можно работать как: {fact_value}.",
            f"Выпускники направления «{direction}» находят себя в профессиях: {fact_value}.",
            f"Карьера после «{direction}» может включать: {fact_value}.",
            f"Сфера деятельности выпускников «{direction}»: {fact_value}.",
            f"Диплом «{direction}» открывает путь к профессиям: {fact_value}."
        ],
        "специализация": [
            f"На «{direction}» изучают: {fact_value}.",
            f"Программа «{direction}» охватывает: {fact_value}.",
            f"Специализация включает: {fact_value}.",
            f"Основные дисциплины направления «{direction}»: {fact_value}.",
            f"Учебный план «{direction}» включает: {fact_value}."
        ],
        "предпочтения": [
            f"Если тебе интересно {fact_value}, то {direction} - отличный выбор! 😊",
            f"Да, {direction} идеально подойдет, если тебе нравится {fact_value}!",
            f"Вижу совпадение! Твои интересы к {fact_value} и {direction} хорошо сочетаются 👍",
            f"С такими увлечениями как {fact_value} тебе понравится на направление {direction}",
            f"Точно угадал! направления {direction} как раз для тех, кому интересно {fact_value}",
        ]
    }
    
    # Обработка специальных случаев для предпочтений
    if fact_type == "предпочтения":
        # Вопрос о конкретном предпочтении и направлении
        if direction and interest:
            if direction in preference_index.get(interest, []):
                return random.choice([
                    f"Да, направление «{direction}» идеально подходит для интереса к {interest}!",
                    f"Безусловно! Интерес к {interest} является ключевым для направления «{direction}».",
                    f"Совершенно верно! На направлении «{direction}» вы сможете развивать свои знания в области {interest}."
                ])
            else:
                return random.choice([
                    f"Нет, направление «{direction}» не фокусируется на {interest}.",
                    f"К сожалению, {interest} не является основным интересом для «{direction}».",
                    f"Направление «{direction}» не специализируется на {interest}."
                ])
        
        # Вопрос о направлениях для конкретного интереса
        elif interest:
            directions = preference_index.get(interest, [])
            if directions:
                if len(directions) > 1:
                    dir_list = ", ".join(directions[:-1]) + " и " + directions[-1]
                else:
                    dir_list = directions[0]
                    
                return random.choice([
                    f"Вы можете изучать {interest} на следующих направлениях: {dir_list}.",
                    f"Если вас интересует {interest}, то вы можете поступить на следующие направления: {dir_list}.",
                    f"С такими интересами вы можете постпить на: {dir_list}."
                ])
            else:
                return f"К сожалению, направлений, связанных с {interest}, не найдено."
    
    # Стандартная обработка
    return random.choice(templates.get(fact_type, [fact_value]))

train_data = []
MAX_EXAMPLES = 2000  # Увеличили лимит для новых данных
attempts = 0

# 1. Генерация вопросов о конкретных предпочтениях (без привязки к направлению)
for interest in all_interests:
    for q_tpl in [tpl for tpl in QUESTION_TEMPLATES["предпочтения"] if "{interest}" in tpl and "{dir}" not in tpl]:
        question = f"{random_prefix()} {q_tpl.format(interest=interest)} {random_suffix()}".strip()
        answer = generate_answer("предпочтения", None, interest=interest)
        train_data.append(InputExample(texts=[question, answer]))
        
        if len(train_data) >= MAX_EXAMPLES:
            break
    if len(train_data) >= MAX_EXAMPLES:
        break

# 2. Основной цикл обработки направлений
for _, row in df.iterrows():
    direction = row["Направление"]
    
    # Основные факты
    facts = {
        "прох_балл_бюджет": str(row.get("Проходной балл ЕГЭ(Бюджет)", "не указано")),
        "мин_балл_информатика": str(row.get("Минимальный балл ЕГЭ(информатика)", "не указано")),
        "мин_балл_физика": str(row.get("Минимальный балл ЕГЭ(физика)", "не указано")),
        "бюджетных_мест": str(row.get("Бюджетных мест", "не указано")),
        "контрактных_мест": str(row.get("Контрактных мест", "не указано")),
        "баллы": str(row.get("Проходной балл ЕГЭ(Бюджет)", "не указаno")),
        "предметы": str(row.get("Предметы ЕГЭ", "не указано")),
        "профессии": str(row.get("Будущие профессии", "не указано")),
        "специализация": str(row.get("Специализация", "не указано")) if "Специализация" in row else "не указано",
    }

    # Обработка предпочтений для направления
    if "Интересы (предпочтения)" in row:
        interests = str(row["Интересы (предпочтения)"])
        if interests != "не указано" and interests.strip():
            # Форматируем интересы в читаемый вид
            interests_list = [i.strip() for i in interests.split(",") if i.strip()]
            formatted_interests = ", ".join(interests_list[:-1]) + " и " + interests_list[-1] if len(interests_list) > 1 else interests_list[0]
            
            # Генерация общих вопросов о предпочтениях для направления
            for q_tpl in [tpl for tpl in QUESTION_TEMPLATES["предпочтения"] if "{dir}" in tpl and "{interest}" not in tpl]:
                question = f"{random_prefix()} {q_tpl.format(dir=direction)} {random_suffix()}".strip()
                answer = generate_answer("предпочтения", formatted_interests, direction=direction)
                train_data.append(InputExample(texts=[question, answer]))
                if len(train_data) >= MAX_EXAMPLES:
                    break
            
            # Генерация вопросов о конкретных предпочтениях для направления
            for interest in interests_list:
                for q_tpl in [tpl for tpl in QUESTION_TEMPLATES["предпочтения"] if "{dir}" in tpl and "{interest}" in tpl]:
                    question = f"{random_prefix()} {q_tpl.format(dir=direction, interest=interest)} {random_suffix()}".strip()
                    answer = generate_answer(
                        "предпочтения", 
                        None, 
                        direction=direction,
                        interest=interest
                    )
                    train_data.append(InputExample(texts=[question, answer]))
                    if len(train_data) >= MAX_EXAMPLES:
                        break
                if len(train_data) >= MAX_EXAMPLES:
                    break
            if len(train_data) >= MAX_EXAMPLES:
                break

    # Генерация вопросов для остальных фактов
    for fact_type, templates_list in QUESTION_TEMPLATES.items():
        if fact_type == "предпочтения":  # Уже обработали
            continue
            
        fact_value = facts.get(fact_type, "не указано")
        if fact_value == "не указано" or fact_value.strip() == "":
            continue

        for q_tpl in templates_list:
            question = f"{random_prefix()} {q_tpl.format(dir=direction)} {random_suffix()}".strip()
            for _ in range(random.randint(1, 3)):
                answer = generate_answer(fact_type, fact_value, direction=direction)
                train_data.append(InputExample(texts=[question, answer]))
                if len(train_data) >= MAX_EXAMPLES:
                    break
            if len(train_data) >= MAX_EXAMPLES:
                break
        if len(train_data) >= MAX_EXAMPLES:
            break
    if len(train_data) >= MAX_EXAMPLES:
        break

# Благодарности и прощания
thank_you_phrases = [
    "спасибо", "спасибо большое", "огромное спасибо", "спс", "благодарю",
    "спасибо тебе", "благодарствую", "спасибо за помощь", "thanks", "thank you",
    "спасибо огромное", "ты помог", "ты молодец", "спасибо за ответ", "спасибо, бот",
    "пасиб", "пасибки", "благодарю за всё", "респект", "благодарочка"
]
thank_you_responses = [
    "Я ещё ничего не сделал, но спасибо!",
    "Спасибо, но я пока не помог!",
    "Благодарю, хотя я ещё не участвовал в диалоге!",
    "Спасибо за вежливость, но я жду вашего вопроса!",
    "Приятно слышать благодарность, но давайте начнём диалог!",
]

goodbye_phrases = [
    "пока", "до свидания", "увидимся", "бай", "bye",
    "до встречи", "прощай", "всего доброго", "счастливо", "чао",
    "мне пора", "я пошёл", "ещё увидимся", "увидимся позже", "я ухожу",
    "выключаюсь", "до завтра", "всё, пока", "бот, пока", "счастливо оставаться"
]
goodbye_responses = [
    "Прощайте!",
    "До свидания!",
    "Всего доброго!",
    "Пока!"
]

for phrase in thank_you_phrases:
    for response in thank_you_responses:
        train_data.append(InputExample(texts=[phrase, response]))

for phrase in goodbye_phrases:
    for response in goodbye_responses:
        train_data.append(InputExample(texts=[phrase, response]))

print(f"➕ Добавлено {len(thank_you_phrases) * len(thank_you_responses)} благодарностей и {len(goodbye_phrases) * len(goodbye_responses)} прощаний.")

with open("train_data.jsonl", "w", encoding="utf-8") as f:
    for example in train_data:
        json.dump({"texts": example.texts}, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Сгенерировано примеров: {len(train_data)}")
