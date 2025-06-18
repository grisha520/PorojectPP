from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer, util
import json
import random
import pandas as pd
import os

app = Flask(__name__)

# Инициализация глобальных переменных
dialog_history = []
SIMILARITY_THRESHOLD = 0.5
GREETINGS_LIST = ["привет", "здравствуй", "добрый день", "хай", "hello", "hi"]


# Загрузка модели и данных
def load_model():
    return SentenceTransformer("trained_embed_model")


def load_qa_pairs(jsonl_path):
    questions, answers = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if len(obj["texts"]) == 2:
                questions.append(obj["texts"][0])
                answers.append(obj["texts"][1])
    return questions, answers


model_embed = load_model()
questions_list, answers_list = load_qa_pairs("train_data.jsonl")


# Функции для обработки вопросов
def get_best_answer_with_threshold(model, question, questions_list, answers_list):
    question_embedding = model.encode(question, convert_to_tensor=True)
    db_embeddings = model.encode(questions_list, convert_to_tensor=True)

    similarities = util.cos_sim(question_embedding, db_embeddings)[0]
    top_idx = similarities.argmax().item()
    top_sim = similarities[top_idx].item()

    best_answer = answers_list[top_idx]
    dialog_history.append((question, best_answer))
    return best_answer, top_sim


def is_greeting(text: str) -> bool:
    return any(g in text.lower() for g in GREETINGS_LIST)


def is_thank_you(text: str) -> bool:
    thank_you_phrases = [
        "спасибо", "спасибо большое", "огромное спасибо", "спс", "благодарю",
        "спасибо тебе", "благодарствую", "спасибо за помощь", "thanks", "thank you",
        "спасибо огромное", "ты помог", "ты молодец", "спасибо за ответ", "спасибо, бот",
        "пасиб", "пасибки", "благодарю за всё", "респект", "благодарочка"
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in thank_you_phrases)


def is_goodbye(text: str) -> bool:
    goodbye_phrases = [
        "пока", "до свидания", "увидимся", "бай", "bye",
        "до встречи", "прощай", "всего доброго", "счастливо", "чао",
        "мне пора", "я пошёл", "ещё увидимся", "увидимся позже", "я ухожу",
        "выключаюсь", "до завтра", "всё, пока", "бот, пока", "счастливо оставаться"
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in goodbye_phrases)


def get_greeting_response():
    return "Привет! Чем могу помочь?"


def get_thank_you_response():
    if not dialog_history:
        return random.choice([
            "Я ещё ничего не сделал, но спасибо!",
            "Спасибо, но я пока не помог!",
            "Благодарю, хотя я ещё не участвовал в диалоге!",
            "Спасибо за вежливость, но я жду вашего вопроса!",
            "Приятно слышать благодарность, но давайте начнём диалог!",
        ])
    else:
        return random.choice([
            "Пожалуйста! Обращайтесь ещё!",
            "Всегда рад помочь!",
            "Не стоит благодарности!",
            "Рад был помочь!",
        ])


def get_goodbye_response():
    if not dialog_history:
        return random.choice([
            "Прощайте!",
            "До свидания!",
            "Всего доброго!",
            "Пока!",
        ])
    else:
        return random.choice([
            "До свидания! Удачи!",
            "Был рад помочь! До новых встреч!",
            "До встречи!",
        ])


# Маршруты Flask
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(force=True)
    score = data.get("score", 0)
    subject = data.get("subject", "informatics")

    df = pd.read_excel("Направления с предпочтениями.xlsx")

    min_score_col = "Минимальный балл ЕГЭ(информатика)" if subject == "informatics" else "Минимальный балл ЕГЭ(физика)"

    available_directions = df[df[min_score_col] <= score].copy()

    available_directions["accessibility"] = score - available_directions["Проходной балл ЕГЭ(Бюджет)"]

    available_directions = available_directions.sort_values("accessibility", ascending=False)

    if not available_directions.empty:
        directions_list = []
        for _, row in available_directions.iterrows():
            directions_list.append({
                "name": row["Направление"],
                "min_score": row[min_score_col],
                "avg_score": row["Проходной балл ЕГЭ(Бюджет)"],
                "accessibility": row["accessibility"],
                "budget_places": row["Бюджетных мест"]
            })

        return jsonify({
            "status": "success",
            "score": score,
            "subject": subject,
            "available_directions": directions_list
        })
    else:
        return jsonify({
            "status": "no_directions",
            "message": "К сожалению, с вашим баллом доступных направлений не найдено.",
            "min_possible_score": df[min_score_col].min()
        })


@app.route("/")
def index():
    return send_from_directory("static", "index_v3.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    raw_q = data.get("question", "").strip()

    if not raw_q:
        return jsonify({
            "question": raw_q,
            "answer": "Пожалуйста, задайте вопрос.",
            "history": dialog_history[-3:] if len(dialog_history) > 3 else dialog_history,
            "close_chat": False
        })

    if is_thank_you(raw_q):
        answer = get_thank_you_response()
        return jsonify({
            "question": raw_q,
            "answer": answer,
            "history": dialog_history[-3:] if len(dialog_history) > 3 else dialog_history,
            "close_chat": False
        })

    if is_goodbye(raw_q):
        answer = get_goodbye_response()
        dialog_history.clear()
        return jsonify({
            "question": raw_q,
            "answer": answer,
            "history": [],
            "close_chat": True
        })

    if is_greeting(raw_q):
        words = raw_q.split()
        if len(words) <= 2:
            answer = get_greeting_response()
            return jsonify({
                "question": raw_q,
                "answer": answer,
                "history": dialog_history[-3:] if len(dialog_history) > 3 else dialog_history,
                "close_chat": False
            })

        for greet in GREETINGS_LIST:
            if raw_q.lower().startswith(greet):
                raw_q = raw_q[len(greet):].strip(",.?! ").strip()
                break

    answer, similarity = get_best_answer_with_threshold(model_embed, raw_q, questions_list, answers_list)

    if similarity < SIMILARITY_THRESHOLD:
        answer = "Извините, я не понял вопрос. Пожалуйста, уточните."

    return jsonify({
        "question": raw_q,
        "answer": answer,
        "history": dialog_history[-3:] if len(dialog_history) > 3 else dialog_history,
        "close_chat": False
    })


if __name__ == "__main__":
    app.run(debug=True)
