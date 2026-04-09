import pandas as pd
import json
import os
from datetime import datetime
from collections import Counter
import re

# Файлы для хранения данных
QUERY_HISTORY_FILE = 'query_history.json'
USER_PREFERENCES_FILE = 'user_preferences.json'

# Загрузка истории запросов
if os.path.exists(QUERY_HISTORY_FILE):
    with open(QUERY_HISTORY_FILE, 'r', encoding='utf-8') as f:
        query_history = json.load(f)
else:
    query_history = []

# Загрузка предпочтений пользователей
if os.path.exists(USER_PREFERENCES_FILE):
    with open(USER_PREFERENCES_FILE, 'r', encoding='utf-8') as f:
        user_preferences = json.load(f)
else:
    user_preferences = {}


def save_query(user_id: str, query: str, results: list):
    """Сохраняет запрос пользователя в историю"""
    query_entry = {
        'user_id': user_id,
        'query': query,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': [r[0] for r in results[:5]]  # Сохраняем только названия топ-5
    }

    query_history.append(query_entry)

    # Сохранение в файл
    with open(QUERY_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(query_history, f, ensure_ascii=False, indent=2)


def analyze_user_preferences(user_id: str):
    """Анализирует предпочтения пользователя на основе истории запросов"""
    user_queries = [q for q in query_history if q['user_id'] == user_id]

    if len(user_queries) == 0:
        return None

    # Анализ ключевых слов
    all_queries = ' '.join([q['query'].lower() for q in user_queries])

    # Подсчет жанров
    genres = {
        'action': all_queries.count('action') + all_queries.count('экшен'),
        'strategy': all_queries.count('strategy') + all_queries.count('стратегия'),
        'rpg': all_queries.count('rpg') + all_queries.count('рпг'),
        'shooter': all_queries.count('shooter') + all_queries.count('шутер'),
        'adventure': all_queries.count('adventure') + all_queries.count('приключение'),
        'indie': all_queries.count('indie') + all_queries.count('инди'),
        'simulation': all_queries.count('simulation') + all_queries.count('симулятор'),
        'horror': all_queries.count('horror') + all_queries.count('хоррор'),
    }

    # Подсчет характеристик
    characteristics = {
        'популярный': all_queries.count('популярный') + all_queries.count('popular'),
        'новый': all_queries.count('новый') + all_queries.count('new'),
        'дешевый': all_queries.count('дешевый') + all_queries.count('cheap') + all_queries.count('free'),
        'хороший': all_queries.count('хороший') + all_queries.count('good'),
    }

    preferences = {
        'user_id': user_id,
        'total_queries': len(user_queries),
        'favorite_genres': sorted(genres.items(), key=lambda x: x[1], reverse=True)[:3],
        'favorite_characteristics': sorted(characteristics.items(), key=lambda x: x[1], reverse=True)[:3],
        'last_query': user_queries[-1]['query'],
        'last_query_time': user_queries[-1]['timestamp']
    }

    # Сохранение предпочтений
    user_preferences[user_id] = preferences
    with open(USER_PREFERENCES_FILE, 'w', encoding='utf-8') as f:
        json.dump(user_preferences, f, ensure_ascii=False, indent=2)

    return preferences


def get_personalized_recommendations(user_id: str, df, top_k: int = 5):
    """Генерирует персональные рекомендации на основе истории пользователя"""
    preferences = analyze_user_preferences(user_id)

    if preferences is None:
        return None

    scores = []

    for idx, row in df.iterrows():
        score = 0
        genre_text = str(row['genre']).lower()
        tags_text = str(row['popular_tags']).lower()

        # Бонус за любимые жанры
        for genre, count in preferences['favorite_genres']:
            if count > 0 and genre in genre_text or genre in tags_text:
                score += 30 + (count * 5)  # Чем чаще искал, тем больше бонус

        # Бонус за любимые характеристики
        from main import extract_rating, extract_price, extract_year

        rating = extract_rating(row['all_reviews'])
        price = extract_price(row['original_price'])
        year = extract_year(row['release_date'])

        for char, count in preferences['favorite_characteristics']:
            if count > 0:
                if char == 'популярный' and rating > 80:
                    score += 20 + (count * 3)
                elif char == 'новый' and year >= 2018:
                    score += 20 + (count * 3)
                elif char == 'дешевый' and price < 10:
                    score += 20 + (count * 3)
                elif char == 'хороший' and rating > 85:
                    score += 20 + (count * 3)

        # Базовый скор на основе рейтинга
        score += rating * 0.2

        scores.append((row['name'], score, row))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def get_query_statistics():
    """Возвращает статистику по всем запросам"""
    if len(query_history) == 0:
        return None

    # Общая статистика
    total_queries = len(query_history)
    unique_users = len(set([q['user_id'] for q in query_history]))

    # Популярные запросы
    all_queries = [q['query'].lower() for q in query_history]
    query_counter = Counter(all_queries)
    popular_queries = query_counter.most_common(10)

    # Популярные жанры в запросах
    all_text = ' '.join(all_queries)
    genres_count = {
        'action': all_text.count('action') + all_text.count('экшен'),
        'strategy': all_text.count('strategy') + all_text.count('стратегия'),
        'rpg': all_text.count('rpg') + all_text.count('рпг'),
        'shooter': all_text.count('shooter') + all_text.count('шутер'),
        'adventure': all_text.count('adventure') + all_text.count('приключение'),
        'indie': all_text.count('indie') + all_text.count('инди'),
    }

    popular_genres = sorted(genres_count.items(), key=lambda x: x[1], reverse=True)

    # Временная статистика
    dates = [datetime.strptime(q['timestamp'], '%Y-%m-%d %H:%M:%S').date() for q in query_history]
    date_counter = Counter([str(d) for d in dates])

    stats = {
        'total_queries': total_queries,
        'unique_users': unique_users,
        'popular_queries': popular_queries,
        'popular_genres': popular_genres,
        'queries_by_date': dict(date_counter.most_common(10)),
        'avg_queries_per_user': round(total_queries / unique_users, 2) if unique_users > 0 else 0
    }

    return stats


def train_simple_model():
    """Простое 'обучение' модели на основе истории запросов"""
    if len(query_history) == 0:
        return None

    # Анализ паттернов
    genre_weights = {}
    characteristic_weights = {}

    for query_entry in query_history:
        query = query_entry['query'].lower()

        # Обновление весов жанров
        genres = ['action', 'strategy', 'rpg', 'shooter', 'adventure', 'indie', 'simulation', 'horror']
        for genre in genres:
            if genre in query:
                genre_weights[genre] = genre_weights.get(genre, 0) + 1

        # Обновление весов характеристик
        chars = ['популярный', 'новый', 'дешевый', 'хороший', 'быстрый', 'сложный']
        for char in chars:
            if char in query:
                characteristic_weights[char] = characteristic_weights.get(char, 0) + 1

    # Нормализация весов
    total_genre = sum(genre_weights.values()) if genre_weights else 1
    total_char = sum(characteristic_weights.values()) if characteristic_weights else 1

    normalized_genres = {k: round(v / total_genre, 3) for k, v in genre_weights.items()}
    normalized_chars = {k: round(v / total_char, 3) for k, v in characteristic_weights.items()}

    model = {
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_samples': len(query_history),
        'genre_weights': normalized_genres,
        'characteristic_weights': normalized_chars,
        'most_popular_genre': max(genre_weights.items(), key=lambda x: x[1])[0] if genre_weights else None,
        'most_popular_characteristic': max(characteristic_weights.items(), key=lambda x: x[1])[0] if characteristic_weights else None
    }

    # Сохранение модели
    with open('trained_model.json', 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    return model


if __name__ == '__main__':
    # Тест функций
    print("Модуль ML готов к использованию")
