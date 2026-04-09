import pandas as pd
import random
import re
import json
import os
from datetime import datetime

# Импорт ML модуля и звуков
from ml_module import (
    save_query, get_personalized_recommendations,
    get_query_statistics, train_simple_model, analyze_user_preferences
)
from sound_module import play_sound_async, play_recommendation_sound, play_success_sound, play_melody

# Загрузка датасета
df = pd.read_csv('steam_games.csv')
print(f"Загружено игр: {len(df)}")

# Воспроизведение мелодии при запуске
play_sound_async(play_melody)

# Очистка данных
df['genre'] = df['genre'].fillna('')
df['popular_tags'] = df['popular_tags'].fillna('')
df['all_reviews'] = df['all_reviews'].fillna('')
df['original_price'] = df['original_price'].fillna('0')
df['release_date'] = df['release_date'].fillna('')

# Файл для хранения отзывов
REVIEWS_FILE = 'game_reviews.json'

# Загрузка отзывов
if os.path.exists(REVIEWS_FILE):
    with open(REVIEWS_FILE, 'r', encoding='utf-8') as f:
        game_reviews = json.load(f)
else:
    game_reviews = {}


def extract_rating(reviews_text):
    """Извлекает процент положительных отзывов из текста"""
    if pd.isna(reviews_text) or reviews_text == '':
        return 50
    match = re.search(r'(\d+)%', str(reviews_text))
    if match:
        return int(match.group(1))
    return 50


def extract_price(price_text):
    """Извлекает числовую цену из текста"""
    if pd.isna(price_text) or price_text == '':
        return 0

    price_str = str(price_text).lower()

    # Бесплатные игры
    if 'free' in price_str or price_str == '0':
        return 0

    # Ищем число с долларом или просто число
    match = re.search(r'\$?\s*([\d]+\.?\d*)', price_str)
    if match:
        try:
            price = float(match.group(1))
            # Если цена больше 1000, это скорее всего ошибка формата (например 1.020 вместо $1.02)
            if price > 200:
                return 0
            return price
        except ValueError:
            return 0
    return 0


def extract_year(date_text):
    """Извлекает год из даты релиза"""
    if pd.isna(date_text) or date_text == '':
        return 2000
    match = re.search(r'\d{4}', str(date_text))
    if match:
        return int(match.group(0))
    return 2000


def search_by_name(name_query):
    """Поиск игры по имени"""
    name_query = name_query.lower()
    results = df[df['name'].str.lower().str.contains(name_query, na=False)]

    if len(results) == 0:
        print(f"Игры с названием '{name_query}' не найдены")
        return

    print(f"\nНайдено игр: {len(results)}")
    for idx, row in results.head(5).iterrows():
        print(f"\n{'='*60}")
        print(f"🎮 {row['name'].upper()}")
        print(f"{'='*60}")

        # Основная информация
        print(f"📊 Жанр: {row['genre']}")
        print(f"🏢 Разработчик: {row['developer']}")
        print(f"📢 Издатель: {row['publisher']}")

        # Дата и цена
        print(f"📅 Дата выхода: {row['release_date']}")
        year = extract_year(row['release_date'])
        print(f"🗓️  Год: {year}")

        # Цена
        price = extract_price(row['original_price'])
        if price == 0:
            print(f"💰 Цена: Бесплатно")
        else:
            print(f"💰 Цена: ${price:.2f}")

        # Рейтинг
        rating = extract_rating(row['all_reviews'])
        if rating >= 90:
            rating_emoji = "⭐⭐⭐⭐⭐"
        elif rating >= 80:
            rating_emoji = "⭐⭐⭐⭐"
        elif rating >= 70:
            rating_emoji = "⭐⭐⭐"
        elif rating >= 60:
            rating_emoji = "⭐⭐"
        else:
            rating_emoji = "⭐"
        print(f"⭐ Рейтинг: {rating}% {rating_emoji}")

        # Теги
        if pd.notna(row['popular_tags']):
            tags = str(row['popular_tags']).split(',')[:5]
            print(f"🏷️  Теги: {', '.join(tags)}")

        # Описание (если есть)
        if pd.notna(row['desc_snippet']):
            desc = str(row['desc_snippet'])[:150]
            print(f"📝 Описание: {desc}...")

        print(f"{'='*60}")


def get_worst_games(top_k: int = 5):
    """Возвращает топ слабых игр с низким рейтингом"""
    scores = []

    for idx, row in df.iterrows():
        rating = extract_rating(row['all_reviews'])
        # Инвертируем рейтинг - чем ниже, тем выше в топе
        score = 100 - rating
        scores.append((row['name'], score, row, rating))

    # Сортировка по убыванию (худшие первыми)
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def add_review(game_name: str, review_text: str, rating: int):
    """Добавляет отзыв на игру"""
    if game_name not in game_reviews:
        game_reviews[game_name] = []

    review = {
        'text': review_text,
        'rating': rating,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    game_reviews[game_name].append(review)

    # Сохранение в файл
    with open(REVIEWS_FILE, 'w', encoding='utf-8') as f:
        json.dump(game_reviews, f, ensure_ascii=False, indent=2)

    print(f"\nОтзыв добавлен для игры '{game_name}'")


def show_reviews(game_name: str):
    """Показывает отзывы для игры"""
    if game_name not in game_reviews or len(game_reviews[game_name]) == 0:
        print(f"\nДля игры '{game_name}' пока нет отзывов")
        return

    print(f"\n--- ОТЗЫВЫ ДЛЯ '{game_name}' ---")
    for i, review in enumerate(game_reviews[game_name], 1):
        print(f"\nОтзыв #{i} ({review['date']})")
        print(f"Оценка: {review['rating']}/10")
        print(f"Текст: {review['text']}")


def export_statistics():
    """Выгружает статистику по играм с отзывами"""
    if len(game_reviews) == 0:
        print("\nНет отзывов для выгрузки статистики")
        return

    stats = []
    for game_name, reviews in game_reviews.items():
        avg_rating = sum(r['rating'] for r in reviews) / len(reviews)
        stats.append({
            'game': game_name,
            'reviews_count': len(reviews),
            'avg_rating': round(avg_rating, 2),
            'latest_review': reviews[-1]['date']
        })

    # Сортировка по количеству отзывов
    stats.sort(key=lambda x: x['reviews_count'], reverse=True)

    # Сохранение в файл
    stats_file = 'game_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n--- СТАТИСТИКА ПО ИГРАМ ---")
    print(f"Всего игр с отзывами: {len(stats)}")
    print(f"\nТоп-5 игр по количеству отзывов:")
    for i, stat in enumerate(stats[:5], 1):
        print(f"{i}. {stat['game']}")
        print(f"   Отзывов: {stat['reviews_count']}, Средняя оценка: {stat['avg_rating']}/10")

    print(f"\nСтатистика сохранена в файл: {stats_file}")


def search_by_range(min_rating: int = 0, max_rating: int = 100,
                    min_price: float = 0, max_price: float = 1000,
                    min_year: int = 1980, max_year: int = 2030,
                    top_k: int = 10):
    """Поиск игр по диапазону характеристик"""
    results = []

    for idx, row in df.iterrows():
        rating = extract_rating(row['all_reviews'])
        price = extract_price(row['original_price'])
        year = extract_year(row['release_date'])

        # Проверка диапазонов
        if (min_rating <= rating <= max_rating and
            min_price <= price <= max_price and
            min_year <= year <= max_year):
            results.append((row['name'], row, rating, price, year))

    # Сортировка по рейтингу
    results.sort(key=lambda x: x[2], reverse=True)

    return results[:top_k]


def simple_recommendation_system(query: str, top_k: int = 5):
    """Система рекомендаций на основе текстового запроса"""
    query = query.lower()
    scores = []

    # Если запрос слишком короткий или общий - случайная выборка с учетом рейтинга
    if len(query) < 3 or query in ['игра', 'выбери', 'что-то']:
        for idx, row in df.iterrows():
            rating = extract_rating(row['all_reviews'])
            year = extract_year(row['release_date'])

            # Базовый скор на основе рейтинга и новизны
            score = rating * 0.3 + (year - 2000) * 0.5 + random.uniform(0, 10)
            scores.append((row['name'], score, row))
    else:
        for idx, row in df.iterrows():
            score = 0
            genre_text = str(row['genre']).lower()
            tags_text = str(row['popular_tags']).lower()

            # Проверка жанров (+40-50 баллов)
            if any(word in query for word in ['action', 'экшен', 'боевик']):
                if 'action' in genre_text or 'action' in tags_text:
                    score += 45

            if any(word in query for word in ['strategy', 'стратегия', 'стратегический']):
                if 'strategy' in genre_text or 'strategy' in tags_text:
                    score += 45

            if any(word in query for word in ['rpg', 'рпг', 'ролевая']):
                if 'rpg' in genre_text or 'rpg' in tags_text:
                    score += 45

            if any(word in query for word in ['shooter', 'шутер', 'стрелялка']):
                if 'shooter' in tags_text or 'fps' in tags_text:
                    score += 50

            if any(word in query for word in ['adventure', 'приключение', 'приключенческая']):
                if 'adventure' in genre_text or 'adventure' in tags_text:
                    score += 40

            if any(word in query for word in ['indie', 'инди', 'независимая']):
                if 'indie' in genre_text or 'indie' in tags_text:
                    score += 40

            if any(word in query for word in ['simulation', 'симулятор', 'симуляция']):
                if 'simulation' in genre_text or 'simulation' in tags_text:
                    score += 40

            if any(word in query for word in ['horror', 'хоррор', 'ужасы']):
                if 'horror' in genre_text or 'horror' in tags_text:
                    score += 50

            # Проверка ключевых слов (характеристики)
            rating = extract_rating(row['all_reviews'])
            price = extract_price(row['original_price'])
            year = extract_year(row['release_date'])

            if any(word in query for word in ['популярный', 'popular', 'известный', 'топ']):
                # Чем выше рейтинг, тем больше баллов
                if rating >= 90:
                    score += 40
                elif rating >= 80:
                    score += 30
                elif rating >= 70:
                    score += 20

            if any(word in query for word in ['новый', 'new', 'свежий', 'недавний']):
                if year >= 2019:
                    score += 30
                elif year >= 2017:
                    score += 15

            if any(word in query for word in ['дешевый', 'cheap', 'бесплатный', 'free']):
                if price == 0:
                    score += 40
                elif price < 10:
                    score += 20

            if any(word in query for word in ['дорогой', 'expensive', 'ааа', 'aaa']):
                if price > 30:
                    score += 25

            if any(word in query for word in ['хороший', 'good', 'качественный', 'отличный']):
                if rating >= 90:
                    score += 35
                elif rating >= 80:
                    score += 25
                elif rating >= 70:
                    score += 15

            if any(word in query for word in ['быстрый', 'fast', 'динамичный']):
                if 'fast' in tags_text or 'action' in genre_text or 'fps' in tags_text:
                    score += 20

            if any(word in query for word in ['сложный', 'hard', 'difficult', 'хардкорный']):
                if 'difficult' in tags_text or 'souls' in tags_text or 'hard' in tags_text:
                    score += 25

            # Если нет совпадений по жанру/ключевым словам - минимальный скор на основе рейтинга
            if score == 0:
                score = rating * 0.1 + random.uniform(0, 3)

            # Добавляем небольшую случайность для разнообразия
            score += random.uniform(0, 5)
            scores.append((row['name'], score, row))

    # Сортировка по убыванию
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def main():
    """Главный цикл программы"""
    print("\n" + "="*60)
    print("СИСТЕМА РЕКОМЕНДАЦИЙ ИГР STEAM")
    print("="*60)
    print("\nЧто я понимаю:")
    print("  • Жанры: action, strategy, rpg, shooter, adventure, indie, simulation, horror")
    print("  • Характеристики: популярный, новый, дешевый, хороший, быстрый, сложный")
    print("  • Примеры запросов:")
    print("    - 'популярный shooter'")
    print("    - 'новая rpg'")
    print("    - 'дешевый indie'")
    print("    - 'хороший action'")
    print("\nКоманды:")
    print("  • 'случайный' - случайная игра")
    print("  • 'поиск <название>' - поиск по имени")
    print("  • 'худшие' - топ-5 худших игр")
    print("  • 'отзыв <название>' - добавить отзыв на игру")
    print("  • 'показать отзывы <название>' - показать отзывы игры")
    print("  • 'статистика' - выгрузить статистику по отзывам")
    print("  • 'диапазон' - поиск по диапазону характеристик")
    print("  • 'персональные' - рекомендации на основе вашей истории")
    print("  • 'мой профиль' - ваши предпочтения")
    print("  • 'статистика запросов' - общая статистика всех запросов")
    print("  • 'обучить' - обучить модель на всех запросах")
    print("\nМашинное обучение (sklearn):")
    print("  • 'ml регрессия' - обучить модель предсказания рейтинга")
    print("  • 'ml кластеризация' - кластерный анализ игр")
    print("  • 'выход' - завершить программу")
    print("="*60)

    # ID пользователя (для консольной версии используем 'console_user')
    user_id = 'console_user'

    while True:
        query = input('\n>>> Ваш запрос: ').strip()

        if query.lower() in ['выход', 'exit', 'quit', 'q']:
            print('Завершение программы...')
            break

        # Случайная игра
        if query.lower() in ['случайный', 'рандомный', 'random', 'r']:
            print('\n--- СЛУЧАЙНАЯ ИГРА ---')
            random_game = df.sample(1).iloc[0]
            print(f"\n{random_game['name'].upper()}")
            print(f"Жанр: {random_game['genre']}")
            print(f"Разработчик: {random_game['developer']}")
            print(f"Издатель: {random_game['publisher']}")
            print(f"Дата выхода: {random_game['release_date']}")
            print(f"Цена: {random_game['original_price']}")
            rating = extract_rating(random_game['all_reviews'])
            print(f"Рейтинг: {rating}%")
            continue

        # Поиск по имени
        if query.lower().startswith('поиск ') or query.lower().startswith('search '):
            name_query = query.split(' ', 1)[1] if ' ' in query else ''
            if name_query:
                search_by_name(name_query)
            else:
                print("Укажите название игры для поиска")
            continue

        # Худшие игры
        if query.lower() in ['худшие', 'worst', 'слабые']:
            print('\n--- ТОП-5 ХУДШИХ ИГР ---')
            results = get_worst_games(5)
            for i, (name, score, game, rating) in enumerate(results, 1):
                print(f"\n{i}. {name.upper()}")
                print(f"   Жанр: {game['genre']}")
                print(f"   Разработчик: {game['developer']}")
                print(f"   Дата выхода: {game['release_date']}")
                print(f"   Цена: {game['original_price']}")
                print(f"   Рейтинг: {rating}%")
            continue

        # Добавить отзыв
        if query.lower().startswith('отзыв ') or query.lower().startswith('review '):
            game_name = query.split(' ', 1)[1] if ' ' in query else ''
            if game_name:
                review_text = input('Введите текст отзыва: ').strip()
                try:
                    review_rating = int(input('Введите оценку (1-10): ').strip())
                    if 1 <= review_rating <= 10:
                        add_review(game_name, review_text, review_rating)
                    else:
                        print('Оценка должна быть от 1 до 10')
                except ValueError:
                    print('Неверный формат оценки')
            else:
                print("Укажите название игры")
            continue

        # Показать отзывы
        if query.lower().startswith('показать отзывы ') or query.lower().startswith('show reviews '):
            game_name = query.split(' ', 2)[2] if len(query.split(' ')) > 2 else ''
            if game_name:
                show_reviews(game_name)
            else:
                print("Укажите название игры")
            continue

        # Статистика
        if query.lower() in ['статистика', 'stats', 'statistics']:
            export_statistics()
            continue

        # Поиск по диапазону
        if query.lower() in ['диапазон', 'range', 'фильтр']:
            print('\n--- ПОИСК ПО ДИАПАЗОНУ ---')
            try:
                min_rating = int(input('Минимальный рейтинг (0-100): ').strip() or '0')
                max_rating = int(input('Максимальный рейтинг (0-100): ').strip() or '100')
                min_price = float(input('Минимальная цена ($): ').strip() or '0')
                max_price = float(input('Максимальная цена ($): ').strip() or '1000')
                min_year = int(input('Минимальный год (1980-2030): ').strip() or '1980')
                max_year = int(input('Максимальный год (1980-2030): ').strip() or '2030')

                results = search_by_range(min_rating, max_rating, min_price, max_price, min_year, max_year, 10)

                if len(results) == 0:
                    print('\nИгры не найдены в указанном диапазоне')
                else:
                    print(f'\n--- НАЙДЕНО ИГР: {len(results)} ---')
                    for i, (name, game, rating, price, year) in enumerate(results, 1):
                        print(f"\n{i}. {name}")
                        print(f"   Жанр: {game['genre']}")
                        print(f"   Рейтинг: {rating}%")
                        print(f"   Цена: ${price}")
                        print(f"   Год: {year}")
            except ValueError:
                print('Неверный формат ввода')
            continue

        # Персональные рекомендации
        if query.lower() in ['персональные', 'personal', 'для меня']:
            print('\n--- ПЕРСОНАЛЬНЫЕ РЕКОМЕНДАЦИИ ---')
            results = get_personalized_recommendations(user_id, df, 5)

            if results is None:
                print('У вас пока нет истории запросов.')
                print('Сделайте несколько запросов, чтобы я мог изучить ваши предпочтения!')
            else:
                play_sound_async(play_recommendation_sound)
                max_score = results[0][1] if results else 100

                for i, (name, score, game) in enumerate(results, 1):
                    compatibility = min(100, (score / max_score) * 100) if max_score > 0 else 0
                    rating = extract_rating(game['all_reviews'])

                    print(f"\n{i}. {name.upper()}")
                    print(f"   Жанр: {game['genre']}")
                    print(f"   Разработчик: {game['developer']}")
                    print(f"   Дата выхода: {game['release_date']}")
                    print(f"   Цена: {game['original_price']}")
                    print(f"   Рейтинг: {rating}%")
                    print(f"   Совместимость: {compatibility:.1f}%")
            continue

        # Мой профиль
        if query.lower() in ['мой профиль', 'профиль', 'myprofile', 'profile']:
            print('\n--- ВАШ ПРОФИЛЬ ---')
            preferences = analyze_user_preferences(user_id)

            if preferences is None:
                print('У вас пока нет истории запросов.')
            else:
                print(f"\nВсего запросов: {preferences['total_queries']}")

                print("\nЛюбимые жанры:")
                for genre, count in preferences['favorite_genres']:
                    if count > 0:
                        print(f"  • {genre}: {count} раз")

                print("\nЛюбимые характеристики:")
                for char, count in preferences['favorite_characteristics']:
                    if count > 0:
                        print(f"  • {char}: {count} раз")

                print(f"\nПоследний запрос: {preferences['last_query']}")
                print(f"Время: {preferences['last_query_time']}")
            continue

        # Статистика запросов
        if query.lower() in ['статистика запросов', 'query stats', 'stats queries']:
            print('\n--- СТАТИСТИКА ЗАПРОСОВ ---')
            stats = get_query_statistics()

            if stats is None:
                print('Пока нет данных для статистики.')
            else:
                print(f"\nВсего запросов: {stats['total_queries']}")
                print(f"Уникальных пользователей: {stats['unique_users']}")
                print(f"Среднее запросов на пользователя: {stats['avg_queries_per_user']}")

                print("\nПопулярные запросы:")
                for query_text, count in stats['popular_queries'][:5]:
                    print(f"  • {query_text}: {count} раз")

                print("\nПопулярные жанры:")
                for genre, count in stats['popular_genres'][:5]:
                    if count > 0:
                        print(f"  • {genre}: {count} упоминаний")
            continue

        # Обучить модель
        if query.lower() in ['обучить', 'train', 'обучение']:
            print('\n--- ОБУЧЕНИЕ МОДЕЛИ ---')
            model = train_simple_model()

            if model is None:
                print('Недостаточно данных для обучения.')
            else:
                play_sound_async(play_success_sound)
                print(f"\nМодель обучена!")
                print(f"Обучающих примеров: {model['training_samples']}")
                print(f"Время обучения: {model['trained_at']}")

                print(f"\nСамый популярный жанр: {model['most_popular_genre'] or 'N/A'}")
                print(f"Самая популярная характеристика: {model['most_popular_characteristic'] or 'N/A'}")

                print("\nВеса жанров:")
                for genre, weight in sorted(model['genre_weights'].items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  • {genre}: {weight}")

                print("\nМодель сохранена в файл: trained_model.json")
            continue

        # ML Регрессия (sklearn)
        if query.lower() in ['ml регрессия', 'ml regression', 'регрессия', 'regression']:
            print('\n--- МАШИННОЕ ОБУЧЕНИЕ: ЛИНЕЙНАЯ РЕГРЕССИЯ ---')
            try:
                from sklearn_ml import train_price_prediction_model
                model_info = train_price_prediction_model(df)
                if model_info:
                    play_sound_async(play_success_sound)
            except Exception as e:
                print(f"Ошибка: {e}")
            continue

        # ML Кластеризация (sklearn)
        if query.lower() in ['ml кластеризация', 'ml clustering', 'кластеризация', 'clustering']:
            print('\n--- МАШИННОЕ ОБУЧЕНИЕ: КЛАСТЕРНЫЙ АНАЛИЗ ---')
            try:
                from sklearn_ml import perform_clustering_analysis
                clustering_info = perform_clustering_analysis(df)
                if clustering_info:
                    play_sound_async(play_success_sound)
            except Exception as e:
                print(f"Ошибка: {e}")
            continue

        # Рекомендации
        print('\n--- РЕКОМЕНДАЦИИ ---')
        results = simple_recommendation_system(query)

        # Воспроизведение звука при выдаче рекомендаций
        play_sound_async(play_recommendation_sound)

        # Сохранение запроса в историю
        save_query(user_id, query, results)

        max_score = results[0][1] if results else 100

        for i, (name, score, game) in enumerate(results, 1):
            compatibility = min(100, (score / max_score) * 100) if max_score > 0 else 0

            print(f"\n{i}. {name.upper()}")
            print(f"   Жанр: {game['genre']}")
            print(f"   Разработчик: {game['developer']}")
            print(f"   Издатель: {game['publisher']}")
            print(f"   Дата выхода: {game['release_date']}")
            print(f"   Цена: {game['original_price']}")

            rating = extract_rating(game['all_reviews'])
            print(f"   Рейтинг: {rating}%")
            print(f"   Совместимость с запросом: {compatibility:.1f}%")


if __name__ == '__main__':
    main()
