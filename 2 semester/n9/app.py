from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import os
from main import (
    extract_rating, extract_price, extract_year,
    simple_recommendation_system, search_by_range,
    get_worst_games, add_review, game_reviews, df
)

from ml_module import (
    save_query, get_personalized_recommendations,
    get_query_statistics, train_simple_model, analyze_user_preferences
)

app = Flask(__name__)


@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """API для получения рекомендаций"""
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 5)
    user_id = data.get('user_id', 'web_user')

    if not query:
        return jsonify({'error': 'Пустой запрос'}), 400

    results = simple_recommendation_system(query, top_k)

    # Сохранение запроса в историю
    save_query(user_id, query, results)

    recommendations = []
    max_score = results[0][1] if results else 100

    for name, score, game in results:
        compatibility = min(100, (score / max_score) * 100) if max_score > 0 else 0
        rating = extract_rating(game['all_reviews'])

        recommendations.append({
            'name': name,
            'genre': game['genre'],
            'developer': game['developer'],
            'publisher': game['publisher'],
            'release_date': game['release_date'],
            'price': game['original_price'],
            'rating': rating,
            'compatibility': round(compatibility, 1)
        })

    return jsonify({'recommendations': recommendations})


@app.route('/api/worst', methods=['GET'])
def worst():
    """API для получения худших игр"""
    top_k = request.args.get('top_k', 5, type=int)
    results = get_worst_games(top_k)

    worst_games = []
    for name, score, game, rating in results:
        worst_games.append({
            'name': name,
            'genre': game['genre'],
            'developer': game['developer'],
            'release_date': game['release_date'],
            'price': game['original_price'],
            'rating': rating
        })

    return jsonify({'worst_games': worst_games})


@app.route('/api/search_range', methods=['POST'])
def search_range():
    """API для поиска по диапазону"""
    data = request.json

    min_rating = data.get('min_rating', 0)
    max_rating = data.get('max_rating', 100)
    min_price = data.get('min_price', 0)
    max_price = data.get('max_price', 1000)
    min_year = data.get('min_year', 1980)
    max_year = data.get('max_year', 2030)
    top_k = data.get('top_k', 10)

    results = search_by_range(min_rating, max_rating, min_price, max_price, min_year, max_year, top_k)

    games = []
    for name, game, rating, price, year in results:
        games.append({
            'name': name,
            'genre': game['genre'],
            'rating': rating,
            'price': price,
            'year': year
        })

    return jsonify({'games': games, 'count': len(games)})


@app.route('/api/review', methods=['POST'])
def add_review_api():
    """API для добавления отзыва"""
    data = request.json
    game_name = data.get('game_name', '')
    review_text = data.get('review_text', '')
    rating = data.get('rating', 5)

    if not game_name or not review_text:
        return jsonify({'error': 'Не указано название игры или текст отзыва'}), 400

    if not (1 <= rating <= 10):
        return jsonify({'error': 'Рейтинг должен быть от 1 до 10'}), 400

    add_review(game_name, review_text, rating)
    return jsonify({'success': True, 'message': 'Отзыв добавлен'})


@app.route('/api/reviews/<game_name>', methods=['GET'])
def get_reviews(game_name):
    """API для получения отзывов игры"""
    if game_name not in game_reviews or len(game_reviews[game_name]) == 0:
        return jsonify({'reviews': [], 'count': 0})

    return jsonify({
        'reviews': game_reviews[game_name],
        'count': len(game_reviews[game_name])
    })


@app.route('/api/statistics', methods=['GET'])
def statistics():
    """API для получения статистики"""
    if len(game_reviews) == 0:
        return jsonify({'statistics': [], 'count': 0})

    stats = []
    for game_name, reviews in game_reviews.items():
        avg_rating = sum(r['rating'] for r in reviews) / len(reviews)
        stats.append({
            'game': game_name,
            'reviews_count': len(reviews),
            'avg_rating': round(avg_rating, 2),
            'latest_review': reviews[-1]['date']
        })

    stats.sort(key=lambda x: x['reviews_count'], reverse=True)

    return jsonify({'statistics': stats, 'count': len(stats)})


@app.route('/api/personal', methods=['POST'])
def personal_recommendations():
    """API для персональных рекомендаций"""
    data = request.json
    user_id = data.get('user_id', 'web_user')
    top_k = data.get('top_k', 5)

    results = get_personalized_recommendations(user_id, df, top_k)

    if results is None:
        return jsonify({'error': 'Нет истории запросов', 'recommendations': []}), 200

    recommendations = []
    max_score = results[0][1] if results else 100

    for name, score, game in results:
        compatibility = min(100, (score / max_score) * 100) if max_score > 0 else 0
        rating = extract_rating(game['all_reviews'])

        recommendations.append({
            'name': name,
            'genre': game['genre'],
            'developer': game['developer'],
            'release_date': game['release_date'],
            'price': game['original_price'],
            'rating': rating,
            'compatibility': round(compatibility, 1)
        })

    return jsonify({'recommendations': recommendations})


@app.route('/api/profile', methods=['POST'])
def user_profile():
    """API для получения профиля пользователя"""
    data = request.json
    user_id = data.get('user_id', 'web_user')

    preferences = analyze_user_preferences(user_id)

    if preferences is None:
        return jsonify({'error': 'Нет истории запросов'}), 200

    return jsonify({'profile': preferences})


@app.route('/api/query_statistics', methods=['GET'])
def query_statistics():
    """API для статистики запросов"""
    stats = get_query_statistics()

    if stats is None:
        return jsonify({'statistics': None, 'message': 'Нет данных'}), 200

    return jsonify({'statistics': stats})


@app.route('/api/train', methods=['POST'])
def train():
    """API для обучения модели"""
    model = train_simple_model()

    if model is None:
        return jsonify({'error': 'Недостаточно данных для обучения'}), 400

    return jsonify({'success': True, 'model': model})


if __name__ == '__main__':
    # Создаем папку для шаблонов если её нет
    if not os.path.exists('templates'):
        os.makedirs('templates')

    print("\n" + "="*60)
    print("ВЕБ-ИНТЕРФЕЙС СИСТЕМЫ РЕКОМЕНДАЦИЙ ИГР")
    print("="*60)
    print("\nСервер запущен на: http://127.0.0.1:5000")
    print("Нажмите Ctrl+C для остановки")
    print("="*60 + "\n")

    app.run(debug=True, host='127.0.0.1', port=5000)
