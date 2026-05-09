"""
Скрипт для генерации начальных данных в базу SQLite

Использование:
    python generate_data.py           # Спросит перед очисткой
    python generate_data.py --clear   # Автоматически очистит базу
"""
import numpy as np
import sys
from datetime import datetime, timedelta
from database import Database

def generate_initial_reviews(count=200, auto_clear=False):
    db = Database()

    existing_count = db.get_review_count()

    if existing_count > 0:
        print(f"База данных уже содержит {existing_count} отзывов.")
        if auto_clear:
            print("Автоматическая очистка базы данных...")
            db.clear_all_data()
            print("База данных очищена.")
        else:
            print("Используйте флаг --clear для очистки базы данных")
            print("Пример: python generate_data.py --clear")
            return

    print(f"\nГенерация {count} отзывов...")

    positive_reviews = [
        "отлично", "замечательно", "супер", "рекомендую", "лучший",
        "качество отличное", "доволен", "прекрасно", "великолепно", "нравится",
        "превосходно", "идеально", "восхитительно", "потрясающе", "шикарно"
    ]

    negative_reviews = [
        "ужасно", "плохо", "разочарован", "не рекомендую", "брак",
        "качество плохое", "ужасное качество", "не работает", "деньги на ветер", "кошмар",
        "отвратительно", "ужас", "обман", "подделка", "развалилось"
    ]

    neutral_reviews = [
        "нормальный товар", "средне", "так себе", "ничего особенного",
        "обычный товар", "приемлемо", "на троечку"
    ]

    products = [
        "Смартфон", "Ноутбук", "Наушники", "Часы", "Планшет",
        "Телевизор", "Камера", "Колонки"
    ]

    for i in range(count):
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])

        if rating >= 4:
            text = np.random.choice(positive_reviews)
            sentiment = "позитивный"
            confidence = np.random.uniform(85, 99)
        elif rating <= 2:
            text = np.random.choice(negative_reviews)
            sentiment = "негативный"
            confidence = np.random.uniform(85, 99)
        else:
            text = np.random.choice(neutral_reviews)
            sentiment = None
            confidence = None

        user_id = f"user_{np.random.randint(1, 31)}"
        product = np.random.choice(products)

        db.add_review(user_id, product, rating, text, sentiment, confidence)

        if (i + 1) % 50 == 0:
            print(f"  Создано {i + 1}/{count} отзывов...")

    print("\nОбновление дат отзывов...")
    import sqlite3
    conn = sqlite3.connect('reviews.db')
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM reviews ORDER BY id")
    review_ids = [row[0] for row in cursor.fetchall()]

    for review_id in review_ids:
        days_ago = np.random.randint(0, 60)
        random_date = datetime.now() - timedelta(days=days_ago)
        cursor.execute("UPDATE reviews SET date = ? WHERE id = ?",
                      (random_date.strftime('%Y-%m-%d %H:%M:%S'), review_id))

    conn.commit()
    conn.close()

    print(f"\n[OK] Успешно создано {count} отзывов в базе данных!")

    print("\n" + "="*60)
    print("СТАТИСТИКА")
    print("="*60)

    all_reviews = db.get_all_reviews()
    print(f"Всего отзывов: {len(all_reviews)}")
    print(f"Уникальных пользователей: {all_reviews['user_id'].nunique()}")
    print(f"Уникальных товаров: {all_reviews['product'].nunique()}")

    print(f"\nРаспределение по рейтингам:")
    rating_counts = all_reviews['rating'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        print(f"  {rating} звезд: {count} отзывов")

    print(f"\nРаспределение по тональности:")
    sentiment_counts = all_reviews['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        if sentiment:
            print(f"  {sentiment}: {count} отзывов")

    print("\n" + "="*60)
    print("Теперь можете запустить main.py или app.py")
    print("="*60)


if __name__ == '__main__':
    auto_clear = '--clear' in sys.argv
    generate_initial_reviews(200, auto_clear=auto_clear)
