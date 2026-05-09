from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import uuid
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from database import Database

app = Flask(__name__)
app.secret_key = 'ml-reviews-secret'

db = Database()

vectorizer = None
models = {}
products_list = ["Смартфон", "Ноутбук", "Наушники", "Часы", "Планшет", "Телевизор", "Камера", "Колонки"]


def get_or_create_user_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        db.register_user(session['user_id'])
    else:
        db.register_user(session['user_id'])
    return session['user_id']


def train_sentiment_models():
    global vectorizer, models

    df = db.get_all_reviews()

    if len(df) < 10:
        return None

    df_binary = df[df['sentiment'].notna()].copy()

    if len(df_binary) < 10:
        return None

    X = df_binary['text'].values
    y = (df_binary['sentiment'] == 'позитивный').astype(int).values

    vectorizer = TfidfVectorizer(max_features=100)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    models_to_train = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }

    results = []

    for name, model in models_to_train.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0)
        }
        results.append(metrics)
        models[name] = model

    return pd.DataFrame(results)


def predict_sentiment(text, model_name='LogisticRegression'):
    if vectorizer is None or model_name not in models:
        return None, 0

    X_new = vectorizer.transform([text])
    model = models[model_name]
    pred = model.predict(X_new)[0]
    pred_int = int(pred)
    proba = model.predict_proba(X_new)[0][pred_int]

    result = "позитивный" if pred_int == 1 else "негативный"
    confidence = proba * 100

    return result, confidence


def get_collaborative_recommendations(user_id, top_n=5):
    """Персональные рекомендации на основе collaborative filtering"""
    all_ratings = db.get_all_user_ratings()

    if len(all_ratings) < 5:
        return []

    user_item_matrix = all_ratings.pivot_table(
        index='user_id',
        columns='product',
        values='rating',
        fill_value=0
    )

    if user_id not in user_item_matrix.index:
        return []

    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]

    user_ratings = db.get_user_ratings(user_id)
    rated_products = set(user_ratings['product'].values)

    recommendations = {}

    for similar_user_id, similarity_score in similar_users.items():
        similar_user_ratings = db.get_user_ratings(similar_user_id)

        for _, row in similar_user_ratings.iterrows():
            product = row['product']
            rating = row['rating']

            if product not in rated_products and rating >= 4:
                if product not in recommendations:
                    recommendations[product] = 0
                recommendations[product] += similarity_score * rating

    sorted_recommendations = sorted(
        recommendations.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return [{'product': prod, 'score': score} for prod, score in sorted_recommendations]


def generate_wordcloud(text_data):
    if not text_data:
        return None

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis'
    ).generate(' '.join(text_data))

    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)

    return base64.b64encode(img.getvalue()).decode()


@app.route('/')
def index():
    user_id = get_or_create_user_id()

    recommendations = get_collaborative_recommendations(user_id, top_n=5)
    recent_reviews = db.get_all_reviews().head(10)

    total_reviews = db.get_review_count()
    total_users = db.get_user_count()

    return render_template(
        'index.html',
        products=products_list,
        recommendations=recommendations,
        recent_reviews=recent_reviews.to_dict('records') if len(recent_reviews) > 0 else [],
        total_reviews=total_reviews,
        total_users=total_users,
        user_id=user_id[:8]
    )


@app.route('/submit_review', methods=['POST'])
def submit_review():
    user_id = get_or_create_user_id()

    product = request.form.get('product')
    rating = int(request.form.get('rating'))
    text = request.form.get('text')

    sentiment = None
    confidence = None

    if models:
        sentiment, confidence = predict_sentiment(text)

    db.add_review(user_id, product, rating, text, sentiment, confidence)

    return redirect(url_for('index'))


@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text', '')

    if not text or not models:
        return jsonify({'error': 'No text provided or models not trained'})

    sentiment, confidence = predict_sentiment(text)

    return jsonify({
        'sentiment': sentiment,
        'confidence': round(confidence, 2)
    })


@app.route('/statistics')
def statistics():
    df = db.get_all_reviews()

    if len(df) == 0:
        return render_template('statistics.html', no_data=True)

    product_stats = []
    for product in db.get_all_products():
        stats = db.get_product_stats(product)
        if stats:
            stats['product'] = product
            product_stats.append(stats)

    rating_dist_img = None
    if len(df) > 0:
        img = io.BytesIO()
        plt.figure(figsize=(8, 5))
        plt.hist(df['rating'], bins=5, edgecolor='black', color='royalblue', alpha=0.7)
        plt.title('Распределение оценок', fontsize=14, fontweight='bold')
        plt.xlabel('Рейтинг')
        plt.ylabel('Количество')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()
        img.seek(0)
        rating_dist_img = base64.b64encode(img.getvalue()).decode()

    reviews_time_img = None
    reviews_by_date = db.get_reviews_by_date()
    if len(reviews_by_date) > 0:
        img = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.plot(pd.to_datetime(reviews_by_date['date']), reviews_by_date['count'], 'b-o', linewidth=2)
        plt.title('Динамика отзывов', fontsize=14, fontweight='bold')
        plt.xlabel('Дата')
        plt.ylabel('Количество отзывов')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()
        img.seek(0)
        reviews_time_img = base64.b64encode(img.getvalue()).decode()

    wordcloud_img = None
    if len(df) > 0:
        all_text = df['text'].values
        wordcloud_img = generate_wordcloud(all_text)

    return render_template(
        'statistics.html',
        product_stats=product_stats,
        rating_dist_img=rating_dist_img,
        reviews_time_img=reviews_time_img,
        wordcloud_img=wordcloud_img,
        no_data=False
    )


@app.route('/models')
def models_page():
    results_df = train_sentiment_models()

    if results_df is None:
        return render_template('models.html', no_data=True)

    results = results_df.to_dict('records')
    best_model = results_df.loc[results_df['F1'].idxmax(), 'Model']

    comparison_img = None
    img = io.BytesIO()
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results_df))
    width = 0.2
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b3']

    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, results_df[metric], width, label=metric, color=colors[i], alpha=0.8)

    ax.set_xlabel('Модели', fontweight='bold')
    ax.set_ylabel('Значение метрики', fontweight='bold')
    ax.set_title('Сравнение моделей классификации', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df['Model'])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    comparison_img = base64.b64encode(img.getvalue()).decode()

    return render_template(
        'models.html',
        results=results,
        best_model=best_model,
        comparison_img=comparison_img,
        no_data=False
    )


@app.route('/product/<product_name>')
def product_detail(product_name):
    stats = db.get_product_stats(product_name)
    reviews = db.get_reviews_by_product(product_name)

    return render_template(
        'product_detail.html',
        product=product_name,
        stats=stats,
        reviews=reviews.to_dict('records') if len(reviews) > 0 else []
    )


if __name__ == '__main__':
    print("="*60)
    print("ML-ПРОДУКТ: ВЕБ-ИНТЕРФЕЙС АНАЛИЗА ОТЗЫВОВ")
    print("="*60)

    review_count = db.get_review_count()
    if review_count == 0:
        print("\nВНИМАНИЕ: База данных пуста!")
        print("Запустите: python generate_data.py --clear")
        print("="*60)
        exit(1)

    print(f"\nБаза данных содержит {review_count} отзывов")

    print("\nОбучение моделей машинного обучения...")
    results = train_sentiment_models()
    if results is not None:
        print(f"[OK] Модели обучены успешно")
    else:
        print("[ВНИМАНИЕ] Недостаточно данных для обучения моделей")

    print("\nЗапуск Flask приложения...")
    print("Откройте браузер: http://127.0.0.1:5000")
    print("\nДля остановки нажмите Ctrl+C")
    print("="*60)

    app.run(debug=True, host='0.0.0.0', port=5000)
