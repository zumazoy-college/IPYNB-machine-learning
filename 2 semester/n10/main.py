import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re

from database import Database

vectorizer = None
models = {}
df = None
df_binary = None
product_texts = None
similarity_matrix = None
daily_counts = None
ts_model = None
db = Database()


def load_data():
    global df

    print("\nЗагружаем данные из базы...")

    review_count = db.get_review_count()

    if review_count == 0:
        print("ОШИБКА: База данных пуста!")
        print("Запустите: python generate_data.py --clear")
        return None

    df = db.get_all_reviews()
    df['date'] = pd.to_datetime(df['date'])

    # Конвертация bytes в int для rating
    if df['rating'].dtype == object:
        def safe_int_convert(x):
            if isinstance(x, bytes):
                return int.from_bytes(x[:1], byteorder='little')
            return int(x)
        df['rating'] = df['rating'].apply(safe_int_convert)
    else:
        df['rating'] = df['rating'].astype(int)

    df['sentiment_binary'] = df['sentiment'].map({'позитивный': 1, 'негативный': 0})

    print(f"[OK] Загружено {len(df)} записей из базы данных")
    return df


def train_models():
    global vectorizer, models, df_binary

    print("\n" + "="*60)
    print("КЛАССИФИКАЦИЯ ТЕКСТОВ - СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*60)

    df_binary = df[df['sentiment_binary'].notna()].copy()
    X = df_binary['text'].values
    y = df_binary['sentiment_binary'].values

    print(f"Обучающих примеров: {len(X)} (позитив: {sum(y)}, негатив: {len(y)-sum(y)})")

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
        print(f"\nОбучение {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred)
        }
        results.append(metrics)
        models[name] = model

        print(f"   Accuracy:  {metrics['Accuracy']:.2%}")
        print(f"   Precision: {metrics['Precision']:.2%}")
        print(f"   Recall:    {metrics['Recall']:.2%}")
        print(f"   F1-score:  {metrics['F1']:.2%}")

    results_df = pd.DataFrame(results)

    print("\n" + "="*60)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА МОДЕЛЕЙ")
    print("="*60)
    print(results_df.to_string(index=False))

    best_model = results_df.loc[results_df['F1'].idxmax(), 'Model']
    print(f"\nЛучшая модель по F1-score: {best_model}")

    best_model_obj = models[best_model]
    y_pred_best = best_model_obj.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)

    print(f"\nМатрица ошибок ({best_model}):")
    print(f"   Негатив -> предсказано негатив: {cm[0,0]}, позитив: {cm[0,1]}")
    print(f"   Позитив -> предсказано негатив: {cm[1,0]}, позитив: {cm[1,1]}")

    return results_df, cm, X_test, y_test


def predict_sentiment(text, model_name='LogisticRegression'):
    """Функция 1: Предсказание тональности отзыва"""
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


def get_top_products(top_n=5):
    """Функция 2: Топ товаров по доле позитива"""
    if df_binary is None:
        return []

    product_sentiment = df_binary.groupby('product').agg({
        'sentiment_binary': ['sum', 'count']
    }).reset_index()

    product_sentiment.columns = ['product', 'positive_count', 'total_count']
    product_sentiment['positive_ratio'] = product_sentiment['positive_count'] / product_sentiment['total_count']
    product_sentiment = product_sentiment.sort_values('positive_ratio', ascending=False)

    return product_sentiment.head(top_n)[['product', 'positive_ratio', 'total_count']].to_dict('records')


def get_frequent_negative_words(top_n=5):
    """Функция 3: Самые частые слова в негативных отзывах"""
    if df_binary is None:
        return []

    negative_reviews = df_binary[df_binary['sentiment_binary'] == 0]['text'].values
    all_text = ' '.join(negative_reviews)
    words = re.findall(r'\b[а-яА-Яa-zA-Z]+\b', all_text.lower())
    word_counts = Counter(words)

    return word_counts.most_common(top_n)


def analyze_time_series():
    """Анализ временных рядов и прогноз"""
    global daily_counts, ts_model

    print("\n" + "="*60)
    print("ВРЕМЕННЫЕ РЯДЫ")
    print("="*60)

    daily_counts = df.groupby(df['date'].dt.date).size().reset_index(name='reviews_count')
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    daily_counts = daily_counts.sort_values('date')

    daily_counts['day_of_week'] = daily_counts['date'].dt.dayofweek
    for lag in [1, 2, 3]:
        daily_counts[f'lag_{lag}'] = daily_counts['reviews_count'].shift(lag)

    daily_counts = daily_counts.dropna()

    predictions = None
    y_test_ts = None

    if len(daily_counts) >= 10:
        feature_cols = ['day_of_week', 'lag_1', 'lag_2', 'lag_3']
        X_train_ts = daily_counts[feature_cols].values[:-3]
        y_train_ts = daily_counts['reviews_count'].values[:-3]
        X_test_ts = daily_counts[feature_cols].values[-3:]
        y_test_ts = daily_counts['reviews_count'].values[-3:]

        ts_model = LinearRegression()
        ts_model.fit(X_train_ts, y_train_ts)
        predictions = ts_model.predict(X_test_ts)

        print(f"\nПРОГНОЗ НА 3 ДНЯ:")
        for i, (actual, pred) in enumerate(zip(y_test_ts, predictions)):
            print(f"   День {i+1}: Факт = {int(actual):2d} | Прогноз = {int(max(0, pred)):2d}")
    else:
        print("Недостаточно данных")

    return predictions, y_test_ts


def build_recommendation_system():
    """Рекомендательная система на основе контента"""
    global product_texts, similarity_matrix

    print("\n" + "="*60)
    print("РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА")
    print("="*60)

    product_texts = df.groupby('product').agg({'text': lambda x: ' '.join(x)}).reset_index()
    vec = TfidfVectorizer(max_features=50)
    vectors = vec.fit_transform(product_texts['text'])
    similarity_matrix = cosine_similarity(vectors)

    def recommend(product_name, top_n=3):
        idx = product_texts[product_texts['product'] == product_name].index[0]
        scores = list(enumerate(similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]
        return [product_texts.iloc[i]['product'] for i, _ in scores]

    products = df['product'].unique()
    for p in products[:2]:
        print(f"\nРекомендации для '{p}':")
        for rec in recommend(p):
            print(f"   -> {rec}")

    return recommend


def visualize_and_save_report(results_df, cm, predictions, y_test_ts):
    print("\n" + "="*60)
    print("ВИЗУАЛИЗАЦИЯ И СОЗДАНИЕ ОТЧЁТА")
    print("="*60)

    import matplotlib.font_manager as fm

    available_fonts = [f.name for f in fm.fontManager.ttflist]
    cyrillic_fonts = ['DejaVu Sans', 'Arial', 'Verdana', 'Tahoma', 'Times New Roman']

    font_to_use = 'DejaVu Sans'
    for font in cyrillic_fonts:
        if font in available_fonts:
            font_to_use = font
            break

    plt.rcParams['font.family'] = font_to_use
    plt.rcParams['font.sans-serif'] = [font_to_use]
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

    # График 1: Распределение оценок
    ax1 = fig.add_subplot(gs[0, 0])
    rating_counts = df['rating'].value_counts().sort_index()
    all_ratings = [0, 0, 0, 0, 0]
    for rating, count in rating_counts.items():
        rating_int = int(rating)
        if 1 <= rating_int <= 5:
            all_ratings[rating_int - 1] = count

    bars = ax1.bar([1, 2, 3, 4, 5], all_ratings,
                   edgecolor='black', color='royalblue', alpha=0.7, width=0.6)
    ax1.set_title('Распределение оценок', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Рейтинг')
    ax1.set_ylabel('Количество')
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.set_xticklabels(['1', '2', '3', '4', '5'])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
    ax1.grid(alpha=0.3, axis='y')

    # График 2: Динамика отзывов
    ax2 = fig.add_subplot(gs[0, 1])
    if daily_counts is not None and len(daily_counts) > 0:
        if 'date' in daily_counts.columns and 'reviews_count' in daily_counts.columns:
            ax2.plot(daily_counts['date'], daily_counts['reviews_count'],
                    'b-o', linewidth=2, markersize=4)
            ax2.set_title('Динамика отзывов по дням', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Дата')
            ax2.set_ylabel('Количество отзывов')
            import matplotlib.dates as mdates
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
            if len(daily_counts) > 15:
                ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(daily_counts)//10)))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax2.transAxes)

    # График 3: Прогноз
    ax3 = fig.add_subplot(gs[0, 2])
    if predictions is not None and y_test_ts is not None and len(predictions) > 0:
        x_pos = np.arange(len(predictions))
        width = 0.35
        bars1 = ax3.bar(x_pos - width/2, y_test_ts, width, alpha=0.7, label='Факт', color='blue')
        bars2 = ax3.bar(x_pos + width/2, predictions, width, alpha=0.7, label='Прогноз', color='orange')
        ax3.set_title('Прогноз отзывов', fontsize=12, fontweight='bold')
        ax3.set_xlabel('День')
        ax3.set_ylabel('Количество')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'День {i+1}' for i in range(len(predictions))])
        ax3.legend()
        ax3.grid(alpha=0.3, axis='y')
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'Недостаточно данных\nдля прогноза', ha='center', va='center',
                transform=ax3.transAxes, fontsize=10)
        ax3.set_title('Прогноз отзывов', fontsize=12, fontweight='bold')

    # График 4: Матрица ошибок
    ax4 = fig.add_subplot(gs[1, 0])
    im = ax4.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, cm[i, j], ha='center', va='center', fontsize=14, fontweight='bold')
    ax4.set_title('Матрица ошибок', fontsize=12, fontweight='bold')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Негатив', 'Позитив'])
    ax4.set_yticklabels(['Негатив', 'Позитив'])

    # График 5: Популярность товаров
    ax5 = fig.add_subplot(gs[1, 1])
    product_counts = df['product'].value_counts()
    bars = ax5.bar(range(len(product_counts)), product_counts.values, color='seagreen', alpha=0.7)
    ax5.set_title('Популярность товаров', fontsize=12, fontweight='bold')
    ax5.set_xticks(range(len(product_counts)))
    ax5.set_xticklabels(product_counts.index, rotation=45, ha='right')
    ax5.set_ylabel('Количество отзывов')
    ax5.grid(alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # График 6: Соотношение тональности
    ax6 = fig.add_subplot(gs[1, 2])
    sentiment_counts = df_binary['sentiment_binary'].value_counts().sort_index()
    colors = ['#ff6b6b', '#51cf66']
    labels = ['Негатив', 'Позитив']
    ax6.pie(sentiment_counts.values, labels=labels,
            colors=colors, autopct='%1.1f%%', startangle=90)
    ax6.set_title('Соотношение тональности', fontsize=12, fontweight='bold')

    # График 7: Сравнение моделей
    ax7 = fig.add_subplot(gs[2, :2])
    x = np.arange(len(results_df))
    width = 0.2
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    colors_bar = ['#4c72b0', '#55a868', '#c44e52', '#8172b3']

    for i, metric in enumerate(metrics):
        ax7.bar(x + i*width, results_df[metric], width, label=metric, color=colors_bar[i], alpha=0.8)

    ax7.set_xlabel('Модели', fontweight='bold')
    ax7.set_ylabel('Значение метрики', fontweight='bold')
    ax7.set_title('Сравнение моделей классификации', fontsize=12, fontweight='bold')
    ax7.set_xticks(x + width * 1.5)
    ax7.set_xticklabels(results_df['Model'])
    ax7.legend()
    ax7.grid(alpha=0.3, axis='y')
    ax7.set_ylim([0, 1.1])

    # График 8: Топ товаров по позитиву
    ax8 = fig.add_subplot(gs[2, 2])
    top_products = get_top_products(5)
    if top_products:
        products_names = [p['product'] for p in top_products]
        ratios = [p['positive_ratio'] for p in top_products]
        bars = ax8.barh(products_names, ratios, color='gold', alpha=0.7)
        ax8.set_xlabel('Доля позитива')
        ax8.set_title('Топ товаров по позитиву', fontsize=12, fontweight='bold')
        ax8.set_xlim([0, 1])
        ax8.grid(alpha=0.3, axis='x')
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax8.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{ratios[i]:.1%}', ha='left', va='center', fontsize=9)

    # График 9: Облако слов
    ax9 = fig.add_subplot(gs[3, :])
    all_text = ' '.join(df['text'].values)
    wordcloud = WordCloud(width=1200, height=300, background_color='white',
                          colormap='viridis', max_words=100).generate(all_text)
    ax9.imshow(wordcloud, interpolation='bilinear')
    ax9.axis('off')
    ax9.set_title('Облако слов из отзывов', fontsize=12, fontweight='bold', pad=10)

    fig.suptitle('ML-ПРОДУКТ: АНАЛИЗ ОТЗЫВОВ КЛИЕНТОВ', fontsize=16, fontweight='bold', y=0.995)

    filename = 'ml_analysis_report.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"[OK] Отчёт сохранён: {filename}")

    plt.show()


def main():
    print("="*60)
    print("ML-ПРОДУКТ: АНАЛИЗ ОТЗЫВОВ")
    print("="*60)

    if load_data() is None:
        return

    results_df, cm, X_test, y_test = train_models()

    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ НОВЫХ ФУНКЦИЙ")
    print("="*60)

    print("\n1. Функция predict_sentiment():")
    test_phrases = [
        "отличный товар, всем рекомендую",
        "ужасное качество, разочарован",
        "нормально, но ничего особенного"
    ]
    for phrase in test_phrases:
        sentiment, confidence = predict_sentiment(phrase)
        print(f"   '{phrase}' -> {sentiment} ({confidence:.1f}%)")

    print("\n2. Функция get_top_products():")
    top_products = get_top_products(5)
    for i, prod in enumerate(top_products, 1):
        print(f"   {i}. {prod['product']}: {prod['positive_ratio']:.1%} позитива ({prod['total_count']} отзывов)")

    print("\n3. Функция get_frequent_negative_words():")
    freq_words = get_frequent_negative_words(5)
    for word, count in freq_words:
        print(f"   '{word}': {count} раз")

    predictions, y_test_ts = analyze_time_series()
    build_recommendation_system()

    visualize_and_save_report(results_df, cm, predictions, y_test_ts)

    print("\n" + "="*60)
    print("[OK] АНАЛИЗ ЗАВЕРШЁН")
    print("="*60)


if __name__ == '__main__':
    main()
