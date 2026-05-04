import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Импорт функций из main.py
from main import (
    df, simple_recommendation_system, get_worst_games,
    extract_rating, extract_price,
    add_review, game_reviews
)

from ml_module import (
    save_query, get_personalized_recommendations,
    get_query_statistics, train_simple_model, analyze_user_preferences
)

# Получение токена из переменной окружения
load_dotenv()
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

if not BOT_TOKEN:
    print("\n" + "="*60)
    print("ОШИБКА: Токен бота не найден!")
    print("="*60)
    print("\nСоздайте файл .env в корне проекта и добавьте:")
    print("TELEGRAM_BOT_TOKEN=ваш_токен_здесь")
    print("\nИли установите переменную окружения:")
    print("Windows: set TELEGRAM_BOT_TOKEN=ваш_токен")
    print("Linux/Mac: export TELEGRAM_BOT_TOKEN=ваш_токен")
    print("="*60 + "\n")
    exit(1)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    user = update.effective_user
    welcome_text = f"""
Привет, {user.first_name}! 👋

Я бот для рекомендаций игр Steam 🎮

Доступные команды:
/help - показать помощь
/recommend <запрос> - получить рекомендации
/search <название> - поиск игры по названию
/worst - топ-5 худших игр
/personal - персональные рекомендации
/stats - статистика запросов
/train - обучить модель
/myprofile - мой профиль

💬 Отзывы:
/addreview <название> - добавить отзыв
/reviews <название> - отзывы игры
/allreviews - все отзывы

Просто напишите запрос, например:
"популярный shooter"
"новая rpg"
"дешевый indie"
    """
    await update.message.reply_text(welcome_text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /help"""
    help_text = """
📖 Помощь по командам:

🔍 Поиск игр:
/recommend <запрос> - рекомендации по запросу
/search <название> - поиск игры по названию
Пример: /recommend популярный shooter
Пример: /search witcher

Или просто напишите запрос без команды:
"новая rpg"
"дешевый indie"

📊 Статистика:
/worst - топ-5 худших игр
/stats - общая статистика запросов
/myprofile - ваш профиль и предпочтения

🤖 Персонализация:
/personal - рекомендации на основе вашей истории
/train - обучить модель на всех запросах

💬 Отзывы:
/addreview <название> - добавить отзыв на игру
/reviews <название> - показать отзывы конкретной игры
/allreviews - показать все отзывы

🎯 Поддерживаемые жанры:
action, strategy, rpg, shooter, adventure, indie, simulation, horror

✨ Характеристики:
популярный, новый, дешевый, хороший, быстрый, сложный
    """
    await update.message.reply_text(help_text)


async def recommend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /recommend"""
    user_id = str(update.effective_user.id)
    query = ' '.join(context.args) if context.args else ''

    if not query:
        await update.message.reply_text("Укажите запрос. Пример: /recommend популярный shooter")
        return

    await update.message.reply_text("🔍 Ищу игры...")

    try:
        results = simple_recommendation_system(query, top_k=5)

        # Сохраняем запрос в историю
        save_query(user_id, query, results)

        if not results:
            await update.message.reply_text("Игры не найдены 😔")
            return

        response = f"🎮 Рекомендации по запросу '{query}':\n\n"
        max_score = results[0][1] if results else 100

        for i, (name, score, game) in enumerate(results, 1):
            compatibility = min(100, (score / max_score) * 100) if max_score > 0 else 0
            rating = extract_rating(game['all_reviews'])

            response += f"{i}. {name}\n"
            response += f"   Жанр: {game['genre']}\n"
            response += f"   Рейтинг: {rating}%\n"
            response += f"   Цена: {game['original_price']}\n"
            response += f"   Совместимость: {compatibility:.1f}%\n\n"

        await update.message.reply_text(response)

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")


async def worst_games(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /worst"""
    await update.message.reply_text("🔍 Ищу худшие игры...")

    try:
        results = get_worst_games(5)

        response = "💀 Топ-5 худших игр:\n\n"

        for i, (name, score, game, rating) in enumerate(results, 1):
            response += f"{i}. {name}\n"
            response += f"   Жанр: {game['genre']}\n"
            response += f"   Рейтинг: {rating}%\n"
            response += f"   Цена: {game['original_price']}\n\n"

        await update.message.reply_text(response)

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")


async def personal_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /personal - персональные рекомендации"""
    user_id = str(update.effective_user.id)

    await update.message.reply_text("🤖 Анализирую ваши предпочтения...")

    try:
        results = get_personalized_recommendations(user_id, df, top_k=5)

        if results is None:
            await update.message.reply_text(
                "У вас пока нет истории запросов 😔\n"
                "Сделайте несколько запросов, чтобы я мог изучить ваши предпочтения!"
            )
            return

        response = "🎯 Персональные рекомендации для вас:\n\n"
        max_score = results[0][1] if results else 100

        for i, (name, score, game) in enumerate(results, 1):
            compatibility = min(100, (score / max_score) * 100) if max_score > 0 else 0
            rating = extract_rating(game['all_reviews'])

            response += f"{i}. {name}\n"
            response += f"   Жанр: {game['genre']}\n"
            response += f"   Рейтинг: {rating}%\n"
            response += f"   Совместимость: {compatibility:.1f}%\n\n"

        await update.message.reply_text(response)

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")


async def user_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /myprofile - профиль пользователя"""
    user_id = str(update.effective_user.id)

    try:
        preferences = analyze_user_preferences(user_id)

        if preferences is None:
            await update.message.reply_text("У вас пока нет истории запросов 😔")
            return

        response = f"👤 Ваш профиль:\n\n"
        response += f"📊 Всего запросов: {preferences['total_queries']}\n\n"

        response += "🎮 Любимые жанры:\n"
        for genre, count in preferences['favorite_genres']:
            if count > 0:
                response += f"   • {genre}: {count} раз\n"

        response += "\n✨ Любимые характеристики:\n"
        for char, count in preferences['favorite_characteristics']:
            if count > 0:
                response += f"   • {char}: {count} раз\n"

        response += f"\n🕐 Последний запрос: {preferences['last_query']}\n"
        response += f"📅 Время: {preferences['last_query_time']}\n"

        await update.message.reply_text(response)

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")


async def statistics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /stats - общая статистика"""
    await update.message.reply_text("📊 Собираю статистику...")

    try:
        stats = get_query_statistics()

        if stats is None:
            await update.message.reply_text("Пока нет данных для статистики 😔")
            return

        response = "📊 Общая статистика:\n\n"
        response += f"🔢 Всего запросов: {stats['total_queries']}\n"
        response += f"👥 Уникальных пользователей: {stats['unique_users']}\n"
        response += f"📈 Среднее запросов на пользователя: {stats['avg_queries_per_user']}\n\n"

        response += "🔥 Популярные запросы:\n"
        for query, count in stats['popular_queries'][:5]:
            response += f"   • {query}: {count} раз\n"

        response += "\n🎮 Популярные жанры:\n"
        for genre, count in stats['popular_genres'][:5]:
            if count > 0:
                response += f"   • {genre}: {count} упоминаний\n"

        await update.message.reply_text(response)

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")


async def train_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /train - обучение модели"""
    await update.message.reply_text("🤖 Обучаю модель на основе всех запросов...")

    try:
        model = train_simple_model()

        if model is None:
            await update.message.reply_text("Недостаточно данных для обучения 😔")
            return

        response = "✅ Модель обучена!\n\n"
        response += f"📚 Обучающих примеров: {model['training_samples']}\n"
        response += f"🕐 Время обучения: {model['trained_at']}\n\n"

        response += "🎯 Самый популярный жанр: " + (model['most_popular_genre'] or 'N/A') + "\n"
        response += "✨ Самая популярная характеристика: " + (model['most_popular_characteristic'] or 'N/A') + "\n\n"

        response += "📊 Веса жанров:\n"
        for genre, weight in sorted(model['genre_weights'].items(), key=lambda x: x[1], reverse=True)[:5]:
            response += f"   • {genre}: {weight}\n"

        await update.message.reply_text(response)

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")


async def search_game(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /search - поиск игры по названию"""
    query = ' '.join(context.args) if context.args else ''

    if not query:
        await update.message.reply_text("Укажите название игры. Пример: /search witcher")
        return

    await update.message.reply_text(f"🔍 Ищу игры с названием '{query}'...")

    try:
        name_query = query.lower()
        results = df[df['name'].str.lower().str.contains(name_query, na=False)]

        if len(results) == 0:
            await update.message.reply_text(f"Игры с названием '{query}' не найдены 😔")
            return

        response = f"🎮 Найдено игр: {len(results)}\n\n"

        for idx, row in results.head(5).iterrows():
            response += f"{'='*40}\n"
            response += f"🎮 {row['name'].upper()}\n"
            response += f"{'='*40}\n"
            response += f"📊 Жанр: {row['genre']}\n"
            response += f"🏢 Разработчик: {row['developer']}\n"
            response += f"📢 Издатель: {row['publisher']}\n"
            response += f"📅 Дата выхода: {row['release_date']}\n"

            price = extract_price(row['original_price'])
            if price == 0:
                response += f"💰 Цена: Бесплатно\n"
            else:
                response += f"💰 Цена: ${price:.2f}\n"

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
            response += f"⭐ Рейтинг: {rating}% {rating_emoji}\n\n"

        await update.message.reply_text(response)

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")


async def add_review_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /addreview - добавить отзыв на игру"""
    game_name = ' '.join(context.args) if context.args else ''

    if not game_name:
        await update.message.reply_text("Укажите название игры. Пример: /addreview Witcher 3")
        return

    # Сохраняем название игры в контексте пользователя
    context.user_data['pending_review_game'] = game_name
    context.user_data['review_step'] = 'rating'

    await update.message.reply_text(
        f"📝 Добавление отзыва на игру '{game_name}'\n\n"
        f"Введите оценку от 1 до 10:"
    )


async def show_game_reviews(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /reviews - показать отзывы конкретной игры"""
    game_name = ' '.join(context.args) if context.args else ''

    if not game_name:
        await update.message.reply_text("Укажите название игры. Пример: /reviews Witcher 3")
        return

    try:
        if game_name not in game_reviews or len(game_reviews[game_name]) == 0:
            await update.message.reply_text(f"Для игры '{game_name}' пока нет отзывов 😔")
            return

        response = f"💬 ОТЗЫВЫ ДЛЯ '{game_name}':\n\n"

        for i, review in enumerate(game_reviews[game_name], 1):
            response += f"📝 Отзыв #{i} ({review['date']})\n"
            response += f"⭐ Оценка: {review['rating']}/10\n"
            response += f"💭 Текст: {review['text']}\n"
            response += f"{'-'*40}\n\n"

        await update.message.reply_text(response)

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")


async def show_all_reviews(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /allreviews - показать все отзывы"""
    try:
        if len(game_reviews) == 0:
            await update.message.reply_text("Пока нет отзывов 😔")
            return

        response = "💬 ВСЕ ОТЗЫВЫ:\n\n"

        for game_name, reviews in game_reviews.items():
            avg_rating = sum(r['rating'] for r in reviews) / len(reviews)
            response += f"🎮 {game_name}\n"
            response += f"   📊 Отзывов: {len(reviews)}\n"
            response += f"   ⭐ Средняя оценка: {avg_rating:.1f}/10\n"
            response += f"   📅 Последний отзыв: {reviews[-1]['date']}\n\n"

        await update.message.reply_text(response)

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых сообщений (запросов без команды)"""
    user_id = str(update.effective_user.id)
    query = update.message.text.strip()

    if not query:
        return

    # Проверяем, ожидается ли ввод для отзыва
    if 'review_step' in context.user_data:
        if context.user_data['review_step'] == 'rating':
            try:
                rating = int(query)
                if 1 <= rating <= 10:
                    context.user_data['review_rating'] = rating
                    context.user_data['review_step'] = 'text'
                    await update.message.reply_text("Теперь введите текст отзыва:")
                    return
                else:
                    await update.message.reply_text("Оценка должна быть от 1 до 10. Попробуйте снова:")
                    return
            except ValueError:
                await update.message.reply_text("Неверный формат. Введите число от 1 до 10:")
                return

        elif context.user_data['review_step'] == 'text':
            game_name = context.user_data['pending_review_game']
            rating = context.user_data['review_rating']
            review_text = query

            try:
                add_review(game_name, review_text, rating)
                await update.message.reply_text(
                    f"✅ Отзыв добавлен для игры '{game_name}'!\n"
                    f"⭐ Оценка: {rating}/10\n"
                    f"💭 Текст: {review_text}"
                )
            except Exception as e:
                await update.message.reply_text(f"Ошибка при добавлении отзыва: {str(e)}")

            # Очищаем контекст
            context.user_data.clear()
            return

    # Обычный поиск рекомендаций
    await update.message.reply_text("🔍 Ищу игры...")

    try:
        results = simple_recommendation_system(query, top_k=5)

        # Сохраняем запрос в историю
        save_query(user_id, query, results)

        if not results:
            await update.message.reply_text("Игры не найдены 😔")
            return

        response = f"🎮 Рекомендации:\n\n"
        max_score = results[0][1] if results else 100

        for i, (name, score, game) in enumerate(results, 1):
            compatibility = min(100, (score / max_score) * 100) if max_score > 0 else 0
            rating = extract_rating(game['all_reviews'])

            response += f"{i}. {name}\n"
            response += f"   Жанр: {game['genre']}\n"
            response += f"   Рейтинг: {rating}%\n"
            response += f"   Совместимость: {compatibility:.1f}%\n\n"

        await update.message.reply_text(response)

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")


def main():
    """Запуск бота"""
    print("\n" + "="*60)
    print("TELEGRAM БОТ - СИСТЕМА РЕКОМЕНДАЦИЙ ИГР")
    print("="*60)
    print("\nБот запускается...")
    print("Нажмите Ctrl+C для остановки")
    print("="*60 + "\n")

    # Создание приложения
    application = Application.builder().token(BOT_TOKEN).build()

    # Регистрация обработчиков команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("recommend", recommend))
    application.add_handler(CommandHandler("search", search_game))
    application.add_handler(CommandHandler("worst", worst_games))
    application.add_handler(CommandHandler("personal", personal_recommendations))
    application.add_handler(CommandHandler("myprofile", user_profile))
    application.add_handler(CommandHandler("stats", statistics))
    application.add_handler(CommandHandler("train", train_model))
    application.add_handler(CommandHandler("addreview", add_review_command))
    application.add_handler(CommandHandler("reviews", show_game_reviews))
    application.add_handler(CommandHandler("allreviews", show_all_reviews))

    # Обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
