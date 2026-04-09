import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import asyncio

# Импорт функций из main.py
from main import (
    df, simple_recommendation_system, get_worst_games,
    search_by_range, extract_rating, extract_price, extract_year,
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
/worst - топ-5 худших игр
/personal - персональные рекомендации
/stats - статистика запросов
/train - обучить модель
/myprofile - мой профиль

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
Пример: /recommend популярный shooter

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
/review <название> - добавить отзыв
/reviews <название> - показать отзывы

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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых сообщений (запросов без команды)"""
    user_id = str(update.effective_user.id)
    query = update.message.text.strip()

    if not query:
        return

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
    application.add_handler(CommandHandler("worst", worst_games))
    application.add_handler(CommandHandler("personal", personal_recommendations))
    application.add_handler(CommandHandler("myprofile", user_profile))
    application.add_handler(CommandHandler("stats", statistics))
    application.add_handler(CommandHandler("train", train_model))

    # Обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
