import winsound
import threading


def play_recommendation_sound():
    """Воспроизводит звук при выдаче рекомендации"""
    try:
        # Частота 1000 Гц, длительность 200 мс
        winsound.Beep(1000, 200)
    except Exception as e:
        print(f"Не удалось воспроизвести звук: {e}")


def play_success_sound():
    """Воспроизводит звук успеха (две ноты вверх)"""
    try:
        winsound.Beep(800, 150)
        winsound.Beep(1200, 150)
    except Exception as e:
        print(f"Не удалось воспроизвести звук: {e}")


def play_error_sound():
    """Воспроизводит звук ошибки (низкая нота)"""
    try:
        winsound.Beep(400, 300)
    except Exception as e:
        print(f"Не удалось воспроизвести звук: {e}")


def play_notification_sound():
    """Воспроизводит звук уведомления (короткий писк)"""
    try:
        winsound.Beep(1500, 100)
    except Exception as e:
        print(f"Не удалось воспроизвести звук: {e}")


def play_melody():
    """Воспроизводит мелодию при запуске"""
    try:
        notes = [
            (523, 200),  # C
            (587, 200),  # D
            (659, 200),  # E
            (698, 200),  # F
            (784, 400),  # G
        ]
        for freq, duration in notes:
            winsound.Beep(freq, duration)
    except Exception as e:
        print(f"Не удалось воспроизвести мелодию: {e}")


def play_sound_async(sound_func):
    """Воспроизводит звук в отдельном потоке, чтобы не блокировать программу"""
    thread = threading.Thread(target=sound_func)
    thread.daemon = True
    thread.start()


if __name__ == '__main__':
    print("Тест звуков:")
    print("1. Рекомендация...")
    play_recommendation_sound()

    print("2. Успех...")
    play_success_sound()

    print("3. Ошибка...")
    play_error_sound()

    print("4. Уведомление...")
    play_notification_sound()

    print("5. Мелодия...")
    play_melody()

    print("Тест завершен!")
