from datetime import timedelta

SECRET_KEY = "your-secret-key-here"  # Секретный ключ для подписи токена

# Настройки для создания токена
TOKEN_EXPIRATION = timedelta(minutes=600)  # Время жизни токена

# Алгоритм подписи токена
ALGORITHM = "HS256"
