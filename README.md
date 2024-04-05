# Серверная часть диагностики кожных поражений

Эта серверная часть построена с использованием FastAPI и использует SQLite в качестве базы данных. Она служит частью системы поддержки принятия решений для диагностики кожных поражений.

## Особенности

- Операции CRUD для пациентов, диагностики, изображений и аутентификации пользователей.
- Сегментация повреждений кожи с использованием нейронной сети, обученной на 7500 изображениях, для классификации невусов, меланомы и себорейного кератоза с точностью до 86%.
- Конечная точка сегментации с точностью до 85%.

## Настройка

Чтобы запустить серверную часть на вашем локальном компьютере, выполните следующие действия:

1. Клонируйте репозиторий:


```bash
git clone https://github.com/your/repository.git
cd repository

python -m venv venv
venv\Scripts\activate.bat


pip install -r requirements.txt

uvicorn main:app --reload
```

## Конечные точки API

### /patients

- **GET**: Восстановить всех пациентов
- **POST**: Добавить нового пациента
- **PUT**: Обновить существующего пациента
- **DELETE**: Удалить пациента

### /diagnoses

- **GET**: Восстановить все диагнозы
- **POST**: Добавить новый диагноз
- **PUT**: Обновить существующий диагноз
- **DELETE**: Удалить диагноз

### /images

- **GET**: Получить все изображения
- **POST**: Добавить новое изображение
- **PUT**: Обновить существующее изображение
- **DELETE**: Удалить изображение

### /auth

- **POST**: Аутентифицировать пользователя

### /сегментация

- **POST**: Выполнить сегментацию повреждения

## Конечная точка сегментации

### POST /classification

Выполните классификацию повреждения кожи.

## Модель нейронной сети

Модель нейронной сети, используемая для сегментации efficient-b0, обеспечивает точность в 86% при разделении поражений кожи на невусы, меланому и себорейный кератоз. Используя модель нейронной сети, обученную на 7500 изображениях. Ручка возвращает диагноз и вероятность.
