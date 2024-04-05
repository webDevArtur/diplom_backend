from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, status, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jwt import PyJWTError, decode, encode
from datetime import datetime, timedelta
import io
import torch
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
import base64
from jwt_setting import SECRET_KEY, ALGORITHM
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import sqlite3
from typing import List, Dict, Any, Optional
from fastapi.middleware.cors import CORSMiddleware
from datetime import date

app = FastAPI()
security = HTTPBearer()

origins = [
    "*",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],  # Разрешить все заголовки
)

# Путь к базе данных SQLite
DATABASE_NAME = "medical_data.db"

# Создаем подключение к базе данных SQLite
conn = sqlite3.connect(DATABASE_NAME)
cursor = conn.cursor()

# Создаем таблицы, если они еще не существуют
cursor.execute('''
CREATE TABLE IF NOT EXISTS doctors (
    id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    patronymic TEXT NOT NULL,
    username TEXT NOT NULL,
    password TEXT NOT NULL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    patronymic TEXT NOT NULL,
    gender TEXT NOT NULL,
    date_of_birth DATE NOT NULL
)
''')


cursor.execute('''
CREATE TABLE IF NOT EXISTS diagnoses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    diagnosis TEXT NOT NULL,
    diagnosis_date DATE NOT NULL, 
    description TEXT,  
    FOREIGN KEY (patient_id) REFERENCES patients (id)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS diagnosis_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    diagnosis_id INTEGER NOT NULL,
    image_id INTEGER NOT NULL,
    FOREIGN KEY (diagnosis_id) REFERENCES diagnoses(id),
    FOREIGN KEY (image_id) REFERENCES images(id)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_data BLOB NOT NULL,
    upload_date DATE NOT NULL DEFAULT CURRENT_DATE
)
''')

# Закрываем соединение с базой данных
conn.close()



# Путь к предобученным моделям для классификации и сегментации
model_path_classification = "classification_with_no_segmentation_0.86.pth"
model_path_segmentation = 'deeplabv3_segmentation_0.85.pth'

# Определение числа классов
num_classes = 3  # Количество классов: меланома, невус, себорейный

# Проверяем доступность GPU и определяем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка предобученных моделей для классификации
model_classification = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes).to(device)
model_classification.load_state_dict(torch.load(model_path_classification, map_location=device))
model_classification.eval()

# Преобразования для тестового изображения для классификации
transform_classification = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Загрузка модели сегментации и определение класса DeepLabV3Wrapper
class DeepLabV3Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=21, aux_loss=True)
        self.classifier = nn.Conv2d(21, 1, kernel_size=1)

    def forward(self, image):
        output = self.model(image)['out']
        output = self.classifier(output)
        return output


# Инициализация модели и загрузка обученных весов
model_segmentation = DeepLabV3Wrapper().to(device)
model_segmentation.load_state_dict(torch.load(model_path_segmentation, map_location=device), strict=False)
model_segmentation.eval()


# Функция для создания JWT токена
def create_jwt_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(minutes=600)  # Время жизни токена 30 минут
    }
    token = encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


# Функция для проверки токена и извлечения данных пользователя
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Хранилище активных токенов
active_tokens = set()


# Ручка для аутентификации и создания JWT токена
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    # Создаем подключение к базе данных SQLite
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Проверяем логин и пароль в таблице doctors
    cursor.execute("SELECT username, password FROM doctors WHERE username=?", (username,))
    user = cursor.fetchone()

    # Закрываем соединение с базой данных
    conn.close()

    # Если пользователь существует и пароль верный
    if user and user[1] == password:
        token = create_jwt_token(username)
        active_tokens.add(token)  # Добавляем токен в список активных токенов
        return {"access_token": token, "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Ручка для выхода (удаления токена)
@app.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        if token in active_tokens:
            active_tokens.remove(token)  # Удаляем токен из списка активных токенов
            return {"message": "Successfully logged out"}
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )



# Ручка для классификации изображения с возвратом диагноза и графика вероятностей
@app.post("/classify_diagnosis/")
async def classify_diagnosis(current_user: dict = Depends(get_current_user), file: UploadFile = File(...)):
    # Считывание изображения из запроса и преобразование его в тензор для классификации
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_classification = transform_classification(image).unsqueeze(0).to(device)

    # Классификация изображения
    with torch.no_grad():
        output_classification = model_classification(image_classification)
        probabilities = torch.softmax(output_classification, dim=1)
    predicted_class_index = torch.argmax(probabilities, dim=1).item()
    predicted_class = ["меланома", "невус", "себорейный кератоз"][predicted_class_index]

    # Построение графика вероятностей
    class_names = ["меланома", "невус", "себорейный кератоз"]
    probabilities = probabilities.squeeze().cpu().numpy()
    plt.bar(class_names, probabilities)
    plt.xlabel('Класс')
    plt.ylabel('Вероятность')
    plt.title('Вероятности классов')
    plt.tight_layout()

    # Преобразование графика в изображение в формате base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()

    return JSONResponse(content={"diagnosis": predicted_class, "probabilities_graph": img_str},
                        media_type="application/json")


# Ручка для сегментации изображения
@app.post("/segmentation/")
async def segment_image(current_user: dict = Depends(get_current_user), file: UploadFile = File(...)):
    # Считывание изображения из запроса и преобразование его в тензор
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    transform_segmentation = transforms.Compose([
        transforms.Resize((224, 224)),  # Масштабирование изображения до размера 224x224
        transforms.ToTensor(),  # Преобразование в тензор
    ])
    image_tensor = transform_segmentation(image).unsqueeze(0).to(device)

    # Сегментация изображения
    with torch.no_grad():
        outputs = model_segmentation(image_tensor)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.squeeze(0).cpu().numpy()
        outputs = (outputs > 0.5).astype(np.uint8) * 255
        segmented_image = Image.fromarray(outputs[0], mode='L')

    # Убедимся, что размеры маски совпадают с размерами изображения
    segmented_image = segmented_image.resize(image.size)

    # Наложение маски на исходное изображение
    image_with_mask = image.copy()
    image_with_mask.putalpha(segmented_image)

    # Преобразование изображения с маской в формат base64
    buffered = io.BytesIO()
    image_with_mask.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return JSONResponse(content={"segmented_image": img_str}, media_type="application/json")

# Модель данных для пациента
class Patient(BaseModel):
    id: int = Field(default=None, title="ID пациента (автозаполняемый)", description="ID пациента, автоматически генерируемый сервером")
    first_name: str
    last_name: str
    patronymic: str
    gender: str
    date_of_birth: date



# Модель данных для диагноза
class Diagnosis(BaseModel):
    id: int = Field(default=None, title="ID диагноза (автозаполняемый)", description="ID диагноза, автоматически генерируемый сервером")
    patient_id: int
    diagnosis: str
    diagnosis_date: str
    description: Optional[str]

@app.get("/doctors/{doctor_id}/name")
async def get_doctor_name(doctor_id: int):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT first_name, last_name, patronymic FROM doctors WHERE id=?", (doctor_id,))
    doctor_data = cursor.fetchone()

    conn.close()

    if doctor_data:
        first_name, last_name, patronymic = doctor_data
        full_name = f"{last_name} {first_name} {patronymic}"
        return {"full_name": full_name}
    else:
        raise HTTPException(status_code=404, detail="Doctor not found")

# Ручка для создания нового пациента
@app.post("/patients/", response_model=Patient, dependencies=[Depends(get_current_user)])
async def create_patient(patient: Patient):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("INSERT INTO patients (first_name, last_name, patronymic, gender, date_of_birth) VALUES (?, ?, ?, ?, ?)",
                   (patient.first_name, patient.last_name, patient.patronymic, patient.gender, patient.date_of_birth))
    patient_id = cursor.lastrowid

    conn.commit()
    conn.close()

    patient.id = patient_id
    return patient

# Ручка для получения списка всех пациентов
@app.get("/patients/", response_model=List[Patient], dependencies=[Depends(get_current_user)])
async def read_patients():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM patients")
    patients_data = cursor.fetchall()

    patients_list = []
    for patient_data in patients_data:
        patient = Patient(id=patient_data[0], first_name=patient_data[1], last_name=patient_data[2],
                          patronymic=patient_data[3], gender=patient_data[4], date_of_birth=patient_data[5])
        patients_list.append(patient)

    conn.close()
    return patients_list

# Ручка для получения информации о конкретном пациенте по его ID
@app.get("/patients/{patient_id}", response_model=Patient, dependencies=[Depends(get_current_user)])
async def read_patient(patient_id: int):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM patients WHERE id=?", (patient_id,))
    patient_data = cursor.fetchone()

    if patient_data:
        patient = Patient(id=patient_data[0], first_name=patient_data[1], last_name=patient_data[2],
                          patronymic=patient_data[3], gender=patient_data[4], date_of_birth=patient_data[5])
    else:
        raise HTTPException(status_code=404, detail="Patient not found")

    conn.close()
    return patient

# Ручка для обновления данных о пациенте
@app.put("/patients/{patient_id}", response_model=Patient, dependencies=[Depends(get_current_user)])
async def update_patient(patient_id: int, patient: Patient):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Проверяем наличие пациента с указанным идентификатором
    cursor.execute("SELECT id FROM patients WHERE id=?", (patient_id,))
    existing_patient = cursor.fetchone()
    if not existing_patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    cursor.execute("UPDATE patients SET first_name=?, last_name=?, patronymic=?, gender=?, date_of_birth=? WHERE id=?",
                   (patient.first_name, patient.last_name, patient.patronymic, patient.gender, patient.date_of_birth, patient_id))

    conn.commit()
    conn.close()

    return patient


@app.delete("/patients/{patient_id}")
async def delete_patient(patient_id: int):
    # Создаем подключение к базе данных SQLite
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    try:
        # Проверяем наличие пациента с указанным идентификатором
        cursor.execute("SELECT id FROM patients WHERE id=?", (patient_id,))
        existing_patient = cursor.fetchone()
        if not existing_patient:
            return JSONResponse(status_code=404, content={"message": "Patient not found"})

        # Находим все диагнозы, связанные с этим пациентом
        cursor.execute("SELECT id FROM diagnoses WHERE patient_id=?", (patient_id,))
        diagnoses = cursor.fetchall()

        # Удаляем каждый диагноз и связанные с ним изображения
        for diagnosis_id in diagnoses:
            cursor.execute("SELECT image_id FROM diagnosis_images WHERE diagnosis_id=?", (diagnosis_id[0],))
            images = cursor.fetchall()
            for image_id in images:
                cursor.execute("DELETE FROM images WHERE id=?", (image_id[0],))
            cursor.execute("DELETE FROM diagnosis_images WHERE diagnosis_id=?", (diagnosis_id[0],))
            cursor.execute("DELETE FROM diagnoses WHERE id=?", (diagnosis_id[0],))

        # Удаляем данные о пациенте из таблицы
        cursor.execute("DELETE FROM patients WHERE id=?", (patient_id,))

        # Закрываем соединение с базой данных
        conn.commit()
        conn.close()

        return {"message": "Patient and associated diagnoses and images deleted successfully"}

    except Exception as e:
        conn.rollback()  # Откатываем транзакцию в случае ошибки
        return JSONResponse(status_code=500, content={"message": str(e)})  # Возвращаем сообщение об ошибке с кодом 500


# Ручка для добавления диагноза пациенту
@app.post("/patients/{patient_id}/diagnoses/", response_model=Diagnosis, dependencies=[Depends(get_current_user)])
async def add_diagnosis(patient_id: int, diagnosis_data: Diagnosis):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Проверяем наличие пациента с заданным patient_id
    cursor.execute("SELECT id FROM patients WHERE id=?", (patient_id,))
    existing_patient = cursor.fetchone()
    if not existing_patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Добавляем диагноз только если пациент существует
    cursor.execute("INSERT INTO diagnoses (patient_id, diagnosis, diagnosis_date, description) VALUES (?, ?, ?, ?)",
                   (patient_id, diagnosis_data.diagnosis, diagnosis_data.diagnosis_date, diagnosis_data.description))
    diagnosis_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return {"id": diagnosis_id, "patient_id": patient_id, "diagnosis": diagnosis_data.diagnosis, "diagnosis_date": diagnosis_data.diagnosis_date, "description": diagnosis_data.description}

# Ручка для получения диагнозов конкретного пациента
@app.get("/patients/{patient_id}/diagnoses/", response_model=List[Diagnosis], dependencies=[Depends(get_current_user)])
async def get_diagnoses(patient_id: int):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT id, diagnosis, diagnosis_date, description FROM diagnoses WHERE patient_id=?", (patient_id,))
    diagnoses_data = cursor.fetchall()

    diagnoses_list = [Diagnosis(id=diagnosis_data[0], patient_id=patient_id, diagnosis=diagnosis_data[1],
                                diagnosis_date=diagnosis_data[2], description=diagnosis_data[3]) for diagnosis_data in diagnoses_data]

    conn.close()
    return diagnoses_list

# Ручка для удаления диагноза по его ID
@app.delete("/diagnoses/{diagnosis_id}/")
async def delete_diagnosis(diagnosis_id: int):
    # Создаем подключение к базе данных SQLite
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    try:
        # Проверяем наличие диагноза с указанным ID
        cursor.execute("SELECT 1 FROM diagnoses WHERE id=?", (diagnosis_id,))
        existing_diagnosis = cursor.fetchone()
        if not existing_diagnosis:
            raise HTTPException(status_code=404, detail="Diagnosis not found")

        # Находим все изображения, связанные с этим диагнозом, и удаляем их
        cursor.execute("SELECT image_id FROM diagnosis_images WHERE diagnosis_id=?", (diagnosis_id,))
        images = cursor.fetchall()
        for image_id in images:
            cursor.execute("DELETE FROM images WHERE id=?", image_id)  # Исправлено здесь
        cursor.execute("DELETE FROM diagnosis_images WHERE diagnosis_id=?", (diagnosis_id,))

        # Удаляем диагноз из таблицы
        cursor.execute("DELETE FROM diagnoses WHERE id=?", (diagnosis_id,))

        # Закрываем соединение с базой данных
        conn.commit()
        conn.close()

        return {"message": "Diagnosis and associated images deleted successfully"}

    except Exception as e:
        conn.rollback()  # Откатываем транзакцию в случае ошибки
        raise e  # Повторно поднимаем исключение для обработки во фреймворке


# Ручка для обновления данных о диагнозе
@app.put("/diagnoses/{diagnosis_id}/", response_model=Dict[str, Any])
async def update_diagnosis(diagnosis_id: int, diagnosis_update: Diagnosis):
    # Создаем подключение к базе данных SQLite
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    try:
        # Проверяем существование диагноза с указанным ID
        cursor.execute("SELECT 1 FROM diagnoses WHERE id=?", (diagnosis_id,))
        existing_diagnosis = cursor.fetchone()
        if not existing_diagnosis:
            raise HTTPException(status_code=404, detail="Diagnosis not found")

        # Обновляем данные о диагнозе
        cursor.execute("UPDATE diagnoses SET diagnosis=?, diagnosis_date=?, description=? WHERE id=?",
                       (diagnosis_update.diagnosis, diagnosis_update.diagnosis_date, diagnosis_update.description, diagnosis_id))

        # Закрываем соединение с базой данных
        conn.commit()
        conn.close()

        return {"message": "Diagnosis updated successfully"}

    except Exception as e:
        conn.rollback()  # Откатываем транзакцию в случае ошибки
        raise e  # Повторно поднимаем исключение для обработки во фреймворке


# Функция для проверки существования диагноза по его ID
def diagnosis_exists(diagnosis_id: int) -> bool:
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM diagnoses WHERE id=?", (diagnosis_id,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


# Функция для создания записи об изображении и привязки его к диагнозу
def create_image_and_attach_to_diagnosis(diagnosis_id: int, image_data: bytes, upload_date: datetime) -> int:
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO images (image_data, upload_date) VALUES (?, ?)", (image_data, upload_date))
    image_id = cursor.lastrowid
    cursor.execute("INSERT INTO diagnosis_images (diagnosis_id, image_id) VALUES (?, ?)", (diagnosis_id, image_id))
    conn.commit()
    conn.close()
    return image_id


# Эндпоинт для загрузки изображения и привязки его к диагнозу
@app.post("/diagnoses/{diagnosis_id}/images/")
async def upload_image_and_attach_to_diagnosis(diagnosis_id: int, image: UploadFile = File(...), upload_date: str = Form(...)):
    if not diagnosis_exists(diagnosis_id):
        raise HTTPException(status_code=404, detail="Diagnosis not found")

    contents = await image.read()

    # Создаем запись об изображении и привязываем его к диагнозу
    image_id = create_image_and_attach_to_diagnosis(diagnosis_id, contents, upload_date)

    return {"image_id": image_id}

# Эндпоинт для получения списка изображений для конкретного диагноза
@app.get("/diagnoses/{diagnosis_id}/images/", response_model=List[Dict[str, Any]])
async def get_images_for_diagnosis(diagnosis_id: int):
    if not diagnosis_exists(diagnosis_id):
        raise HTTPException(status_code=404, detail="Diagnosis not found")

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT images.id, image_data, upload_date FROM images "
                   "INNER JOIN diagnosis_images ON images.id = diagnosis_images.image_id "
                   "WHERE diagnosis_images.diagnosis_id=?", (diagnosis_id,))
    images_data = cursor.fetchall()
    images = [{"image_id": row[0], "image": base64.b64encode(row[1]).decode('utf-8'), "upload_date": row[2]} for row in images_data]
    conn.close()

    return images

# Эндпоинт для удаления изображения
@app.delete("/images/{image_id}/")
async def delete_image(image_id: int):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Проверяем, существует ли указанное изображение
    cursor.execute("SELECT 1 FROM images WHERE id=?", (image_id,))
    existing_image = cursor.fetchone()
    if not existing_image:
        conn.close()
        raise HTTPException(status_code=404, detail="Image not found")

    # Удаляем изображение из таблицы images
    cursor.execute("DELETE FROM images WHERE id=?", (image_id,))
    conn.commit()
    conn.close()

    return {"message": "Image deleted successfully"}

# Эндпоинт для обновления изображения
@app.put("/images/{image_id}/")
async def update_image(image_id: int, image: UploadFile = File(...)):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Проверяем, существует ли указанное изображение
    cursor.execute("SELECT 1 FROM images WHERE id=?", (image_id,))
    existing_image = cursor.fetchone()
    if not existing_image:
        conn.close()
        raise HTTPException(status_code=404, detail="Image not found")

    contents = await image.read()

    # Обновляем запись в таблице images
    cursor.execute("UPDATE images SET image_data=? WHERE id=?", (contents, image_id))

    conn.commit()
    conn.close()

    return {"message": "Image updated successfully"}