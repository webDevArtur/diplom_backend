# Skin Lesion Diagnosis Backend

This backend is built using FastAPI and utilizes SQLite as its database. It serves as a part of a decision support system for diagnosing skin lesions.

## Features

- CRUD operations for patients, diagnoses, images, and user authentication.
- Skin lesion segmentation using a neural network trained on 7500 images for classifying nevus, melanoma, and seborrheic keratosis with an accuracy of 86%.
- Segmentation endpoint with an accuracy of 85%.

## Setup

To run the backend on your local machine, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your/repository.git
cd repository

python -m venv venv
venv\Scripts\activate.bat


pip install -r requirements.txt

uvicorn main:app --reload

## API Endpoints

### /patients

- **GET**: Retrieve all patients
- **POST**: Add a new patient
- **PUT**: Update an existing patient
- **DELETE**: Delete a patient

### /diagnoses

- **GET**: Retrieve all diagnoses
- **POST**: Add a new diagnosis
- **PUT**: Update an existing diagnosis
- **DELETE**: Delete a diagnosis

### /images

- **GET**: Retrieve all images
- **POST**: Add a new image
- **PUT**: Update an existing image
- **DELETE**: Delete an image

### /auth

- **POST**: Authenticate user

### /segmentation

- **POST**: Perform lesion segmentation

## Segmentation Endpoint

### POST /segmentation

Perform skin lesion segmentation using a neural network model trained on 7500 images. Input image is required as part of the request payload. Returns segmented image.

## Neural Network Model

The neural network model used for segmentation achieves an accuracy of 85% in accurately segmenting skin lesions into nevus, melanoma, and seborrheic keratosis.
