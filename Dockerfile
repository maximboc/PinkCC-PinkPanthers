FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set the default command to run your app
CMD ["python", "models/monai_models/monai_models.py"]
