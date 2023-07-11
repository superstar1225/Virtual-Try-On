FROM python:3.8

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev

COPY . .

CMD ["python", "test.py"]