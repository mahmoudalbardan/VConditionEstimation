FROM python:3.8-slim

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app
COPY src/model/model.pkl /app
COPY src/scripts/app.py /app

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
