FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY wheels/ /wheels/

RUN pip install \
    --no-cache-dir \
    --no-index \
    --find-links=/wheels \
    -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
