FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# تنظیم متغیر محیطی برای کش tiktoken و دانلود فایل توکنایزر در مرحله بیلد
ENV TIKTOKEN_CACHE_DIR=/tiktoken_cache
RUN mkdir -p /tiktoken_cache && \
    python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
