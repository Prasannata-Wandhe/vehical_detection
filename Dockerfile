FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PORT=7860

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
