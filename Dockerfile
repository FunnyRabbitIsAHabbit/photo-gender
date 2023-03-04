FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10-slim-2023-01-09

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

ENTRYPOINT ["python", "main.py"]