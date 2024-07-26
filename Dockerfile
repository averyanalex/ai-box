FROM python:3.12 AS builder
RUN python -m pip install --no-cache-dir poetry poetry-plugin-export
COPY poetry.lock pyproject.toml ./
RUN poetry export -f requirements.txt --output requirements.txt

FROM python:3.12
WORKDIR /app
COPY --from=builder requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY models.py ./
RUN python -m models
COPY app.py ./
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8736"]
