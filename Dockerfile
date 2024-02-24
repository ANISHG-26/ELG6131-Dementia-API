FROM python:3.9-slim-buster
WORKDIR /app
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "app.py"]