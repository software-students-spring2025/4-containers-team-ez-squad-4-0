FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 5001

CMD ["python", "app.py"]
