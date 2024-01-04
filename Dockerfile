FROM python:3.9

WORKDIR /app/
ADD requirements.txt /app/
RUN pip install -r requirements.txt
ADD ./app /app/app
ADD ./config.ini /app/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]