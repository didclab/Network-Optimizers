FROM node:14 AS build
WORKDIR /app
COPY job-metrics-visualization/package.json ./
COPY job-metrics-visualization/package-lock.json ./
RUN npm install
COPY job-metrics-visualization/ ./
RUN npm run build

FROM python:3.9
WORKDIR /app/
ADD requirements.txt /app/
RUN pip install -r requirements.txt
COPY --from=build /app/build /app/build
ADD ./app /app/app
ADD ./config.ini /app/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]