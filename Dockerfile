FROM python:3.7-slim

WORKDIR /PA3-PAGERANK

COPY . .

RUN pip install -r requirements.txt

CMD ["pagerank.py"]

ENTRYPOINT ["python3"]
