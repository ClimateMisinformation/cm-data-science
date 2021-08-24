FROM python:3.7.3-slim

LABEL version="0.3"
LABEL description="cm-ds"

WORKDIR /data

COPY requirements.txt .

RUN apt-get update && apt-get install -y gcc

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

COPY . /data

EXPOSE 8887

CMD ["jupyter","notebook","--ip=0.0.0.0", "--port=8887", "--no-browser", "--allow-root"]
