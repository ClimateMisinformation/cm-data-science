FROM continuumio/miniconda3

LABEL version="0.2"
LABEL description="cm-ds"

WORKDIR /data

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

COPY . /data

EXPOSE 8887

CMD ["jupyter","lab","--ip=0.0.0.0", "--port=8887", "--no-browser", "--allow-root"]
