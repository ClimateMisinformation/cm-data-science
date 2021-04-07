FROM continuumio/miniconda3

LABEL version="0.2"
LABEL description="cm-ds"

WORKDIR /data

RUN pip install pandas seaborn scikit-learn gensim nltk emoji jupyterlab 
RUN pip install pytest tqdm
RUN python -m nltk.downloader stopwords wordnet

COPY . /data

EXPOSE 8887

CMD ["jupyter","lab","--ip=0.0.0.0", "--port=8887", "--no-browser", "--allow-root"]
