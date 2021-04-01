FROM continuumio/miniconda3

LABEL version="0.2"
LABEL description="cm-ds"

WORKDIR /data

COPY . /data

RUN pip install pandas seaborn scikit-learn gensim nltk emoji jupyterlab

EXPOSE 8888

CMD ["jupyter","lab","--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
