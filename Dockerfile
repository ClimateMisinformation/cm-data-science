FROM continuumio/miniconda3

LABEL version="0.1"
LABEL description="cm-ds"

WORKDIR /data

COPY . /data

RUN pip install pandas seaborn scikit-learn jupyterlab

EXPOSE 8888

CMD ["jupyter","lab","--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
