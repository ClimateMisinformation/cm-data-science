# cm-data-science

Hi all and welcome to the climate misinformation data science repo! 

Feel free to create your own branches and start playing around with the data that is stored in the labelled_data directory.

You will find text preprocessing and embedding pipeline in the text_preprocessing directory. 

In the models directory you will find the implementation of several models and their performance evaluation

### Install dependences

#####One time only: 
- Set up virtual environment and add this environment to ipykernel.
- The virtual environment allows you to install and use specific 
packages for this project without interfering with other projects. 
You will need to enter the virtual environment each time before running
code.

    ```
    python3 -m venv ~/venvs/cm-venv
    source ~/venvs/cm-venv/bin/activate
    pip install -r requirements.txt
    python -m ipykernel install --name=cm-venv
    ```

##### Every time before running code:
1. Enter virtual environment.
    ```
    source ~/venvs/cm-venv/bin/activate
    ```
2. You will see your Terminal prompt begin with '(cm-venv)'.

##### Every time to run Jupyter Notebok:
1. Open jupyter notebook.
    ```
    jupyter notebook
    ```
2. Select 'Kerrnel' > 'Change kernel' > cm-venv


### Docker setup
```
docker build -t cd-ds .
docker run --rm -it -p 8887:8887 -v "`pwd`":/data cd-ds
```

Then follow the link with 127.0.0.1 to open Jupyterlab


<!-- ##### Model evaluation

Model comparison...
![png](models/model_evaluation.PNG)


"Best" model...
![png](models/model_evaluation_best.PNG) -->
