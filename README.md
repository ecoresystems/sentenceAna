# A simple sentence analysis web application

A simple web application which will take user's input sentence and generate corresponding vector or do sentiment analysis.

## Getting Started

### Prerequisites

The following are the required packages and their versions for the project, please make sure your tensorflow version is 1.x.x. 
```
numpy~=1.17.0
uvicorn~=0.11.8
fastapi~=0.61.0
pydantic~=1.6.1
starlette~=0.13.6
pandas~=1.0.5
tensorflow~=1.14.0
tensorflow-gpu~=1.14.0
scikit-learn~=0.23.1
```

### Installing

#### Download the datasets
Run the following shell script and the dataset will be ready to use

```
sh download_dataset.sh
```

#### Install additional packages (Not required, you can always use source code directly)
```
pip install bert
pip install tensorflow_hub
pip install bert-serving-server
pip install bert-serving-client
```
## Running the tests


### Train

Run train.py and it will use Stanford Large Movie Review Dataset to fine tune the model

```
python train.py
```

### Run the application

The application is divided into two parts: vectorization service and sentimental analysis, 
each part can work alone. However, the fastAPI is implemented on app.py and is associated with sentimental analysis module.
It is advised to run both.

#### Run the vectorization service

```
cd bert_server
python bert_server.py
```

#### Run the sentimental analysis service along with web service

```
python app.py
```



## Acknowledgments

* The code is based on Google's BERT models and its sample code (https://github.com/google-research/bert)