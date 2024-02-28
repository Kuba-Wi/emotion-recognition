#Contributing guide
### Main information if you would like to help in this project

### 1. Fork
> git clone **<FORK_GIT_REPO_PATH>**

### 2. Preparing local workspace

To start developing first download 'virtualenv' and create new virtual environment.
> pip install virtualenv
> 
> python -m virtualenv venv

Now activate it.
### Linux
> source venv/bin/activate
### Windows
>.\venv\Scripts\activate.bat

### What next
Install development packages.
> pip install -r dev-requirements.txt

In order to run tests:
> tox -e black-check,mypy,flake8

To run just type command:
> emotion_recognition

Before submitting automatically format source files:
>tox -e black