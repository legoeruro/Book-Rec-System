### This project is initialized through poetry (because python package management is hell :). 
Please install python and poetry before you start! Once you've finished:
- I recommend running this for easier cleanup (venv will be in project folder): `poetry config --local virtualenvs.in-project true`
- Change dir to folder with `poetry.lock`. Run poetry install to install all dependencies: `poetry install`
- To add packages, call `poetry add <package name>`

After installing all dependencies, run python files with the virtualenv in your IDE or call `poetry run python <script name>.py`

