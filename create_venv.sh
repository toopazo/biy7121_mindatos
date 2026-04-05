#!/bin/bash

# 0) Deactivate any previous venv
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Deactivating previous virtual environment: $VIRTUAL_ENV"
    deactivate
else
    echo "No previous virtual environment to deactivate."
fi

# 1) Remove any previous venv
rm -r .venv

# 2) Create virtualenv
python3 -m venv .venv
# python3 -m venv --system-site-packages venv

# 3) Activate venv
source .venv/bin/activate

# 4) Install requirements
python -m pip install -r requirements.txt

# 5) Check version
python -m pip --version

# 6) Check installed packages
python -m pip list

# 7) Check venv path
echo "Virtual environment path: $VIRTUAL_ENV"