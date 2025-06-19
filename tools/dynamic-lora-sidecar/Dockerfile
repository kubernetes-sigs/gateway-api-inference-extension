FROM python:3.10-slim-buster AS test

WORKDIR /dynamic-lora-reconciler-test
COPY requirements.txt .
COPY sidecar/* ./ 
RUN pip install -r requirements.txt
RUN python -m unittest discover || exit 1  

FROM python:3.10-slim-buster

WORKDIR /dynamic-lora-reconciler

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sidecar/* ./

CMD ["python", "sidecar.py"]