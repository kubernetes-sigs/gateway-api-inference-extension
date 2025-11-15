FROM python:3.12-alpine
WORKDIR /app
COPY server.py /app/server.py

# ENTRYPOINT ensures that arguments passed at runtime are appended to this command.
ENTRYPOINT ["python", "-u", "server.py"]