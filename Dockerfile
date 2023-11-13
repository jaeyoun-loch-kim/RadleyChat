
FROM --platform=linux/amd64 python:3.11

WORKDIR /root





COPY requirements.txt .

COPY app.py .
COPY chainlit.md .
COPY ["public", "./public"]
COPY ["Radley College_persist_512", "./Radley College_persist_512"]
COPY [".chainlit", "./.chainlit"]





RUN pip install -r requirements.txt






ENV OPENAI_API_KEY=

ENTRYPOINT [ "chainlit" ]

CMD ["run", "app.py"]
