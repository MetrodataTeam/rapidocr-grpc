FROM paddlepaddle/paddle:2.6.1-gpu-cuda12.0-cudnn8.9-trt8.6

ARG target=/mdt/run

WORKDIR ${target}

RUN pip install --no-cache-dir poetry==2.0.1 && \
  poetry config virtualenvs.create false

ADD rapidocr/pyproject.toml rapidocr/poetry.lock ${target}/

RUN poetry install --no-cache --no-root
ADD rapidocr/pb/ ${target}/pb/
ADD rapidocr/*.py ${target}/

RUN python -m compileall ${target}
CMD ["python", "server.py"]
EXPOSE 18910
