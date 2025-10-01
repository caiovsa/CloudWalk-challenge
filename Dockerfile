# We install dependencies here to create a smaller final image.
FROM python:3.13-slim AS builder

# Set the working directory
WORKDIR /app

### Etapa do Poetry
# Existe outra forma mais "correta" de instalar o Poetry, mas essa é a mais simples.
RUN pip install poetry

# Copiamos apenas os arquivos necessários para instalar as dependências (pyproject.toml e poetry.lock).
COPY pyproject.toml poetry.lock* ./

# Vai instalar apenas as dependências necessárias para rodar a aplicação (main).
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --only main --no-root

### Etapa da imagem final
FROM python:3.13-slim

# Workdir normal
WORKDIR /app

# Copia as dependências instaladas na etapa anterior
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
# Copia os executáveis (como uvicorn) instalados na etapa anterior
COPY --from=builder /usr/local/bin /usr/local/bin

# Copia o código da aplicação
COPY . .

# Usuario sem privilegios para ficar mais facil
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Vai expor a porta 8005 (a mesma que usamos no uvicorn)
EXPOSE 8005

# Comando uvicorn para rodar a belezura
CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8005"]