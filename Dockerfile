# Name the single Python image we're using everywhere.
ARG python=python:3.10-slim

# Build stage:
FROM ${python} AS build

# Install a full C toolchain and C build-time dependencies for
# everything we're going to need.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install --no-install-recommends --assume-yes \
      build-essential \
      libpq-dev

# Create the virtual environment.
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install the Python library dependencies, including those with
# C extensions.  They'll get installed into the virtual environment.
WORKDIR /app
COPY requirements.txt create_vector_index.py ./
RUN pip install -r requirements.txt

# Accept build argument
ARG OPENAI_API_KEY

# Create the vector database
ENV OPENAI_API_KEY=$OPENAI_API_KEY
RUN python create_vector_index.py

# Final stage:
FROM ${python}

# Install the runtime-only C library dependencies we need.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install --no-install-recommends --assume-yes \
      libpq5


WORKDIR /app

# Copy the virtual environment from the first stage.
COPY --from=build /venv /venv
ENV PATH=/venv/bin:$PATH

# Copy the vector index
COPY --from=build /app/vector-database/ .

# Copy the public key in
COPY public_key.pem ./

# Copy the application in.
COPY app.py utils.py ./

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--log-level", "debug"]