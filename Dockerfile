FROM python:3.12-slim

# We are building a single image that does build the vector index if needed, therefore we need to install the C dependencies
# Install a full C toolchain and C build-time dependencies for
# everything we're going to need.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install --assume-yes --no-install-recommends \
      build-essential \
      libpq-dev \
      libpq5 \
  && apt-get clean

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 5000

CMD [ "./entry_point.sh" ]