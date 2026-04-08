FROM public.ecr.aws/docker/library/python:3.11-slim

LABEL maintainer="FinOps Cloud Optimizer OpenEnv"
LABEL version="1.0.2"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Run server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
