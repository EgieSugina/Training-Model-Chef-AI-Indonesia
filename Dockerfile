# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Explicitly expose port 8080 for Cloud Run
EXPOSE 8080

# Set up Hugging Face token as environment variable (will be passed at runtime)
ENV HF_TOKEN=""

# Configure Hugging Face credentials if token is provided
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Container starting..."\n\
\n\
if [ -n "$HF_TOKEN" ]; then\n\
  echo "Configuring Hugging Face credentials..."\n\
  mkdir -p ~/.huggingface\n\
  echo "{\"tokens\": [\"$HF_TOKEN\"]}" > ~/.huggingface/token\n\
fi\n\
\n\
# Get the PORT environment variable with a default of 8080\n\
PORT="${PORT:-8080}"\n\
\n\
# Set Gradio server port and name\n\
export GRADIO_SERVER_PORT=$PORT\n\
export GRADIO_SERVER_NAME=0.0.0.0\n\
\n\
echo "Starting server on port $PORT..."\n\
exec python food.py\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Run the entrypoint script when the container launches
CMD ["/app/entrypoint.sh"] 