# Use a slim Python base image
FROM python:3.12-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Set the entrypoint to your main script (using shell form for robustness)
ENTRYPOINT python run_with_json_export_tel.py

# Default command to run if no arguments are provided to docker run
# This provides a default input message for the script
CMD ["AI safety measures in autonomous vehicles containerized"]
