# Define the content of the Dockerfile
dockerfile_content = """
# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Command to run the Python script
CMD ["python", "your_script.py"]
"""

# Write the content to the Dockerfile
with open('Dockerfile', 'w') as f:
    f.write(dockerfile_content)

print("Dockerfile has been generated successfully.")
