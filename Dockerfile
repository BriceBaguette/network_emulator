# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Dash will run on (8050 by default)
EXPOSE 8050

# Define environment variable
ENV NAME NetSim-env

# Command to run the app
CMD ["python", "src/app.py"]
