FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files to the container
COPY requirements.txt ./
COPY scaler.pkl ./
COPY model_nn.h5 ./
COPY app.py ./

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
