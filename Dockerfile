# Use the official Python image for Python 3.12
FROM python:3.12.0-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create and set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install -r requirements.txt

# Copy the remaining app files into the container
COPY wineprice_app.py /app/
COPY mapping_dict.py /app/
COPY model.pkl /app/
COPY country_region_mapping.json /app/
COPY model_rmse.txt /app/

# Expose the port that Streamlit will run on (default is 8501)
EXPOSE 8501

# Command to run the Streamlit app
CMD streamlit run wineprice_app.py --server.port $PORT


