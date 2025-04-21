FROM osrf/ros:noetic-desktop-full

# Установка системных зависимостей
RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-numpy && \
    rm -rf /var/lib/apt/lists/*

# Явно указываем совместимые версии
RUN pip3 install --no-cache-dir \
    numpy==1.23.5 \
    pandas==1.5.3 \
    matplotlib \
    scipy
    
WORKDIR /app

COPY bag_to_csv.py .

# Указываем команду для запуска
CMD ["python3", "rosbag_reader.py"]

