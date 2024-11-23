# Swimming Pool Chair Occupancy Detection

## Overview
This project focuses on detecting occupancy in swimming pool chairs using computer vision techniques. We developed a model based on **YOLOv8** and created various workflows to handle images, videos, and live camera feeds. This repository contains Jupyter notebooks, a Streamlit application for demonstration, and plans for future enhancements.

## Table of Contents
- [Project Description](#project-description)
- [Getting Started](#getting-started)
- [Data Collection](#data-collection)
- [Model Development](#model-development)
- [Workflows](#workflows)
- [Future Work](#future-work)
- [Deployment Strategy](#deployment-strategy)
- [Contributing](#contributing)
- [License](#license)

## Project Description
The goal of this project is to detect whether swimming pool chairs are occupied or not using a YOLOv8 model trained on annotated data. The project consists of several components:
- Data collection and augmentation
- Model training and evaluation
- Multiple workflows for different input types
- A Streamlit application for user interaction

## Getting Started
To get started with this project, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/pik1989/SwimmingPoolCV.git
   cd pool-chair-occupancy-detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data Collection
- Initially, we collected images and videos from **Google**.
- We used **Roboflow** for annotating and augmenting our dataset.

![Data Collection](path/to/data-collection-image.png)

## Model Development
- We developed a YOLOv8 model for occupancy detection based on the annotated dataset.
- The training process included hyperparameter tuning to optimize model performance.

## Workflows
We created three different workflows:
1. **Image Detection**: Analyze images for chair occupancy.
2. **Video Detection**: Process video files for real-time analysis.
3. **Live Camera Detection**: Stream live camera feed for immediate occupancy detection.

## Future Work
- **Model Expansion**: Explore additional models to compare performance.
- **Data Augmentation**: Increase the dataset with more relevant images to enhance accuracy.
- **Deployment**: Deploy the application to make it accessible for users.

## Deployment Strategy
We plan to implement an AWS S3-based deployment strategy:
1. **Camera Feed**: Capture real-time data from the camera.
2. **Data Storage**: Store the incoming feed in **AWS S3**.
3. **Processing**: Use the YOLOv8 model to analyze the stored footage.

![Deployment Strategy](path/to/deployment-strategy-image.png)

## Contributing
We welcome contributions! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
