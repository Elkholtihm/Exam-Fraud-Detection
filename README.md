# Cheating Detection in Online Exams 🎓

This is a real-time cheating detection system aims to identify suspicious actions and send notifications of cheating activities through a Telegram bot.

## 📂 Repository Contents

- **report.pdf** : A comprehensive report analyzing the model's results after training, including performance visualizations.
- **code/** : Directory containing all code and resources for the project.
  - **weights/** : Folder containing the trained model weights and images of the results.
  - **Data_processing.ipynb** : Notebook for data preprocessing and preparation before training.
  - **FraudSender.py** : Script for real-time predictions from the camera. When cheating is detected, a screenshot is sent via the Telegram bot.
  - **training.ipynb** : Notebook with code used to train the YOLOv8 model.
  - **yolov8n.pt** : YOLOv8 model weights used for trasfer training.

## 🔍 Project Overview

The goal of this project is to automatically detect suspicious behaviors in online exams, such as the presence of unauthorized persons in the frame or the use of prohibited devices.

### Project Steps:
1. **Data Collection** : The dataset was collected from [Mendeley Data](https://data.mendeley.com/).
2. **Data Processing** : Preprocessing was applied to prepare the images before training.
3. **Model Training** : The YOLOv8 model was used for training. Training results are visualized in graphs included in the report.
4. **Telegram Bot** : A Telegram bot was integrated to send alerts with screenshots when potential cheating is detected.

## 📈 Results

The training results show continuous improvement in precision and recall, as illustrated in the performance analysis charts. Detailed results can be found in the `report.pdf` file.

## 🚀 Using the Project

1. Clone the repository:
```bash
   git clone [repo]
```

2. Install the required packages:
```bash
  pip install ultralytics
  pip install opencv-python requests matplotlib
```

3. Navigate to the code directory:
```bash
  cd code
```

4. Navigate to the code directory:
```bash
  python FraudSender.py
```

## 🚀 Future Enhancements

To improve this project, the following steps are planned:
- **Add New Classes** : Include additional classes to detect objects like phones and earphones, enhancing cheating detection capabilities.
- **Transfer Learning for Improvement** : Use transfer learning techniques to fine-tune the model and improve detection accuracy.
- **Unauthorized Person Detection** : Detect the presence of unauthorized individuals in the frame to ensure only the exam candidate is present.

## 🛠️ Technologies Used

- **YOLOv8** for object detection.
- **Python** for data processing and model training.
- **Telegram API** for real-time notifications.

## 📬 Contact

For any questions or suggestions, please feel free to reach out.

