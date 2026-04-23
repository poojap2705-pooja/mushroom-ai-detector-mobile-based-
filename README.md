# Mushroom Detection Model

This project is an image classification model for detecting mushrooms using TensorFlow and Keras. The model is trained on a dataset of mushroom images and can classify a given image to provide details about the mushroom from an Excel sheet.

## Project Structure

- `train_model.py`: Script to train the CNN model.
- `classify.py`: Script to classify a new image and get mushroom details.
- `requirements.txt`: List of required Python packages.
- `mushroom_model.h5`: Trained Keras model (generated after training).
- `mushroom_model.tflite`: TensorFlow Lite model for mobile deployment (generated after training).
- `class_indices.pkl`: Pickle file with class indices (generated after training).
- `data/`: Folder containing subfolders for each mushroom type with images.
- `mushroom_dataset (A-X).xlsx`: Excel file with mushroom details.

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Prepare your data:
   - Place mushroom images in folders named after the mushroom (e.g., `data/Amanita/image1.jpg`).
   - Create an Excel file `mushroom_details.xlsx` with columns like 'name', 'description', etc.

## Training the Model

Run the training script:
```
python train_model.py
```
This will train the model and save it as `mushroom_model.h5` and `mushroom_model.tflite`.

## Classifying an Image

To classify a new image:
```
python classify.py path/to/your/image.jpg
```
This will output the details of the detected mushroom.

## Mobile Deployment

Use the `mushroom_model.tflite` file for deploying on mobile devices using TensorFlow Lite.

## Notes

- Ensure the Excel sheet has a 'name' column matching the folder names.
- Adjust epochs and batch size in `train_model.py` as needed.
- For better accuracy, collect more images and fine-tune the model.