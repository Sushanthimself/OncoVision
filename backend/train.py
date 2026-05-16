"""
OncoVision AI — Fine-Tuning Script
This script uploads images from the training_data folder and starts a fine-tuning
job on Google GenAI to create a custom model for your specific slides.
"""

from google import genai
from google.genai import types
import os
import sys

# 1. Initialize the client (reads GEMINI_API_KEY from .env automatically)
try:
    from dotenv import load_dotenv
    load_dotenv()
    client = genai.Client()
except Exception as e:
    print(f"Error initializing client. Make sure GEMINI_API_KEY is in your .env file. Error: {e}")
    sys.exit(1)

BASE_DIR = "./training_data"
categories = {
    "dentigerous_cyst": "This is a Dentigerous Cyst. It is benign but associated with impacted teeth.",
    "ductal_carcinoma": "This is Ductal Carcinoma. It is a type of breast cancer and is malignant.",
    "normal_tissue": "This is normal, healthy tissue with no signs of cancer or cysts."
}

training_examples = []

print("🚀 Starting data preparation...")

for folder_name, expected_output in categories.items():
    folder_path = os.path.join(BASE_DIR, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue
        
    files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(files)} images in {folder_name}")
    
    for filename in files:
        full_path = os.path.join(folder_path, filename)
        print(f"Uploading {filename} to Google...")
        
        try:
            # Upload the image to Google's file API
            image_file = client.files.upload(file=full_path)
            
            # Create a training example
            example = types.TunedModelExample(
                text_input="Analyze this biopsy image and provide a diagnosis.",
                image_input=image_file,
                output=expected_output
            )
            training_examples.append(example)
            
        except Exception as e:
            print(f"Failed to upload {filename}: {e}")

if not training_examples:
    print("❌ No training examples found! Please put images in the folders under backend/training_data/ first.")
    sys.exit(1)

print(f"📦 Total training examples ready: {len(training_examples)}")
print("🤖 Starting the fine-tuning job on Google Cloud. This may take some time...")

try:
    operation = client.models.create_tuned_model(
        id="oncovision-custom-model",
        source_model="gemini-2.5-flash",
        training_data=training_examples,
    )
    
    print("Job submitted successfully!")
    print("You can check the status in your Google AI Studio console.")
    print("Once complete, update MODEL_ID in main.py to 'tunedModels/oncovision-custom-model'")
    
except Exception as e:
    print(f"❌ Failed to start fine-tuning job: {e}")
