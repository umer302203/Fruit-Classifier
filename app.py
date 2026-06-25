import gradio as gr
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import re

# ---------- Helper Function: Clean Class Names ----------
def clean_class_name(raw_name):
    """
    Converts raw class names like 'apple_red_1' to 'Apple Red'
    and 'pear_1' to 'Pear'.
    """
    # Remove the trailing underscore and number (e.g., '_1', '_2')
    cleaned = re.sub(r'_\d+$', '', raw_name)
    # Replace underscores with spaces
    cleaned = cleaned.replace('_', ' ')
    # Capitalize each word
    cleaned = cleaned.title()
    return cleaned

# ---------- Load the Model (.keras format) ----------
model_path = 'fruits_classifier.keras'
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# ---------- Load Class Names ----------
with open('class_indices.json', 'r') as f:
    class_names_dict = json.load(f)

# Convert dictionary to a list for easy index-based access
class_names_list = [class_names_dict[str(i)] for i in range(len(class_names_dict))]
print(f"✅ Total classes loaded: {len(class_names_list)}")

# ---------- Prediction Function (Matches Training Preprocessing) ----------
def predict_image(image):
    """
    Preprocessing exactly matches training:
    - Resize to (64, 64)
    - Rescale by 1.0/255.0 (same as ImageDataGenerator rescale)
    - NO preprocess_input (VGG16 mean subtraction) because training didn't use it
    """
    # Step 1: Resize image to (64, 64) - matches target_size in training
    img = image.resize((64, 64))
    
    # Step 2: Convert PIL image to numpy array
    img_array = img_to_array(img)
    
    # Step 3: Rescale pixel values to [0, 1]
    # This matches: rescale=1.0/255.0 in ImageDataGenerator
    img_array = img_array / 255.0
    
    # Step 4: Add batch dimension (1, 64, 64, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Step 5: Run inference (same as model.predict in notebook)
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions, axis=-1)[0]
    confidence = np.max(predictions, axis=-1)[0]
    
    # Step 6: Get the raw class name and clean it for display
    raw_class_name = class_names_list[predicted_index]
    predicted_class = clean_class_name(raw_class_name)
    confidence_percentage = float(confidence) * 100
    
    return predicted_class, f"{confidence_percentage:.2f}%"

# ---------- Create Categories List for Display ----------
cleaned_class_names = [clean_class_name(name) for name in class_names_list]
categories_list = sorted(cleaned_class_names)
categories_text = ", ".join(categories_list)

description_text = f"""
### 🍎 Upload an image of a fruit.

**The model can predict the following {len(categories_list)} categories:**

{', '.join(categories_list)}

---

*Model: VGG16-based Transfer Learning*
*Input size: 64x64 | Preprocessing: Rescale to [0, 1]*
"""

# ---------- Gradio Interface ----------
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Fruit Image"),
    outputs=[
        gr.Textbox(label="🍎 Predicted Fruit"),
        gr.Textbox(label="📊 Confidence")
    ],
    title="🍎 Fruit Classification Using Transfer Learning",
    description=description_text,
)

# ---------- Launch ----------
if __name__ == "__main__":
    interface.launch()