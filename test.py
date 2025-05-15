import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

test_dir = 'data/test'
img_size = (224, 224)

# model = load_model('road_quality_model_qqqq.h5')
model = load_model('road_quality_model_qqqq.h5')
# model = load_model('best_model.h5')

cntGood = 0
cntPoor = 0
cnt1, cnt2 = 0.0, 0.0
for label in ['good', 'poor']:
    folder_path = os.path.join(test_dir, label)
    print(f"\n=== Checking {label.upper()} ===")

    for img_name in os.listdir(folder_path):
        if not img_name.lower().endswith(('.jpg')):
            continue
        img_path = os.path.join(folder_path, img_name)
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)[0][0]
        predicted_label = 'GOOD' if prediction < 0.5 else 'POOR'

        if label == 'good':
            cnt1 += 1
            if predicted_label == 'GOOD':
                cntGood += 1

        if label == 'poor':
            cnt2 += 1
            if predicted_label == 'POOR':
                cntPoor += 1

        print(f"{img_name}: predicted -> {predicted_label} (probebility: {prediction:.2f})")
print("Good prediction: ", (cntGood / cnt1) * 100)
print("Poor prediction: ", (cntPoor / cnt2) * 100)