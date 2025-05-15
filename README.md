**Requirements:** pip install tensorflow numpy, and other things that will trow terminal



**Train model:** python model.py



**To test model:** python test.py



**Dataset link:** https://www.kaggle.com/datasets/juhibhojani/sih-challenge-set/data



Also you can find dataset in Data folder. I did not used all original dataset.



Recomended to do train and test it several times to get better result.




## 📄 Description

I did not use ResNet as I said before — instead, I decided to use a simple CNN.

---

## ⚙️ Hyperparameters

- **Input size:** 224×224×3  
- **Batch size:** 32  
- **Epochs:** 10  
- **Optimizer:** Adam (learning rate = 0.0001)  
- **Loss function:** Binary Crossentropy  

---

## 🧪 Augmentation

- **Rotation (20°):** Simulates camera angle changes from drones or vehicles.  
- **Zoom (0.2):** Accounts for varying distances from the road surface.  
- **Horizontal flip:** Introduces left/right symmetry, useful for bidirectional roads.  
- **Validation split (0.2)**

---

## 🧠 Architecture

- 3 convolutional blocks with increasing filter sizes: **32 → 64 → 128**  
- ReLU activations and MaxPooling layers  
- A fully connected Dense layer with Dropout  
- Final **sigmoid** activation for binary classification  


