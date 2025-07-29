# Mental Health Risk Detection from Social Media Text

This project uses a deep learning model to classify Reddit posts for early-stage mental health risk detection. By processing text from various mental health and general-interest subreddits, the model learns to differentiate between posts indicating potential distress and those that are stable. The final model achieves an accuracy of **89%**.

## Technologies Used
*   Python
*   TensorFlow / Keras
*   Scikit-learn
*   NLTK
*   Pandas
*   PRAW (Python Reddit API Wrapper)

## Model Performance

The model performs well in identifying posts related to mental health, achieving high precision and recall for that class.

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **mental\_health** | **0.95** | **0.90** | **0.92** |
| stable | 0.77 | 0.86 | 0.81 |
| **Overall Accuracy** | | **89%** | |
