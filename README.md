# Comment Categorization & Reply Assistant 💬

This project is a Streamlit web application that classifies user comments into 9 distinct categories and suggests an appropriate reply. It's designed to help creators, brands, and moderators efficiently manage community feedback.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://#)
*(Add your deployed Streamlit URL here if you deploy it!)*

---

## 📸 App Screenshot

(A good screenshot shows your app classifying a comment, like the "Constructive Criticism" example.)

![Screenshot of the Comment Classifier App](https://github.com/somethingshawnx/Comment-Categorization/blob/main/Screenshot%202025-11-14%20164014.png)

---

## ✨ Features

-   **Multi-Class Text Classification:** Sorts comments into 9 categories:
    -   Praise
    -   Support
    -   Emotional
    -   Constructive Criticism
    -   Hate/Abuse
    -   Threat
    -   Irrelevant/Spam
    -   Question/Suggestion
-   **Reply Assistant:** Provides a ready-to-use response template for each category.
-   **Interactive UI:** A simple web interface built with Streamlit to test the model in real-time.
-   **Balanced Model:** The classifier is trained on a custom-balanced dataset to ensure high accuracy, even for rare categories.
-   **Data Visualization:** Includes a bar chart showing the distribution of the final balanced training data.

---

## 🧠 Data & Model Methodology

A key challenge was the highly imbalanced nature of the source data. To solve this, the app builds a custom **balanced dataset** on its first run.

1.  **Data Sourcing & Mapping:** Data is sourced from two files (`comments.csv` and `train.csv`) and mapped to the project categories.
2.  **Augmentation:** To teach the model the *nuance* of categories like `Constructive Criticism`, a large set of diverse, high-quality manual examples are added.
3.  **Balancing (Undersampling):** The model is trained on a smaller, balanced set of ~10,000 comments, which includes a *sample* of the large original categories and *all* of the new, augmented data.

### Data Mapping Summary

| Project Category | Source File | Original Label(s) | Augmentation |
| :--- | :--- | :--- | :--- |
| **Praise** | `comments.csv` | `1 (joy)` | **Yes** (6 new examples, 50x) |
| **Support** | `comments.csv` | `2 (love)` | **Yes** (7 new examples, 50x) |
| **Emotional** | `comments.csv` | `0, 4, 5 (sad, fear, surprise)`| **Yes** (7 new examples, 50x) |
| **Hate/Abuse** | `comments.csv` | `3, 6 (anger, toxic)` | **Yes** (6 new examples, 50x) |
| **Threat** | `train.csv` | `threat == 1` | **Yes** (6 new examples, 50x) |
| **Constructive Criticism**| (N/A) | (N/A) | **Yes** (7 new examples, 50x) |
| **Irrelevant/Spam** | (N/A) | (N/A) | **Yes** (10 new examples, 50x) |
| **Question/Suggestion** | (N/A) | (N/A) | **Yes** (2 new examples, 50x) |

---

## 🛠️ Tech Stack & Installation

-   **Language:** `Python 3.10+`
-   **Application:** `Streamlit`
-   **Machine Learning:** `scikit-learn`
-   **NLP / Data:** `Pandas`, `NLTK`

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-blue?logo=nltk)](https://www.nltk.org/)

Install all libraries:
```bash
pip install streamlit pandas scikit-learn nltk
