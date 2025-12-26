⚖️ 🇮🇳 Indian-Legal-Judgement-Catchphrase-Extraction
Indian-Legal-Judgement-Catchphrase-Extraction is a Deep Learning Natural Language Processing (NLP) project designed to automatically extract meaningful catchphrases and keywords from Indian court judgments and legal files. By leveraging a hybrid architecture of Bidirectional LSTMs and Conditional Random Fields (CRF), this model captures both the sequential context of Indian legal text and the probabilistic dependencies of entity tags.

📖 Overview
Legal documents in the Indian judiciary are notoriously dense and lengthy. For legal professionals, summarizing these judgments into key "Catchphrases" (e.g., SECTION 302 IPC, CULPABLE HOMICIDE, BREACH OF CONTRACT) is essential for indexing, citation, and quick retrieval.

This project treats catchphrase extraction as a Sequence Labeling task (similar to Named Entity Recognition), specifically tailored for the vocabulary and structure found in Indian case law.

Key Features
Contextual Embeddings: Uses Doc2Vec and Word2Vec techniques to understand legal jargon.

Deep Memory Network: Utilizes a stacked Bidirectional LSTM (4 layers) to capture long-range dependencies in complex sentence structures common in Indian judgments.

Probabilistic Output: Employing a CRF (Conditional Random Field) layer to ensure the predicted sequence of tags is valid.

IOB Tagging: Implements the Inside-Outside-Beginning tagging scheme for precise phrase boundary detection.

🧠 Model Architecture
The model processes text through the following pipeline:

Input Layer: Tokenized legal text sequences (padded to max length).

Embedding Layer: Pre-trained embeddings (Doc2Vec) are mapped to an embedding matrix.

Bi-LSTM Stack:

Layer 1: 256 Units (Bidirectional)

Layer 2: 128 Units (Bidirectional)

Layer 3: 64 Units (Bidirectional)

Layer 4: 32 Units (Bidirectional)

CRF Layer: A Conditional Random Field layer acts as the final classifier to predict the optimal sequence of tags (Catchphrase vs. Non-Catchphrase).

🛠️ Tech Stack & Libraries
Language: Python

Deep Learning: TensorFlow, Keras, Keras-contrib (for CRF)

NLP: NLTK, Gensim (Doc2Vec)

Data Handling: Pandas, NumPy

Evaluation: Scikit-learn (Seqeval metrics)

📂 Dataset
The model is trained on a legal dataset (Final_200.csv) containing:

Text: Full text of the court judgment/case.

Catchphrases: The ground truth keywords associated with the case.

Note: The dataset requires preprocessing (lowercasing, punctuation removal, and stopword filtering) before entering the model pipeline.

🚀 Installation & Usage
1. Clone the Repository
Bash

git clone https://github.com/SaurabhMutekar/Indian-Legal-Judgement-Catchphrase-Extraction.git
cd Indian-Legal-Judgement-Catchphrase-Extraction
2. Install Dependencies
This project relies on tensorflow-addons and keras-contrib for the CRF layer.

Bash

pip install tensorflow pandas numpy gensim nltk scikit-learn
pip install tensorflow-addons
pip install git+https://www.github.com/keras-team/keras-contrib.git
pip install sklearn-crfsuite
3. Run the Notebook
Open the Jupyter Notebook to train the model and generate predictions.

Bash

jupyter notebook Legal_Catchphrase_Extraction.ipynb
📊 Performance
The model evaluates predictions using Precision, Recall, and F1-Score.

Training Accuracy: ~95% (High accuracy due to the dominance of the 'O' tag in the dataset).

Evaluation Metric: Macro F1-Score (Used to handle class imbalance between catchphrases and standard text).

Sample Extraction
Input Text:

"...appellant convicted under section 302 of indian penal code for murder..."

Extracted Catchphrases:

"Appellant"

"Conviction"

"Section 302"

"Indian Penal Code"

"Murder"

🔮 Future Improvements
Class Imbalance Handling: Implementing weighted loss functions to better prioritize catchphrase tags over non-catchphrase text.

Transformer Models: Experimenting with Legal-BERT (trained on Indian legal corpus) or RoBERTa for better contextual embeddings compared to Doc2Vec.

Data Augmentation: Increasing the dataset size beyond 200 documents to improve generalization across different courts (High Court vs. Supreme Court).

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
  <img src="https://media.giphy.com/media/l0HlPTbGpCn2xQQSI/giphy.gif" width="200" alt="Legal Hammer">
</p>
