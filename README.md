# 🚀 Anomaly Detection in Network Traffic Using Unsupervised Learning
This project focuses on detecting anomalies in network traffic using unsupervised machine learning techniques, particularly Isolation Forest and Autoencoder Neural Networks. The goal is to identify unusual patterns in the network traffic that may indicate security breaches, system malfunctions, or abnormal usage.

📂 Dataset
The dataset used is the KDD Cup 1999 dataset, a benchmark dataset for network intrusion detection tasks. It contains various types of simulated network attacks and normal traffic with features extracted from TCP/IP connections.

🛠️ Project Workflow
1. Data Preprocessing
   • Loaded the dataset and assigned appropriate column names.

   • Inspected for missing or duplicate values.

   • Encoded categorical variables using Label Encoding.

   • Scaled the features using MinMaxScaler for consistency in model input.

2. Exploratory Data Analysis (EDA)
   • Plotted the distribution of attack vs. normal traffic using Seaborn.

   • Analyzed class imbalance and feature patterns.

3. Binary Labeling
   • Transformed the multi-class labels into binary:

      • normal → 0 (Normal)

      • All other attack types → 1 (Anomaly)

4. Isolation Forest
   • Trained the Isolation Forest model, an ensemble method designed for anomaly detection.

   • Set contamination level to reflect expected anomaly ratio.

   • Predicted anomalies and compared with ground truth.

   • Evaluated performance using metrics:

      • Accuracy

      • Precision

      • Recall

      • F1-Score

      • Confusion Matrix

5. Autoencoder Neural Network
   • Built a deep autoencoder using TensorFlow/Keras:

      • Encoder compresses the input features.

      • Decoder attempts to reconstruct the original input.

   • Calculated Reconstruction Loss (MSE) for each sample.

   • Used statistical thresholding (mean + 2 * std) to classify anomalies.

   • Evaluated model using the same metrics as Isolation Forest.

6. Model Comparison
   • Compared Isolation Forest and Autoencoder performance.

   • Discussed detection capabilities and trade-offs.

📊 Results Summary
| Model            | Accuracy | Precision | Recall | F1-Score |
| ---------------- | -------- | --------- | ------ | -------- |
| Isolation Forest | \~99%    | High      | High   | High     |
| Autoencoder      | \~98%    | High      | High   | High     |

Note: The exact values may vary depending on the threshold and contamination parameters.

💡 Key Takeaways
   • Unsupervised learning is effective for anomaly detection in unlabeled network data.

   • Isolation Forest is faster and simpler to implement.

   • Autoencoders provide more nuanced control and work well for complex data patterns.

   • Threshold tuning is critical for minimizing false positives/negatives.

📁 Project Structure
   Anomaly-Detection-in-Network-Traffic.ipynb  # Main notebook
   README.md                                  # Project documentation

📌 Future Improvements
   • Test with real-time traffic data.

   • Incorporate ensemble of multiple models.

   • Deploy as a live monitoring service with alerts.

🙌 Acknowledgements
This project was completed as part of my internship and aims to demonstrate practical application of machine learning.
