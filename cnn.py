import pandas as pd
import numpy as np

# Load the dataset
file_name = "/content/sample_data/realistic_clothing_size_dataset.csv"
df = pd.read_csv(file_name)

# Check the shape and columns of the DataFrame
print("DataFrame Shape:", df.shape)
print("DataFrame Columns:", df.columns)

# Separate features and target
# Assuming 'size' is the last column (index -1)
x = df.iloc[:, :-1].values  # Features (all columns except 'size')
y = df.iloc[:, -1].values   # Target ('size')

 #Encode the target variable (size) using LabelEncoder and OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert sizes to integers
y_onehot = to_categorical(y_encoded)        # One-hot encode the sizes

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y_onehot, test_size=0.3, random_state=42)

# Normalize the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train.shape()
x_test.shape()
y_test.shape()
y_train.shape()


# Build the CNN model
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

CNN_model = tf.keras.models.Sequential([
    Dense(128, input_dim=x_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])

CNN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = CNN_model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=200,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping]
)

# Evaluate the model
loss, cnn_accuracy = CNN_model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {cnn_accuracy:.4f}")

# Predict on the test set
y_pred = CNN_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert one-hot encoded predictions to class labels
y_test_classes = np.argmax(y_test, axis=1)  # Convert one-hot encoded test labels to class labels

# Classification report
from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_))

# Confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)