import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# Cargar dataset
# =========================
base_dir = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(base_dir, "data", "glass.csv")

df = pd.read_csv(file_path)

print(df.columns)

# Si no tiene headers:
# columns = ["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Type"]
# df = pd.read_csv(file_path, names=columns)

X = df.drop(columns=["Type of glass"]).values
y = df["Type of glass"].values

# =========================
# Codificar clases
# =========================
le = LabelEncoder()
y = le.fit_transform(y)

num_classes = len(np.unique(y))

print("Clases:", le.classes_)
print("Num clases:", num_classes)
print("Shape X:", X.shape)

# =========================
# División Train / Val / Test
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    random_state=SEED,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=SEED,
    stratify=y_temp
)

print("\nShape del dataset")
print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# =========================
# Normalización
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# =========================
# Modelo (EL TUYO)
# =========================
def build_model(input_dim):

    inputs = tf.keras.Input(shape=(input_dim,))
    
    x = tf.keras.layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(inputs)
    
    x = tf.keras.layers.Dropout(0.15)(x)
    
    x = tf.keras.layers.Dense(
        8,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    
    x = tf.keras.layers.Dropout(0.10)(x)
    
    outputs = tf.keras.layers.Dense(
        6, 
        activation="softmax"
    )(x)

    return tf.keras.Model(inputs, outputs)

model = build_model(X_train.shape[1])

model.summary()

# =========================
# Compilación
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# =========================
# Early stopping
# =========================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )
]

# =========================
# Entrenamiento
# =========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# =========================
# Evaluación
# =========================
test_loss, test_acc = model.evaluate(X_test, y_test)

print("\n=== RESULTADOS ===")
print("Loss:", test_loss)
print("Accuracy:", test_acc)

# =========================
# Predicciones
# =========================
y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# =========================
# ROC-AUC y PR-AUC multiclase
# =========================
y_test_bin = label_binarize(y_test, classes=np.arange(6))

roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr")
pr_auc = average_precision_score(y_test_bin, y_prob)

print("\n=== MÉTRICAS AVANZADAS ===")
print("ROC-AUC multiclase:", roc_auc)
print("PR-AUC multiclase:", pr_auc)

print("\n=== MATRIZ DE CONFUSIÓN ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== REPORTE DE CLASIFICACIÓN ===")
print(classification_report(y_test, y_pred))

# =========================
# Gráficas
# =========================
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.title("Evolución de la pérdida")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(history.history["accuracy"], label="Train accuracy")
plt.plot(history.history["val_accuracy"], label="Val accuracy")
plt.title("Evolución de la exactitud")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()