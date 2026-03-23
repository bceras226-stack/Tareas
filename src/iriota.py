import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

## Fijar pseudoaleatoriedad
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


## Cargar datasets
iris = load_iris()
X = iris.data.astype(np.float32) # Características
y = iris.target.astype(np.int32) # Etiquetas
class_names = iris.target_names # Nombre de las clases

print("Nombre de clases", class_names)
print("Shape de X:", X.shape)
print("Shape de Y", y.shape)


# Division del dataset
# Datos de entrenamiento (70%), datos temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size = 0.3,
    random_state = SEED,
    stratify = y
)
# Datos de testing y validation (50% del 30%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size = 0.5,
    random_state = SEED,
    stratify = y_temp
)

print("Shape del dataset")
print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# Nomalización - (Escalado)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# Construcción del model
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
        3, 
        activation="softmax", 
        )(x)
    return tf.keras.Model(inputs, outputs)

model = build_model(X_train.shape[1])

lr = 0.001
# Compilación del modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate = lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

print("\nResumen del modelo \t")
model.summary()

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True)
]


# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=16,
    verbose=1   
)

# Evaluación de los datos de testing
test_loss, test_acc = model.evaluate(X_test, y_test)

print("Resultados")
print("Pérdida testing:", test_loss)
print("Accuracy testing:", test_acc)

# Testing dataset
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\n=== Resutlados de Testing dataset ===")
print(f"Loss de prueba     : {test_loss:.4f}")
print(f"Accuracy de prueba : {test_acc:.4f}")

# Predicciones con dataset
y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

print("\n=== MATRIZ DE CONFUSIÓN ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== REPORTE DE CLASIFICACIÓN ===")
print(classification_report(y_test, y_pred, target_names=class_names))

# algunas predicciones individuales
print("\n=== EJEMPLOS DE PREDICCIÓN ===")
for i in range(min(5, len(X_test))):
    print(f"Muestra {i+1}:")
    print(f"  Clase real      : {class_names[y_test[i]]}")
    print(f"  Clase predicha  : {class_names[y_pred[i]]}")
    print(f"  Probabilidades  : {y_prob[i]}")
    print()

# Gráficas del entrenamiento
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.title("Evolución de la pérdida")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.axis([0, 300, 0, 1])
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(history.history["accuracy"], label="Train accuracy")
plt.plot(history.history["val_accuracy"], label="Val accuracy")
plt.title("Evolución de la exactitud")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.axis([0, 300, 0, 1])
plt.legend()
plt.grid(True)
plt.show()