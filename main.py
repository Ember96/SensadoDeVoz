import os, librosa, math, json
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#variables de entorno
DATASET_PATH = "grabaciones"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

#definiendo la estructura en que se guardarán los datos
data = {
    "mapping": [],
    "mfcc": [],
    "labels": []
}
def mfccToJSON(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_vectors = math.ceil(samples_per_segment / hop_length)

    #recorriendo las grabaciones
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            data["mapping"].append(dirpath.split("/")[-1])

            for f in filenames:
                recording_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(recording_path, sr=SAMPLE_RATE)

                for s in range(num_segments):
                    start_sample = samples_per_segment * s
                    end_sample = start_sample + samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:end_sample], sr=sr,n_fft=n_fft ,n_mfcc=n_mfcc, hop_length=hop_length)
                    mfcc = mfcc.T

                    if len(mfcc) == num_vectors:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(recording_path, s+1))
    
    #se guarda el dataset generado en un archivo, remover "#" si desea activar este comportamiento
    #with open(json_path, "w") as file:
        #json.dump(data, file, indent=4)

def listToArray():
    with open(JSON_PATH, "r") as file:
        data = json.load(file)
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

def accuracyGraph(result):
    fig, axs = plt.subplots(2)
    #eje y: precisión
    axs[0].plot(result.history["accuracy"], label="Precisión en el entrenamiento")
    axs[0].plot(result.history["val_accuracy"], label="Precisión en la prueba")
    axs[0].set_ylabel("Precisión")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Valor de la precisión")

    #eje x: error o pérdida
    axs[1].plot(result.history["loss"], label="Error o pérdida en el entrenamiento")
    axs[1].plot(result.history["val_loss"], label="Error o pérdida en la prueba")
    axs[1].set_ylabel("Error o pérdida")
    axs[1].set_xlabel("Época")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Valor del error o pérdida")

    plt.show()

if __name__ == "__main__":
    mfccToJSON(DATASET_PATH, JSON_PATH)
    inputs, targets = listToArray()

    #separando el dataset generado en datos para entrenamiento y para prueba
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.2)

    #creando el modelo
    model = keras.Sequential([
        #capa de entrada
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        #capas ocultas
        keras.layers.Dense(521, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(11, activation="softmax"),
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    #entrenando el modelo
    result = model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), epochs=100, batch_size=16)

    accuracyGraph(result)