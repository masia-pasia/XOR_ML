import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] <= 0.05:
            print("Stopping training on epoch " + str(epoch) + " because MSE reached below 0.05.")
            self.model.stop_training = True

early_stopping_callback = CustomEarlyStopping()

# Create lists to store weights
weights_hidden_layer = []
weights_output_layer = []

weight_saver = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: (
        weights_hidden_layer.append(model.layers[0].get_weights()[0]),
        weights_output_layer.append(model.layers[1].get_weights()[0])
    )
)

# Train the model with custom callback
history = model.fit(x, y, epochs=3000, verbose=0, batch_size=4, callbacks=[weight_saver, early_stopping_callback])


# Plot MSE
plt.plot(history.history['loss'], label='Wykres błędu sredniokwadartowego')
plt.title('Błąd sredniokwadratowy')
plt.ylabel('MSE')
plt.xlabel('Epoka')

plt.show()

plt.plot(1 - np.array(history.history['accuracy']), label='Wykres błędu klasyfikacji')
plt.title('Wykres błędu klasyfikacji')
plt.xlabel('Epoka')
plt.ylabel('Błąd')
plt.yticks([0, 0.25, 0.5, 0.75, 1])

plt.show()

# Plot weights change in both layers
weights_hidden_layer = np.array(weights_hidden_layer)
weights_output_layer = np.array(weights_output_layer)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(weights_hidden_layer[:, 0, 0], label='Neuron1, Waga1')
plt.plot(weights_hidden_layer[:, 0, 1], label='Neuron1, Waga2')
plt.plot(weights_hidden_layer[:, 1, 0], label='Neuron2, Waga1')
plt.plot(weights_hidden_layer[:, 1, 1], label='Neuron2, Waga2')
plt.title('Wagi warstwy ukrytej')
plt.xlabel('Epoka')
plt.ylabel('Waga')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(weights_output_layer[:, 0, 0], label='Waga1')
plt.plot(weights_output_layer[:, 1, 0], label='Waga2')
plt.title('Wagi warstwy wyjsciowej')
plt.xlabel('Epoka')
plt.ylabel('Waga')
plt.legend()


plt.show()


predictions = model.predict_on_batch(x)
print(predictions)