import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)


hidden_1 = tf.keras.layers.Dense(units=3, input_shape=[1])
hidden_2 = tf.keras.layers.Dense(units=3)
output = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([hidden_1, hidden_2, output])


model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)


print("Starting training...")
history = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Trained model!")


plt.xlabel("# Time")
plt.ylabel("Magnitude of loss")
plt.plot(history.history["loss"])
plt.show()

print("Let's make a prediction!")

result = model.predict(np.array([100.0]))
print(f"The result is {result[0][0]:.2f} fahrenheit!")

print("Internal variables of the model")
print("Hidden layer 1 weights:", hidden_1.get_weights())
print("Hidden layer 2 weights:", hidden_2.get_weights())
print("Output layer weights:", output.get_weights())
