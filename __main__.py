import tensorflow as tf
import numpy as np

celsius = np.array([
    -40, -30, -20, -10, 0, 5, 10, 15, 20, 25, 30, 35, 38,
    40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100
], dtype=float)

fahrenheit = np.array([
    -40, -22, -4, 14, 32, 41, 50, 59, 68, 77, 86, 95, 100,
    104, 113, 122, 131, 140, 149, 158, 167, 176, 185, 194, 203, 212
], dtype=float)

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

print("Let's make a prediction!")
data = float(input("Enter temperature in Celsius: ")) 
result = model.predict(np.array([data]))
print(f"The result is {result[0][0]:.2f} Fahrenheit!")
