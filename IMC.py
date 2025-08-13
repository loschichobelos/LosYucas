# Importamos las librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random

# Función que clasifica el IMC según los rangos establecidos
def clasificar_imc(imc):
    if imc < 18.5:
        return "Bajo peso"
    elif imc < 25:
        return "Peso saludable"
    elif imc < 30:
        return "Sobrepeso"
    else:
        return "Obesidad"

# Establecemos una semilla para reproducibilidad
np.random.seed(42)

# Generamos 100 valores aleatorios de estatura entre 1.5 y 1.9 metros
estaturas = np.random.uniform(1.5, 1.9, 100)

# Generamos 100 valores aleatorios de peso entre 45 y 100 kg
pesos = np.random.uniform(45, 100, 100)

# Calculamos el IMC usando la fórmula: peso / estatura²
imcs = pesos / (estaturas ** 2)

# Clasificamos cada IMC en un nivel numérico (1 a 4)
# 1: Bajo peso, 2: Saludable, 3: Sobrepeso, 4: Obesidad
niveles = []
for imc in imcs:
    if imc < 18.5:
        niveles.append(1)
    elif imc < 25:
        niveles.append(2)
    elif imc < 30:
        niveles.append(3)
    else:
        niveles.append(4)

# Creamos un DataFrame con todos los datos
df = pd.DataFrame({
    'Estatura': estaturas,
    'Peso': pesos,
    'IMC': imcs,
    'Nivel de peso': niveles
})

# Mostramos un ejemplo aleatorio de los datos generados
ejemplo = random.randint(0, 99)
print(f"Ejemplo aleatorio:")
print(f"Estatura: {df.loc[ejemplo, 'Estatura']:.2f} m")
print(f"Peso: {df.loc[ejemplo, 'Peso']:.1f} kg")
print(f"IMC: {df.loc[ejemplo, 'IMC']:.1f}")
print(f"Categoría: {clasificar_imc(df.loc[ejemplo, 'IMC'])}")

# Creamos el modelo de regresión lineal
X = df[['IMC']]               # Variable independiente
y = df['Nivel de peso']       # Variable dependiente

# Entrenamos el modelo con los datos
modelo = LinearRegression()
modelo.fit(X, y)

# Mostramos la ecuación del modelo
print(f"\nModelo de regresión lineal:")
print(f"Nivel de peso = {modelo.coef_[0]:.4f} × IMC + {modelo.intercept_:.4f}")

# Graficamos los datos y la línea de regresión
plt.scatter(df['IMC'], df['Nivel de peso'], color='skyblue', label='Datos reales')  # puntos
plt.plot(df['IMC'], modelo.predict(X), color='red', label='Regresión lineal')       # línea
plt.xlabel('IMC')
plt.ylabel('Nivel de peso')
plt.title('Regresión lineal: IMC vs Nivel de peso')
plt.legend()
plt.grid(True)
plt.show()
