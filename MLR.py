import numpy as np
import math 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('Student_Performance.csv')
x1 = df['Sleep Hours'].values
x2 = df['Sample Question Papers Practiced'].values
x3 = df['Performance Index'].values
x4 = np.array([1 if item == 'Yes' else 0 for item in df['Extracurricular Activities']])
x5 = df['Previous Scores'].values
Y = df['Performance Index'].values

#Medias 
Mx1= sum(x1)/len(x1)
Mx2= sum(x2)/len(x2)
Mx3= sum(x3)/len(x3)
Mx4= sum(x4)/len(x4)
Mx5= sum(x5)/len(x5)
My= sum(Y)/len(Y)


SP_X1Y = sum((x1[i] - Mx1) * (Y[i] - My) for i in range(len(x1)))
SP_X2Y = sum((x2[i] - Mx2) * (Y[i] - My) for i in range(len(x2)))
SP_X3Y = sum((x3[i] - Mx3) * (Y[i] - My) for i in range(len(x3)))
SP_X4Y = sum((x4[i] - Mx4) * (Y[i] - My) for i in range(len(x4)))
SP_X5Y = sum((x4[i] - Mx4) * (Y[i] - My) for i in range(len(x4)))

SP_X1X1 = sum((x1[i] - Mx1) ** 2 for i in range(len(x1)))
SP_X2X2 = sum((x2[i] - Mx2) ** 2 for i in range(len(x2)))
SP_X3X3 = sum((x3[i] - Mx3) ** 2 for i in range(len(x3)))
SP_X3X4 = sum((x4[i] - Mx4) ** 2 for i in range(len(x4)))
SP_X3X5 = sum((x5[i] - Mx5) ** 2 for i in range(len(x5)))

SP_X1X2 = sum((x1[i] - Mx1) * (x2[i] - Mx2) for i in range(len(x1)))
SP_X1X3 = sum((x1[i] - Mx1) * (x3[i] - Mx3) for i in range(len(x1)))
SP_X1X4 = sum((x1[i] - Mx1) * (x4[i] - Mx4) for i in range(len(x1)))
SP_X1X5 = sum((x1[i] - Mx1) * (x5[i] - Mx5) for i in range(len(x1)))
SP_X2X3 = sum((x2[i] - Mx2) * (x3[i] - Mx3) for i in range(len(x2)))
SP_X2X4 = sum((x2[i] - Mx2) * (x4[i] - Mx4) for i in range(len(x2)))
SP_X2X5 = sum((x2[i] - Mx2) * (x5[i] - Mx5) for i in range(len(x2)))
SP_X3X4 = sum((x3[i] - Mx3) * (x4[i] - Mx4) for i in range(len(x3)))
SP_X3X5 = sum((x3[i] - Mx3) * (x5[i] - Mx5) for i in range(len(x3)))
SP_X4X5 = sum((x4[i] - Mx4) * (x5[i] - Mx5) for i in range(len(x4)))
SP_X4X4 = sum((x4[i] - Mx4) * (x4[i] - Mx4) for i in range(len(x4)))
SP_X5X5 = sum((x5[i] - Mx5) * (x5[i] - Mx5) for i in range(len(x4)))


beta1 = SP_X1Y / SP_X1X1
beta2 = SP_X2Y / SP_X2X2
beta3 = SP_X3Y / SP_X3X3
beta4 = SP_X4Y / SP_X4X4
beta5 = SP_X5Y / SP_X5X5

beta0 = My - (beta1 * Mx1) - (beta2 * Mx2) - (beta3 * Mx3) - (beta4 * Mx4) - (beta5 * Mx5)

yprom = beta0 + beta1 * x1 + beta2 * x2 + beta3 * x3 + beta4 * x4 + beta5 * x5

mse = mean_squared_error(Y, yprom)
print(f"MSE -> {mse}")

r2 = r2_score(Y, yprom)
print(f"R2 -> {r2}")

plt.scatter(Y, yprom, label="Población", color="red")
plt.xlabel("Índice de Rendimiento Real")
plt.ylabel("Índice de Rendimiento Predicción")
plt.title("Índice de Rendimiento Real vs. Predicción de la Población")
plt.legend()
plt.show()