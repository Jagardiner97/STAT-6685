import math

inputs = [0, 1]
wi = [0.6, -0.7]
wj = [0.5, 0.4]
wk = [-0.6, 0.8]
w21 = 1
w22 = 1

b1 = -0.4
b2 = -0.5
b3 = -0.5

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def perceptron(z):
    if z >= 0:
        return 1
    else:
        return 0


x = []
p = []
s = []

for i in inputs:
    for j in inputs:
        for k in inputs:
            z1 = i * wi[0] + j * wj[0] + k * wk[0] + b1
            z2 = i * wi[1] + j * wj[1] + k * wk[1] + b2

            # For perceptron
            p1 = perceptron(z1)
            p2 = perceptron(z2)
            pz3 = p1 * w21 + p2 * w22 + b3
            p3 = perceptron(pz3)

            # For sigmoid
            n1 = sigmoid(z1)
            n2 = sigmoid(z2)
            nz3 = n1 * w21 + n2 * w22 + b3
            n3 = sigmoid(nz3)

            # Store results
            x.append([i, j, k])
            p.append(p3)
            s.append(n3)

print("Perceptron")
for index, xval in enumerate(x):
    print(f"x=[{xval[0]}, {xval[1]}, {xval[2]}]: {p[index]}")

print("\nSigmoid")
for index, xval in enumerate(x):
    print(f"x=[{xval[0]}, {xval[1]}, {xval[2]}]: {round(s[index], 5)}")

print("\nTable")
print("x1 x2 x3| p s")
print("___________________")
for index, xval in enumerate(x):
    print(f"{xval[0]}  {xval[1]}  {xval[2]} | {p[index]} {round(s[index], 5)}")
