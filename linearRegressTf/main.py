import tensorflow as tf # for linear regression
import pandas # to load data
import matplotlib.pyplot as plt # to plot the line the linear regression model has acheived during traininf

# creating class for a linear model with a weight(slope) and a bias(y-intercept)
class LinearModel:
    def __init__(self):
        self.w = tf.Variable(tf.random.normal(shape=(1, 1)), dtype="float32", name="w", trainable=True)
        self.b = tf.Variable(tf.random.normal(shape=(1, 1)), dtype="float32", name="b", trainable=True)

    # predicts an output from given input(passes x(input) into the linear function and receive y(output))
    def __call__(self, x):
        return self.w * x + self.b

# creating trainer class for linear model
class Trainer:
    def __init__(self, lr):
        self.lr = lr

    # training the model by calculating the gradient of the line and then adjusting the weight and bias to minimize cosr
    def train(self, linearModel, x, ty, epochs=1000):
        for e in range(epochs):
            with tf.GradientTape() as t:
                py = linearModel(x)
                cost = self.calcCost(ty, py)

            wg, bg = t.gradient(cost, [linearModel.w, linearModel.b])
            self.adjust(linearModel, wg, bg)

    # calculating the cost of the model with it's current weight and bias, uses Mean Squared Error, ty(target output) and py(predicted output)
    def calcCost(self, ty, py):
        return tf.losses.MSE(ty, py)

    # adjusting weight and bias with respect to computed graidents
    def adjust(self, linearModel, wg, bg):
        linearModel.w.assign_sub(wg * self.lr)
        linearModel.b.assign_sub(bg * self.lr)


# load dataframe with pandas       
with open("salaryData.csv") as f:
    df = pandas.read_csv(f)

# split up inputs and outputs
x = df["YearsExperience"]
ty = df["Salary"]

# defining model and defining the trainer for the model
model = LinearModel()
trainer = Trainer(1e-2)
trainer.train(model, x, ty)

# plot the model's acheived results
plt.title("Salary vs Years Experience", loc="left")
plt.xlabel("Years Experience")
plt.ylabel("Salary($)")
plt.scatter(x, ty, c="#F17979", label="Data")
plt.plot(x, [i * model.w.numpy()[0][0] + model.b.numpy()[0][0] for i in x], c="#79F1A0", label="Model's Solution")
plt.legend(loc="upper left")
plt.show()

# testing the model
testX = 6.5
testPy = model(6.5)
print(f"Model Test\nInput: {testX} years, Prediction: ${testPy.numpy()[0][0]}")
