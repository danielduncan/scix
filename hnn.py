# Adaption of a notebook which is an amalgamation of the hybrid qantum-classical neural network code from the Qiskit textbook, Georgina Carson and Samuel Wait's miscellaneous code, with some modification by me (Daniel Duncan) for the sake of data collection automation and ease of variable manipulation.

import qiskit
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import csv

from qiskit import IBMQ
from qiskit import transpile, assemble
from qiskit.visualization import *
from torchvision import datasets, transforms
from torch.autograd import Function

from IPython.display import FileLink

trials = 2
qubitsToUse = 1

# load IBM Q account
# IBMQ.save_account('')
provider = IBMQ.load_account()
backend = provider.backend.ibmq_qasm_simulator

# quantum circuit
class QuantumCircuit:
    def __init__(self, n_qubits, backend, shots):
        self._circuit = qiskit.QuantumCircuit(n_qubits)

        # all qubits in machine
        allQubits = [i for i in range(n_qubits)]
        # theta parameter
        self.theta = qiskit.circuit.Parameter('theta')

        # circuit itself
        self._circuit.h(allQubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, allQubits)
        # measure qubits
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
    
    def run(self, thetas):
        t_qc = transpile(self._circuit, self.backend)
        qobj = assemble(t_qc, shots=self.shots, parameter_binds = [{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        
        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)
        
        return np.array([expectation])

# hybrid function
class HybridFunction(Function):
    # forward pass
    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)

        return result
    
    # backward pass
    @staticmethod
    def backward(ctx, grad_output):
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())

        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left = ctx.quantum_circuit.run(shift_left[i])

            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients = np.array([gradients]).T
            return torch.tensor([gradients]).float() * grad_output.float(), None, None

class Hybrid(nn.Module):
    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(qubitsInUse, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)

# first 5 samples
n_samples = 5
X_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

idx = np.append(np.where(X_train.targets == 0) [0][:n_samples], np.where(X_train.targets == 1)[0][:n_samples])

X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)
n_samples_show = 5

data_iter = iter(train_loader)
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

while n_samples_show > 0:
    images, targets = data_iter.__next__()

    axes[n_samples_show - 1].imshow(images[0].numpy().squeeze(), cmap='gray')
    axes[n_samples_show - 1].set_xticks([])
    axes[n_samples_show - 1].set_yticks([])
    axes[n_samples_show - 1].set_title("Labeled: {}".format(targets.item()))

    n_samples_show -= 1

n_samples = 5

X_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

idx = np.append(np.where(X_test.targets == 0)[0][:n_samples], np.where(X_test.targets == 1)[0][:n_samples])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.hybrid = Hybrid((backend), 8192, np.pi / 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), -1)


# outputs the name of the device
print("Backend in use: ", backend.name())

for i in range(trials):
    for qubitsInUse in range(qubitsToUse):
        qubitsUsed = []
        qubitsUsed.append(qubitsInUse)
        qubitsInUse = qubitsInUse + 1
        print("Qubits currently in use: ", qubitsInUse)
        circuit = QuantumCircuit(qubitsInUse, backend, 8192)
        circuit._circuit.draw()

        optimizer = optim.Adam(Net().parameters(), lr=0.001)
        loss_func = nn.NLLLoss()

        epochs = 2

        loss_list = []
        model = Net()
        model.train()
        for epoch in range(epochs):
            total_loss = []
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                # forward pass
                output = model(data)
                # calculating loss
                loss = loss_func(output, target)
                # backward pass
                loss.backward()
                # optimise weights
                optimizer.step()

                total_loss.append(loss.item())
            loss_list.append(sum(total_loss)/len(total_loss))
            print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / epochs, loss_list[-1]))
        plt.plot(loss_list)
        plt.title('Training Convergence')
        plt.xlabel('Training Iterations')
        plt.ylabel('Neg Log Likelihood Loss')
        model.eval()
        with torch.no_grad():
            correct = 0
            for batch_idx, (data, target) in enumerate(test_loader):
                output = model(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                loss = loss_func(output, target)
                total_loss.append(loss.item())
                accuracy = correct / len(test_loader) * 100
                accuracyPlural = []
                accuracyPlural.append(accuracy)
            print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(sum(total_loss) / len(total_loss), accuracy))
        n_samples_show = 5
        count = 0
        fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if count == n_samples_show:
                    break
                output = model(data)

                pred = output.argmax(dim=1, keepdim=True)

                axes[count].imshow(data[0].numpy().squeeze(), cmap='gray')

                axes[count].set_xticks([])
                axes[count].set_yticks([])
                axes[count].set_title('Predicted {}'.format(pred.item()))

                count += 1
i = i + 1

qubitsVsAccuracy = {qubitsUsed[i]: accuracyPlural[i] for i in range(len(qubitsUsed))}

# function to export your data as a CSV (you can then use in Excel or any programming language you like)
def export_dict(filename, dict):
    with open(filename, 'w') as f:
        w = csv.DictWriter(f, dict.keys())
        w.writeheader()
        w.writerow(dict)
    local_file = FileLink(filename, result_html_prefix="Click here to download: ")
    display(local_file)

filename = 'results.csv'
export_dict(filename, qubitsVsAccuracy)
print("Experiment complete!")