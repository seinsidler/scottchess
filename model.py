#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import optim

class ValueDataset(Dataset):
	def __init__(self):
		data = np.load("data/trainig_data_1M.npz")
		self.X = data['a']
		self.Y = data['b']
		self.X = self.X.astype(float)
		self.Y = self.Y.astype(float)
		print("loaded", self.X.shape, self.Y.shape)
	def __len__(self):
		return self.X.shape[0]
	def __getitem__(self, idx):
		return (self.X[idx], self.Y[idx])

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.fc1 = nn.Linear(209, 100).cuda()
		self.fc2 = nn.Linear(100, 100).cuda()
		self.fc3 = nn.Linear(100, 1).cuda()
		self.tanh = nn.Tanh()
		self.cuda()

	def forward(self, x):
		out = self.fc1(x).cuda()
		out = self.tanh(out).cuda()
		out = self.fc2(out).cuda()
		out = self.tanh(out).cuda()
		out = self.fc3(out).cuda()
		return out

if __name__ == "__main__":
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	chess_dataset = ValueDataset()
	chess_dataset.to(device)
	train_loader = torch.utils.data.DataLoader(chess_dataset, batch_size=100, shuffle=True)
	model = Net()
	model.to(device)
	optimizer = optim.Adam(model.parameters())
	floss = nn.MSELoss()

	if device == "cuda":
		model.cuda()

	model.train()
	for epoch in range(100):
		all_loss = 0
		num_loss = 0
		for i, data in enumerate(train_loader, 0):
			inputs, labels = data
			labels = labels.unsqueeze(-1)
			
			inputs, labels = inputs.to(device), labels.to(device) 

			optimizer.zero_grad()
			outputs = model(inputs.float())
			# outputs = outputs.to(device)
			loss = floss(outputs, labels.float())
			loss.backward()
			optimizer.step()

			all_loss += loss.item()
			num_loss += 1
		print("%3d: %f" % (epoch, all_loss/num_loss))
		torch.save(model.state_dict(), "value.pth")
	print('Finished Training')

