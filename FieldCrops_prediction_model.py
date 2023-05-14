#!/usr/bin/env python
# coding: utf-8

# In[59]:


import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


# In[60]:


df_all = pd.read_csv("data_by_all_20230514.csv")
df_states = pd.read_csv("data_by_states_20230514.csv")


# In[61]:


# ID를 기준으로 dataframe 머지.
df = pd.merge(df_states, df_all, on='ID')


# In[62]:


# 작물이름에 따라 groupby
grouped = df.groupby('crop')

group_dict = {}  # 그룹을 저장할 딕셔너리

for group_id, group_df in grouped:
		if group_id != 'Wheat':
				# 아래 column들 삭제 및 column 명 변경
				group_df = group_df.drop(
						['productions_winter', 'harvested_winter', 'cultivated_winter'], axis=1)
				group_df = group_df.rename(
						columns={'productions_spring': 'productions', 'harvested_spring': 'harvested', 'cultivated_spring': 'cultivated'})
				
				# 결측치 있는 행 삭제
				group_df = group_df.dropna()
				group_dict[group_id] = group_df
		else:
				# productions_spring 열과 productions_winter 열이 모두 빈 행 삭제
				group_df = group_df.dropna(
						subset=['productions_spring', 'productions_winter'], how='all')
				
				# 겨울철 밀 데이터
				winter = group_df.copy()
				winter = winter.drop(
						['productions_spring', 'harvested_spring', 'cultivated_spring'], axis=1)
				winter = winter.rename(
                                    columns={'productions_winter': 'productions', 'harvested_winter': 'harvested', 'cultivated_winter': 'cultivated'})
				
				# 봄철 밀 데이터
				spring = group_df.copy()
				spring = spring.drop(
						['productions_winter', 'harvested_winter', 'cultivated_winter'], axis=1)
				spring = spring.rename(
						columns={'productions_spring': 'productions', 'harvested_spring': 'harvested', 'cultivated_spring': 'cultivated'})
				
				winter.dropna()
				spring.dropna()
				group_dict['Winter_wheat'] = winter
				group_dict['Spring_wheat'] = spring


# In[63]:


print(len(group_dict['Corn']))
print(len(group_dict['Rice']))
print(len(group_dict['Spring_wheat']))
print(len(group_dict['Winter_wheat']))
print(group_dict['Corn'].iloc[0])


# In[64]:


scaler = MinMaxScaler()
def normalized(dataFrame, category):
    normalized_value = scaler.fit_transform(dataFrame[[category]])
    dataFrame[category] = normalized_value
    return dataFrame


# In[65]:


from collections import defaultdict
data_corn = group_dict['Corn'].drop(['crop'], axis=1)
normalized(data_corn, 'temperature_avg')
normalized(data_corn, 'CDD')
normalized(data_corn, 'GDD')
normalized(data_corn, 'HDD')
normalized(data_corn, 'temperature_max')
normalized(data_corn, 'temperature_min')
normalized(data_corn, 'precipitation')
normalized(data_corn, 'snow_depth')
normalized(data_corn, 'snow_fall')
normalized(data_corn, 'sunlight_svm')
normalized(data_corn, 'sunlight_reg')
normalized(data_corn, 'productions')
normalized(data_corn, 'harvested')
normalized(data_corn, 'cultivated')
normalized(data_corn, 'fertilizer_price_index_all')
normalized(data_corn, 'fertilizer_price_index_nitrogen')
normalized(data_corn, 'fertilizer_price_index_phosphate')
normalized(data_corn, 'oil_price')
normalized(data_corn, 'total_meat(kg/capita)')
normalized(data_corn, 'total_meat_us(lb/capita)')
normalized(data_corn, 'fruits(kg/capita)')
normalized(data_corn, 'vegetable(kg/capita)')
normalized(data_corn, 'coffee_us(gal/capita)')
normalized(data_corn, 'tee_us(gal/capita)')
normalized(data_corn, 'cocoa_us(lb/capita)')
# date column의 연도(yyyy) 값 추출
data_corn['year-state'] = data_corn['ID'].str[:4] + data_corn['state']

# 각 'target' 값에 대응하는 'X' 값을 저장하기 위한 딕셔너리 생성
X_dict = defaultdict(list)
for _, row in data_corn.iterrows():
    X = row.drop(['ID', 'productions', 'year-state', 'state']).tolist()
    year_state = row['year-state']
    X_dict[year_state].append(X)

# X와 y 분리
X = [X_dict[year_state] for year_state in data_corn['year-state']]
y = data_corn['productions'].values.tolist()

# 데이터셋 확인
for X_set, target in zip(X, y):
    print(f"y = {target}, X = {X_set}")


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# In[68]:


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


# DataLoader로 데이터셋 생성
batch_size = 2
y_train = y_train  # y_train을 리스트로 변환
train_dataset = CustomDataset(X_train, y_train)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)


# In[70]:


class FieldCropsPredictionModel(nn.Module):
		def __init__(self, input_dim, feature_size, hidden_dim, layer_num=2):
				super(FieldCropsPredictionModel, self).__init__()
				self.filter_layer = nn.Linear(input_dim, 1, bias=True)
				self.hidden_layers = nn.ModuleList()
				self.output_layer = nn.Linear(hidden_dim, 1)
				nn.init.xavier_uniform_(self.filter_layer.weight)
				nn.init.xavier_uniform_(self.output_layer.weight)
				for i in range(layer_num):
						self.hidden_layers.append(
								nn.Linear(feature_size if i == 0 else hidden_dim, hidden_dim))

		def forward(self, input):
				batch_size, sequence_length, feature_size = input.size()

				out = []
				for i in range(batch_size):
						sequence = input[i].t()
						sequence_out = self.filter_layer(sequence)
						sequence_out = sequence_out.t()
						for layer in self.hidden_layers:
								sequence_out = layer(sequence_out)
						sequence_out = self.output_layer(sequence_out)
						out.append(sequence_out)

				out = torch.stack(out, dim=0)
				return out


# In[71]:


def train(model, dataloader, criterion, optim, scheduler, num_epochs):
		model.train()
		train_loss_list = []
		for epoch in range(num_epochs):
				running_loss = 0.0
				
				for X, y in dataloader:
						outputs = model(X)
						loss = criterion(outputs, y)
						
						optim.zero_grad()
						loss.backward()
						optim.step()

						running_loss += loss
				scheduler.step()

				epoch_loss = running_loss / len(dataloader.dataset)
				print('Epoch [{}/{}], train_loss: {:.4f}' .format(epoch+1, num_epochs, epoch_loss))
				train_loss_list.append(epoch_loss)
		
		return model, train_loss_list


# In[72]:


def validation(model, dataloader, criterion, num_epochs):
		model.eval()
		val_loss_list = []
		for epoch in range(num_epochs):
				running_loss = 0.0
				
				for X, y in dataloader:
						outputs = model(X)
						loss = criterion(outputs, y)

						running_loss += loss
				

				epoch_loss = running_loss / len(dataloader.dataset)
				val_loss_list.append(epoch_loss)
		
		return model, val_loss_list


# In[73]:


model = FieldCropsPredictionModel(12, 24, 15, 2)
optimizer_ft = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer_ft, step_size=7, gamma=0.1)
criterion = nn.MSELoss()


# In[74]:


model_train, train_loss = train(
    model, train_dataloader, criterion, optimizer_ft, exp_lr_scheduler, 50)

# model_val, val_loss = validation(
#     model, val_loader, criterion, 50)


# In[76]:


import matplotlib.pyplot as plt
import matplotlib.style as style

x = range(len(train_loss))
y = [loss.detach().numpy() for loss in train_loss]  # .detach().numpy() 사용
plt.plot(x, y, label='train loss')
plt.legend(loc='best')
plt.xlabel('epochs')
plt.show()

