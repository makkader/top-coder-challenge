import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Data Reading
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    reshaped_data = []
    for example in data:
        rs = {
            'trip_duration_days': example.get("input").get("trip_duration_days"),
            'miles_traveled': example.get("input").get("miles_traveled"),
            'total_receipts_amount': example.get("input").get("total_receipts_amount"),
            "expected_output": example.get("expected_output")
        }
        reshaped_data.append(rs)
    return pd.DataFrame(reshaped_data)

# 2. Custom PyTorch Dataset
class ReimbursementDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1) # Add a dimension for regression output

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# 3. Model Definition (Simple MLP)
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128) # Increased width
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32) # Added another layer
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x



from sklearn.metrics import mean_absolute_error

def cscore(y_true, y_pred):
    num_cases = len(y_true)
    #print(num_cases)
    exact_matches = sum([1 for yt, yp in zip(y_true, y_pred) if abs(yt - yp)<=1.00])
    #print(list(zip(y_true, y_pred.round(2))))
    print("close_match: ",exact_matches)
    avg_error = mean_absolute_error(y_true, y_pred)
    score = (avg_error * 100 + (num_cases - exact_matches) * 0.1)
    return score


if __name__ == "__main__":
    # Load data
    df = read_data("public_cases.json")
    

    # Define features (X) and target (y)
    X = df.drop(columns=['expected_output']).values
    y = df['expected_output'].values
   
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Create Dataset and DataLoader
    train_dataset = ReimbursementDataset(X_train, y_train)
    test_dataset = ReimbursementDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # Initialize model, loss function, and optimizer
    input_dim = X_train.shape[1]
    model = MLP(input_dim)

    # Ensure model runs on CPU
    device = torch.device("cpu")
    model.to(device)

    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 500
    print(f"Training model on {device}...")
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training complete.")

    test_criterion = nn.L1Loss()
    # Evaluation
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            print("shapes:",outputs.shape, targets.shape)
            loss = test_criterion(outputs, targets)
            test_loss += loss.item()
            #print("test")
            outputs_np = outputs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
    
    #print("Outputs:", outputs_np.flatten(), "Targets:", targets_np)
    print("CSCORE:",cscore(targets_np.flatten(), outputs_np.flatten()))
    print(f'Test Loss: {test_loss/len(test_loader):.4f}')

    # Save the trained model
    #torch.save(model.state_dict(), 'reimbursement_pytorch_model.pth')
    #print("PyTorch model saved to reimbursement_pytorch_model.pth")
