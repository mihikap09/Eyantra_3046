import torch.nn as nn
import pandas
import torch
import numpy as np
from sklearn.model_selection import train_test_split

def data_preprocessing(task_1a_dataframe):
    from sklearn.preprocessing import LabelEncoder
    encoded_dataframe = task_1a_dataframe.copy()
    encoded_dataframe['Age'] = ((encoded_dataframe['Age']) - 20) // 5
    encoded_dataframe['JoiningYear'] = ((2023 - encoded_dataframe['JoiningYear']) // 3) - 1
    label_encoder = LabelEncoder()
    encoded_dataframe['Education'] = label_encoder.fit_transform(encoded_dataframe['Education'])
    encoded_dataframe['Gender'] = label_encoder.fit_transform(encoded_dataframe['Gender'])
    encoded_dataframe['City'] = label_encoder.fit_transform(encoded_dataframe['City'])
    encoded_dataframe['EverBenched'] = label_encoder.fit_transform(encoded_dataframe['EverBenched'])
    encoded_dataframe['JoiningYear'] = label_encoder.fit_transform(encoded_dataframe['JoiningYear'])
    encoded_dataframe['PaymentTier'] = label_encoder.fit_transform(encoded_dataframe['PaymentTier'])
    encoded_dataframe.fillna(-1, inplace=True)

    return encoded_dataframe

def identify_features_and_targets(encoded_dataframe):
    df = encoded_dataframe
    target_label_name = df.columns[-1]
    target_label = encoded_dataframe[target_label_name]
    # Selecting all columns except the last one as features
    features_name = df.columns[:-1]
    features = encoded_dataframe[features_name]
    # Creating a list with features and target label
    features_and_targets = [features, target_label]

    return features_and_targets

def load_as_tensors(features_and_targets):
    from torch.utils.data import TensorDataset, DataLoader
    features = features_and_targets[0].to_numpy()
    targets = features_and_targets[1].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, shuffle=True)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader

class Salary_Predictor(nn.Module):
    def __init__(self):
        super(Salary_Predictor, self).__init__()
        self.fc1 = nn.Linear(8, 32)  # Increase hidden units
        self.fc2 = nn.Linear(32, 16)  # Add another hidden layer
        self.fc3 = nn.Linear(16, 1)  # Output layer
        self.hidden = nn.ReLU()  # Use ReLU activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.hidden(x)
        x = self.fc2(x)
        x = self.hidden(x)
        x = self.fc3(x)
        return x

def model_loss_function():
    loss_function = nn.MSELoss()
    return loss_function

def model_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Use Adam optimizer
    return optimizer

def model_number_of_epochs():
    number_of_epochs = 200  # Increase the number of epochs
    return number_of_epochs

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in tensors_and_iterable_training_data[4]:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(tensors_and_iterable_training_data[4])
        print(f'Epoch [{epoch + 1}/{number_of_epochs}] - Loss: {average_loss:.4f}')

    trained_model = model
    return trained_model

def validation_function(trained_model, tensors_and_iterable_training_data):
    X_test_tensor = tensors_and_iterable_training_data[1]
    y_test_tensor = tensors_and_iterable_training_data[3]
    trained_model.eval()
    with torch.no_grad():
        predicted_outputs = trained_model(X_test_tensor)
        predicted_labels = (predicted_outputs >= 0.5).squeeze().int()
        correct_predictions = (predicted_labels == y_test_tensor).sum().item()
        total_samples = len(y_test_tensor)
        model_accuracy = (correct_predictions / total_samples) * 100.0
    return model_accuracy

if __name__ == "__main__":

 # converting it to a pandas Dataframe
    task_1a_dataframe = pandas.read_csv('task_1a_dataset.csv')

    # data preprocessing and obtaining encoded data
    encoded_dataframe = data_preprocessing(task_1a_dataframe)
    
    # selecting required features and targets
    features_and_targets = identify_features_and_targets(encoded_dataframe)
    
    # obtaining training and validation data tensors and the iterable
    # training data object
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
    
    
    # model is an instance of the class that defines the architecture of the model
    model = Salary_Predictor()
   
    
    # obtaining loss function, optimizer and the number of training epochs
    loss_function = model_loss_function()
    optimizer = model_optimizer(model)
    number_of_epochs = model_number_of_epochs()
    
    # training the model
    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data,
                                      loss_function, optimizer)

    
    # validating and obtaining accuracy
    model_accuracy = validation_function(
        trained_model, tensors_and_iterable_training_data)
    print(f"Accuracy on the test set = {model_accuracy}")
    X_train_tensor = tensors_and_iterable_training_data[0]
    x = X_train_tensor[0]
    jitted_model = torch.jit.save(torch.jit.trace(
        model, (x)), "task_1a_trained_model.pth")
