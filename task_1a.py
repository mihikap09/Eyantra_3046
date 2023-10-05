'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:			task_1a.py
# Functions:	    [`ideantify_features_and_targets`, `load_as_tensors`,
# 					 `model_loss_function`, `model_optimizer`, `model_number_of_epochs`, `training_function`,
# 					 `validation_functions` ]

####################### IMPORT MODULES #######################
import torch.nn as nn
import pandas
import torch
import numpy
###################### Additional Imports ####################
'''
You can import any additional modules that you require from 
torch, matplotlib or sklearn. 
You are NOT allowed to import any other libraries. It will 
cause errors while running the executable
'''
##############################################################
################# ADD UTILITY FUNCTIONS HERE #################
##############################################################
def data_preprocessing(task_1a_dataframe):
    ''' 
    Purpose:
    This function will be used to load your csv dataset and preprocess it.
    Preprocessing involves cleaning the dataset by removing unwanted features,
    decision about what needs to be done with missing values etc. Note that 
    there are features in the csv file whose values are textual (eg: Industry, 
    Education Level etc)These features might be required for training the model
    but can not be given directly as strings for training. Hence this function 
    should return encoded dataframe in which all the textual features are 
    numerically labeled.
    Input Arguments:
    `task_1a_dataframe`: [Dataframe]
                                              Pandas dataframe read from the provided dataset 	
    Returns:
    `encoded_dataframe` : [ Dataframe ]
                                              Pandas dataframe that has all the features mapped to 
                                              numbers starting from zero

    Example call:
    encoded_dataframe = data_preprocessing(task_1a_dataframe)
    '''

    ################# ADD YOUR CODE HERE	##################
    from sklearn.preprocessing import LabelEncoder
    encoded_dataframe = task_1a_dataframe.copy()
    encoded_dataframe['Age'] = ((encoded_dataframe['Age'])-20)//5
    encoded_dataframe['Age'] = ((encoded_dataframe['Age'])-20)//5
    encoded_dataframe['JoiningYear'] = ((2023- encoded_dataframe['JoiningYear'])//3)-1
    print(encoded_dataframe)
    label_encoder = LabelEncoder()
    encoded_dataframe['Education'] = label_encoder.fit_transform(encoded_dataframe['Education'])
    encoded_dataframe['Gender'] = label_encoder.fit_transform(encoded_dataframe['Gender'])
    encoded_dataframe['City'] = label_encoder.fit_transform(encoded_dataframe['City'])
    encoded_dataframe['EverBenched'] = label_encoder.fit_transform(encoded_dataframe['EverBenched'])
    encoded_dataframe['JoiningYear'] = label_encoder.fit_transform(encoded_dataframe['JoiningYear'])
    encoded_dataframe.fillna(-1, inplace=True)
    return encoded_dataframe

    ##########################################################
def identify_features_and_targets(encoded_dataframe):
    df = encoded_dataframe
    '''
	Purpose:
	The purpose of this function is to define the features and
	the required target labels. The function returns a python list
	in which the first item is the selected features and second 
	item is the target label
	Input Arguments:
	`encoded_dataframe` : [ Dataframe ]
						Pandas dataframe that has all the features mapped to 
						numbers starting from zero
	Returns:
	`features_and_targets` : [ list ]
							python list in which the first item is the 
							selected features and second item is the target label

	Example call:
	features_and_targets = identify_features_and_targets(encoded_dataframe)
	'''

    ################# ADD YOUR CODE HERE	##################
    # Assuming that the target column is the last column in the DataFrame
    target_label_name = df.columns[-1]
    target_label = encoded_dataframe[target_label_name]
    # Selecting all columns except the last one as features
    features_name = df.columns[:-1]
    features = encoded_dataframe[features_name]
    # Creating a list with features and target label
    features_and_targets = [features, target_label]

    #########################################################
    return features_and_targets


def load_as_tensors(features_and_targets):
    ''' 
    Purpose:
    This function aims at loading your data (both training and validation)
    as PyTorch tensors. Here you will have to split the dataset for training 
    and validation, and then load them as as tensors. 
    Training of the model requires iterating over the training tensors. 
    Hence the training sensors need to be converted to iterable dataset
    object.
    Input Arguments:
    `features_and targets` : [ list ]
        python list in which the first item is the 
        selected features and second item is the target label
    Returns:
    `tensors_and_iterable_training_data` : [ list ]
        Items:
        [0]: X_train_tensor: Training features loaded into Pytorch array
        [1]: X_test_tensor: Feature tensors in validation data
        [2]: y_train_tensor: Training labels as Pytorch tensor
        [3]: y_test_tensor: Target labels as tensor in validation data
        [4]: Iterable dataset object and iterating over it in 
                    batches, which are then fed into the model for processing.

    Example call:

    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
    '''

    ################# ADD YOUR CODE HERE	##################
    from torch.utils.data import TensorDataset, DataLoader
    features = features_and_targets[0].to_numpy()
    targets = features_and_targets[1].to_numpy()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, shuffle=True)
    # Convert NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    # Create TensorDataset for training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # Create DataLoader for training
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # Return the tensors and iterable training data
    tensors_and_iterable_training_data = [
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader]
    ##########################################################
    return tensors_and_iterable_training_data


class Salary_Predictor(nn.Module):
    '''
    Purpose:
    The architecture and behavior of your neural network model will be
    defined within this class that inherits from nn.Module. Here you
    also need to specify how the input data is processed through the layers. 
    It defines the sequence of operations that transform the input data into 
    the predicted output. When an instance of this class is created and data
    is passed through it, the `forward` method is automatically called, and 
    the output is the prediction of the model based on the input data.
    Returns:
    `predicted_output` : Predicted output for the given input data
    '''
    def __init__(self):
        import torch.nn as nn
        super(Salary_Predictor, self).__init__()
        
		# Define the type and number of layers
        ####### ADD YOUR CODE HERE	#######
        self.fc1 = nn.Linear(8, 16)  # Input size: 8, Hidden size: 16
        self.sigmoid = nn.Sigmoid()  # Activation function (Sigmoid)
        self.fc2 = nn.Linear(16, 1)  # Hidden size: 16, Output size: 1 (binary)
        ###################################

    def forward(self, x):
        #Define the activation functions
        ####### ADD YOUR CODE HERE	#######
        x = self.fc1(x)
        x = self.sigmoid(x)  # Apply sigmoid activation
        predicted_output = self.fc2(x)
        ###################################

        return predicted_output


def model_loss_function():
    import torch.nn as nn
    '''
	Purpose:
	To define the loss function for the model. Loss function measures 
	how well the predictions of a model match the actual target values  in training data.
	Input Arguments: None
	Returns: 	`loss_function`: This can be a pre-defined loss function in PyTorch or can be user-defined
	Example call:
	loss_function = model_loss_function()
	'''
    ################# ADD YOUR CODE HERE	##################
    # we need to decide which loss function is best, or should we define it ourselves
    loss_function = nn.MSELoss()
    ##########################################################
    return loss_function


def model_optimizer(model):
    import torch.optim as optim
    '''
	Purpose:

	To define the optimizer for the model. Optimizer is responsible 
	for updating the parameters (weights and biases) in a way that 
	minimizes the loss function.
	
	Input Arguments:

	`model`: An object of the 'Salary_Predictor' class
	Returns:
	`optimizer`: Pre-defined optimizer from Pytorch
	Example call:
	optimizer = model_optimizer(model)
	'''
    ################# ADD YOUR CODE HERE	##################
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    learning_rate = 0.01  # Set your desired learning rate
    optimizer = optim.SGD(parameters, lr=learning_rate, momentum=0.8)

    ##########################################################

    return optimizer
def model_number_of_epochs():
    '''
    Purpose:
    To define the number of epochs for training the model
    Input Arguments:
    None
    Returns:
    `number_of_epochs`: [integer value]
    Example call:
    number_of_epochs = model_number_of_epochs()
    '''
    ################# ADD YOUR CODE HERE	##################
    # this is a hyper-param and it's value should be changed depending on model's performance.
    number_of_epochs = 120
    ##########################################################
    return number_of_epochs


def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    '''
    Purpose:
    All the required parameters for training are passed to this function.
    Input Arguments:
    1. `model`: An object of the 'Salary_Predictor' class
    2. `number_of_epochs`: For training the model
    3. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
    and iterable dataset object of training tensors
    4. `loss_function`: Loss function defined for the model
    5. `optimizer`: Optimizer defined for the model
    Returns:
    trained_model
    Example call:
    trained_model = training_function(model, number_of_epochs, iterable_training_data, loss_function, optimizer)
    '''
    ################# ADD YOUR CODE HERE	##################
    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0.0
    # Iterate over the training data in batches
        for inputs, labels in tensors_and_iterable_training_data[4]:
            labels = labels.to(torch.float32)
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs).squeeze(1)
            outputs = outputs.to(torch.float32)
            # Compute the loss
            loss = loss_function(outputs, labels)
            # Backpropagation
            loss.backward()
            # Update model parameters
            optimizer.step()
            total_loss += loss.item()
    # Calculate and print the average loss for this epoch
        average_loss = total_loss/len(tensors_and_iterable_training_data[4])
        print(
            f'Epoch [{epoch + 1}/{number_of_epochs}] - Loss: {average_loss:.4f}')
    trained_model = model
    ##########################################################
    return trained_model
def validation_function(trained_model, tensors_and_iterable_training_data):
    '''
    Purpose:
    This function will utilise the trained model to do predictions on the
    validation dataset. This will enable us to understand the accuracy of
    the model.
    Input Arguments:
    1. `trained_model`: Returned from the training function
    2. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
    and iterable dataset object of training tensors
    Returns:
    model_accuracy: Accuracy on the validation dataset
    Example call:
    model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)
    '''
    ################# ADD YOUR CODE HERE	##################
    X_test_tensor = tensors_and_iterable_training_data[1]
    y_test_tensor = tensors_and_iterable_training_data[3]
    # Ensure the model is in evaluation mode
    trained_model.eval()
# Perform predictions on the validation data
    with torch.no_grad():
        # Forward pass
        predicted_outputs = trained_model(X_test_tensor)
    # Convert predicted outputs to class labels
        predicted_labels = (predicted_outputs >= 0.5).squeeze().int()
    # Calculate accuracy
        correct_predictions = (predicted_labels == y_test_tensor).sum().item()
        total_samples = len(y_test_tensor)
        model_accuracy = (correct_predictions / total_samples) * 100.0

        ##########################################################

    return model_accuracy
########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########
'''
	Purpose:
	The following is the main function combining all the functions
	mentioned above. Go through this function to understand the flow
	of the script
'''
if __name__ == "__main__":

    # reading the provided dataset csv file using pandas library and
    # converting it to a pandas Dataframe
    task_1a_dataframe = pandas.read_csv('task_1a_dataset.csv')

    # data preprocessing and obtaining encoded data
    encoded_dataframe = data_preprocessing(task_1a_dataframe)
    print(encoded_dataframe)
    # selecting required features and targets
    features_and_targets = identify_features_and_targets(encoded_dataframe)
    # obtaining training and validation data tensors and the iterable
    # training data object
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
    print('ran successfully')
    # model is an instance of the class that defines the architecture of the model
    model = Salary_Predictor()
    print("ran successfully")
    # obtaining loss function, optimizer and the number of training epochs
    loss_function = model_loss_function()
    optimizer = model_optimizer(model)
    number_of_epochs = model_number_of_epochs()
    # training the model
    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data,
                                      loss_function, optimizer)

    print("ran successfully")
    # validating and obtaining accuracy
    model_accuracy = validation_function(
        trained_model, tensors_and_iterable_training_data)
    print(f"Accuracy on the test set = {model_accuracy}")
    X_train_tensor = tensors_and_iterable_training_data[0]
    x = X_train_tensor[0]
    jitted_model = torch.jit.save(torch.jit.trace(
        model, (x)), "task_1a_trained_model.pth")
