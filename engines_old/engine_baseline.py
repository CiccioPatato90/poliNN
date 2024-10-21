from imports import *

file_path = "/Volumes/documenti/polimi_outputs"

# keras + tensorflow

# Load the dataset

# Step 1: Fetch data from SQLite and convert to NumPy arrays
db = Database('res/records.db')
data = db.fetch_all()
num_records = len(data)
db.close_conn()

data = pd.DataFrame(data=data)

# media, std_dev, correlazione variabili, n_elem x class, distr var orig, distr var scaled, corr variabili input e il loro rispettivo output

print(data.head())

# Split data into features and labels
y = data.iloc[:, -1]  # Select the last column
X = data.iloc[:, :-1]

print(X.dtypes)
print(y.dtypes)

print("X:HEAD:")
print(X.head())
print("Y:HEAD:")
print(y.head())

#190k class 0 >>>> class 1, 2, 3
#Evaluate.histogram(y)

# Normalize the features
# Gaussian distribution with mean 0 and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels, ONE HOT ENCODING
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
#print(y_encoded)

# Split the dataset into training and testing sets
# fixing random state for reproducible output

#controllare che mantenga la proporzione fra le classi
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Using long for class indices
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]  # Number of features
output_dim = len(label_encoder.classes_)  # Number of classes

model = ClusteringModel(input_dim, output_dim)

# usare pesi per bilanciare le classi piuttosto che usare SMOTE
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification

# default choice for Classification tasks, might try also SGD
optimizer = optim.Adam(model.parameters(), lr=0.001)

# example of oversampling a multi-class classification dataset

# transform the dataset
#oversample = SMOTE(k_neighbors=4)
#X, y = oversample.fit_resample(X_train_tensor, y_train_tensor)


#exit()


# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        total_loss += loss.item()
    
    average_loss = total_loss / len(train_loader)
    
    # Print training progress
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")



def evaluate_accuracy(model, X_test_tensor, y_test_tensor):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(X_test_tensor)  # Get model outputs
        # capire come funziona sotto
        _, predicted_enc = torch.max(outputs, dim=1)  # Get predicted classes
        predicted = label_encoder.inverse_transform(predicted_enc)
        print("Pred:", predicted)

        print("y_tensor:", y_test_tensor)
        #results = pd.DataFrame(predicted == y_test_tensor)
        #results.to_csv("../../Desktop/results.csv")
        accuracy = (predicted == y_test_tensor).float().mean()  # Calculate accuracy
    return [accuracy.item(), predicted]

# CAPIRE I FALSI NEGATIVI E FALSI POSITIVI
# VALIDITA' DEI LABEL

# Calculate accuracy on the test set
[accuracy, pred] = evaluate_accuracy(model, X_test_tensor, y_test_tensor)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test_tensor, pred)
print(cm)
