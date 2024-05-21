import pandas as pd
from fastai.tabular.all import load_learner

# Define the custom accuracy metric function (if it was used)
def accuracy_metric(inp, targ):
    return (inp.argmax(dim=1) == targ).float().mean()

# Load the model from the .pkl file
learn = load_learner('Hascore.pkl')

# Load the new data from the .csv file
new_data_df = pd.read_csv('result_features.csv')

# Display the first few rows of the new data
print(new_data_df.head())

# Define the prediction function
def predict_similarity(learn, df):
    """
    Predict the similarity for each row in the dataframe.
    """
    # Ensure the DataLoader for the test data is created
    dl = learn.dls.test_dl(df)
    
    # Debugging: Check the content of DataLoader
    print("DataLoader:", dl)
    
    # Get predictions and probabilities
    preds, probs = learn.get_preds(dl=dl)
    
    # Debugging: Check the output of get_preds
    print("Predictions:", preds)
    print("Probabilities:", probs)
    
    # If probabilities are None, convert logits to probabilities using sigmoid
    if probs is None:
        probs = preds.sigmoid()
        preds = (probs > 0.5).long()
    
    # Check the output of get_preds
    if preds is None or probs is None:
        raise ValueError("Predictions or probabilities are None. Please check the model and DataLoader.")
    
    return preds, probs

# Extract the features needed for prediction
cont_names = [
    'intermolecular_contacts',
    'charged_charged_contacts',
    'charged_polar_contacts',
    'charged_apolar_contacts',
    'polar_polar_contacts',
    'apolar_polar_contacts',
    'apolar_apolar_contacts'
]

# Ensure that the model is in evaluation mode
learn.model.eval()

# Make predictions on the new data
predictions, probabilities = predict_similarity(learn, new_data_df[cont_names])

# Convert predictions to class labels
predicted_labels = predictions.numpy()

# Add predictions and probabilities to the dataframe
new_data_df['prediction'] = predicted_labels
new_data_df['probabilities'] = [prob.max().item() for prob in probabilities]

# Display the first few rows of the dataframe with predictions
print(new_data_df.head())

# Save the predictions to a new CSV file
new_data_df.to_csv('predictions.csv', index=False)

