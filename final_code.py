import numpy as np
import rasterio
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from skopt import BayesSearchCV

# Define a function to read and resize TIF files
def read_and_resize_tif(tif_path, target_shape):
    with rasterio.open(tif_path) as src:
        data = src.read(
            out_shape=(src.count, *target_shape),
            resampling=rasterio.enums.Resampling.bilinear
        )
        profile = src.profile
    return data.flatten(), profile

# Define paths to your TIF files
tif_files = {
    "Distance_from Road_Reclass1": r"C:\Users\vasuc\Desktop\UG Project\Flood Susceptibility Mapping\Data\Distance_from Road_Reclass1.tif",
    "DD_Reclass1": r"C:\Users\vasuc\Desktop\UG Project\Flood Susceptibility Mapping\Data\DD_Reclass1.tif",
    "Distance_from_river_Reclass1": r"C:\Users\vasuc\Desktop\UG Project\Flood Susceptibility Mapping\Data\Distance_from_river_Reclass1.tif",
    "Elevation_Reclass1": r"C:\Users\vasuc\Desktop\UG Project\Flood Susceptibility Mapping\Data\Elevation_Reclass1.tif",
    "LULC_Reclassify1": r"C:\Users\vasuc\Desktop\UG Project\Flood Susceptibility Mapping\Data\LULC_Reclassify1.tif",
    "NDVI_Reclass1": r"C:\Users\vasuc\Desktop\UG Project\Flood Susceptibility Mapping\Data\NDVI_Reclass1.tif",
    "Slope_Reclass1": r"C:\Users\vasuc\Desktop\UG Project\Flood Susceptibility Mapping\Data\Slope_Reclass1.tif",
    "SPI1": r"C:\Users\vasuc\Desktop\UG Project\Flood Susceptibility Mapping\Data\SPI1.tif",
    "TPI1": r"C:\Users\vasuc\Desktop\UG Project\Flood Susceptibility Mapping\Data\TPI1.tif",
    "TWI_Reclass1": r"C:\Users\vasuc\Desktop\UG Project\Flood Susceptibility Mapping\Data\TWI_Reclass1.tif"
}



# Read and resize all TIF files into a dictionary
data = {}
first_tif_shape = None
for factor, path in tif_files.items():
    try:
        if first_tif_shape is None:
            # Determine the target shape based on the first TIF file
            first_tif_data, _ = read_and_resize_tif(path, (100, 100))
            first_tif_shape = first_tif_data.shape
        data[factor], _ = read_and_resize_tif(path, (100, 100))
    except Exception as e:
        print(f"Error reading or resizing TIF file for factor `{factor}`: {e}")

# Generate a placeholder target variable (replace this with your actual labels if available)
num_samples = min(data[key].shape[0] for key in data)
y = np.random.randint(2, size=num_samples)  # Binary classification, replace with your actual labels

# Trim data to have the same number of samples
for key in data:
    data[key] = data[key][:num_samples]

# Stack data into a single array
X = np.stack([data[factor] for factor in tif_files.keys()], axis=1)

# Scale features to a given range using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define your SVM classifier
svm_classifier = SVC(kernel='linear')

# Define the parameter grid for Bayesian optimization
paramgrid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 10],
}

# Define the Bayesian optimization method
cv = BayesSearchCV(
    estimator=svm_classifier,
    search_spaces=paramgrid,
    scoring="accuracy",
    cv=5,
    n_iter=10,  # Number of iterations for optimization
    verbose=1,
    n_jobs=-1  # Use all available CPUs
)

# Fit the Bayesian optimization method
cv.fit(X_scaled, y)

# Get the best SVM classifier after optimization
best_svm_classifier = cv.best_estimator_

# Get feature importance scores (weights) from the optimized SVM classifier
weights = np.abs(best_svm_classifier.coef_.flatten()) / np.sum(np.abs(best_svm_classifier.coef_))

# Print weights of each factor
for factor, weight in zip(tif_files.keys(), weights):
    print(f"{factor}: {weight:.4f}")
