import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
import os

# Read the data
data = pd.read_csv("Mall_Customers.csv")

# Display basic information about the data
print(data.head())
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Select features for clustering
features = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Scale each feature separately
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Convert the scaled features back to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

# Apply KMeans clustering
model = KMeans(n_clusters=4, random_state=0)
clusters = model.fit_predict(scaled_features)
data["clusters"] = clusters

d_age = data[data.clusters == 0]
d_income = data[data.clusters == 1]
d_score = data[data.clusters == 2]

# Plot the scatter plot
plt.scatter(data["Age"], data["Annual Income (k$)"], c=data["clusters"], cmap='viridis')
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
plt.title("KMeans Clustering")
plt.show()

# Elbow Method for Optimal K
k_rng = range(1, 15)

sse = []

for k in k_rng:
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(scaled_features)
    sse.append(km.inertia_)

# Plot the Elbow Method
plt.plot(k_rng, sse, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal K')
plt.show()

# Prediction part

age = int(input("Enter age "))
income = int(input("Enter Income "))
spending = int(input("Enter spending "))
d = [[age, income, spending]]
nd = scaler.transform(d)
satisfaction = model.predict(nd)
if satisfaction == 0:
    print("Customer has SILVER MEMBERSHIP.")
elif satisfaction == 1:
    print("Customer has BRONZE MEMBERSHIP")
else:
    print("Customer has GOLD MEMBERSHIP.")

model_path = os.path.join(os.getcwd(), "pc.model")

f = None
try:
    f = open(model_path, "wb")
    pickle.dump(model, f)
except Exception as e:
    print("Issue:", e)
finally:
    if f is not None:
        f.close()
