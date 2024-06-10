from sklearn import tree

def main():
    print("Demonstration of Supervised Classification Machine Learning")
    print("Ball Classification case study")

    # Feature Encoding
    Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1]]  # list

    # Label Encoding
    Labels = [1,1,2,1,2,1,2,1,1,1]    #list

    # Decide the algorithm
    obj = tree.DecisionTreeClassifier()

    # Train the Model fit method is used to train the model
    obj = obj.fit(Features, Labels)  #(Features are like questions and Labels are like ans)

    # Test the model
    print(obj.predict([[97,0]]))

if __name__ == "__main__":
    main()

# Dataset size : 15
# Traning size : 10
# Testing size

# Output:
# C:\Users\Priyanka\Desktop\Python_2024\ML\08-06-2023>python BallClassification3.py
# Demonstration of Supervised Classification Machine Learning
# Ball Classification case study
# [2]