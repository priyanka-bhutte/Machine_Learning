from sklearn import tree

def MarvellousClassifier():

    # Feature Encoding
    Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1]]  # list

    # Label Encoding
    Labels = [1,1,2,1,2,1,2,1,1,1]    #list

    # Decide the algorithm
    obj = tree.DecisionTreeClassifier()

    # Train the Model fit method is used to train the model
    obj = obj.fit(Features, Labels)  #(Features are like questions and Labels are like ans)

    # Test the model
    ret = obj.predict([[97,0]])

    if ret == 1:
        print("Your object looks like Tennis ball")
    else:
        print("Your object looks like Cricket ball")


def main():
    print("Demonstration of Supervised Classification Machine Learning")
    print("-------------Ball type Classification case study---------------")

    MarvellousClassifier()
    
if __name__ == "__main__":
    main()

