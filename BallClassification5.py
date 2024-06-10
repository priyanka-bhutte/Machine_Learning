from sklearn import tree

def MarvellousClassifier(weight,surface):

    # Feature Encoding
    Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1]]  # list

    # Label Encoding
    Labels = [1,1,2,1,2,1,2,1,1,1]    #list

    # Decide the algorithm
    obj = tree.DecisionTreeClassifier()

    # Train the Model fit method is used to train the model
    obj = obj.fit(Features, Labels)  #(Features are like questions and Labels are like ans)

    # Test the model
    ret = obj.predict([[weight,surface]])

    if ret == 1:
        print("Your object looks like Tennis ball")
    else:
        print("Your object looks like Cricket ball")


def main():
    print("Demonstration of Supervised Classification Machine Learning")
    print("-------------Ball type Classification case study---------------")

    print("Please enter the information about the object that you want to test")
    print("Please enter weight of your object in grams")
    no = int(input())

    print("Please mention the type of surface Rough/Smooth")
    data= input()

    if data.lower() == "rough":
        data = 1
    elif data.lower() == "smooth":
        data = 0
    else:
        print("Invalid type of surface")
        exit()
    
    MarvellousClassifier(no,data)
    
if __name__ == "__main__":
    main()


# Output:
# C:\Users\Priyanka\Desktop\Python_2024\ML\09-06-2024>python BallClassification5.py
# Demonstration of Supervised Classification Machine Learning
# -------------Ball type Classification case study---------------
# Please enter the information about the object that you want to test
# Please enter weight of your object in grams
# 45
# Please mention the type of surface Rough/Smooth
# rough
# Your object looks like Tennis ball

# C:\Users\Priyanka\Desktop\Python_2024\ML\09-06-2024>python BallClassification5.py
# Demonstration of Supervised Classification Machine Learning
# -------------Ball type Classification case study---------------
# Please enter the information about the object that you want to test
# Please enter weight of your object in grams
# 98
# Please mention the type of surface Rough/Smooth
# smooth
# Your object looks like Cricket ball

# C:\Users\Priyanka\Desktop\Python_2024\ML\09-06-2024>python BallClassification5.py
# Demonstration of Supervised Classification Machine Learning
# -------------Ball type Classification case study---------------
# Please enter the information about the object that you want to test
# Please enter weight of your object in grams
# 98
# Please mention the type of surface Rough/Smooth
# Smooth
# Your object looks like Cricket ball
