from sklearn import tree 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def main():
    print("----------Iris Flower Classification Study--------------")

    iris = load_iris()  # loading inbuild dataset using API(Function)

    print(iris)
    #print(type(data))

    Features = iris.data
    Labels = iris.target

    data_train, data_test, target_train, target_test = train_test_split(Features,Labels,test_size=0.5)

    obj = tree.DecisionTreeClassifier()

    obj.fit(data_train, target_train)

    output = obj.predict(data_test)

    print(output)

if __name__ == "__main__":
    main()


# Output:

# [2 0 2 0 2 2 2 0 0 2 0 2 2 2 1 1 0 0 0 1 1 1 1 1 0 1 0 1 2 2 0 2 2 2 0 1 1
#  1 0 1 1 1 0 2 2 2 1 1 0 0 1 2 0 0 0 2 0 1 2 1 0 2 1 0 1 1 0 1 0 0 0 2 1 0
#  0]



