from sklearn import tree 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

    Accuracy = accuracy_score(target_test, output)
    print("Accuracy is : ",Accuracy*100,"%")

if __name__ == "__main__":
    main()





