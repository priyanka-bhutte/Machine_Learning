from sklearn import tree 
from sklearn.datasets import load_iris

def main():
    print("----------Iris Flower Classification Study--------------")

    iris = load_iris()  # loading inbuild dataset using API(Function)

    print(iris)
    #print(type(data))

    Features = iris.data
    Label = iris.target

    print("Features are :")
    print(Features)

    print("Labels are : ")
    print(Label)




if __name__ == "__main__":
    main()

# shape function is used to get the dimensions

#label encoding will be done
# features encoding is not require becoz features are vailable in numeric format
# iris dataset is inbuilt available in sklearn library