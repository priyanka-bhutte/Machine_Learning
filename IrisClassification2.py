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



if __name__ == "__main__":
    main()



# shape function is used to get the dimensions
#label encoding will be done
# features encoding is not require becoz features are vailable in numeric format
# iris dataset is inbuilt available in sklearn library
# we have 150 size of data set now split that data set into 75/75 test_size=0.5