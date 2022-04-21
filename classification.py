from datasets import temporary_assignment
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def ridge_classification(X_train, X_test, y_train, y_test):
    '''
    Carries out the classification using a basic ridge classifier.
    '''
    start = time.time()

    # Little test using a ridge classifier
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train, y_train)

    end = time.time()

    score = classifier.score(X_test, y_test)
    print('Ridge score: ', score)

    elapsed = end - start

    print('Ridge time: ', elapsed)

    return score, elapsed



def decision_tree_classification(X_train, X_test, y_train, y_test):
    '''
    Carries out the classification using a decision tree and finds the best depth.
    '''
    accuracies =[]
    depths = []
    '''
    for depth in range(1,15):

        dec_tree_clf_reg = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth = depth )
        dec_tree_clf_reg = dec_tree_clf_reg.fit(X_train, y_train)

        predictions = dec_tree_clf_reg.predict(X_test)

        acc = accuracy_score(y_test, predictions)

        accuracies.append(acc)
        depths.append(depth)
        
        print('Accuracy is %.3f with depth = '% acc + str(depth) )
    

    idx = accuracies.index(max(accuracies))
    '''
    #print('\nBest accuracy is %.3f with depth = '% max(accuracies) + str(depths[idx]) )

    dec_tree_clf_reg = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth = 4 )
    dec_tree_clf_reg = dec_tree_clf_reg.fit(X_train, y_train)

    test_acc_dec_tree_reg = dec_tree_clf_reg.score(X_test, y_test)

    print('Tree score: ', test_acc_dec_tree_reg)

    return test_acc_dec_tree_reg



def random_forest_classification(X_train, X_test, y_train, y_test):
    '''
    Carries out the classification using a random forest.
    Could increase the performance by tuning the hyperparams
    '''
    start = time.time()

    # Little test using a ridge classifier
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    end = time.time()

    score = classifier.score(X_test, y_test)
    print('Forest score: ', score)

    elapsed = end - start

    print('Forest time: ', elapsed)

    return score, elapsed


def template_classification(classifier, name_string, X_train, X_test, y_train, y_test):
    start = time.time()

    # Little test using a ridge classifier
    classifier.fit(X_train, y_train)

    end = time.time()

    score = classifier.score(X_test, y_test)
    print(name_string+' score: ', score)

    elapsed = end - start

    print(name_string+' time: ', elapsed)

    return score, elapsed





def main():

    # Load in the data
    X_test = pd.read_csv('data/features/X_test.csv')
    X_train = pd.read_csv('data/features/X_train.csv')

    y_train = pd.read_csv('data/features/y_train.csv').iloc[:,1]
    y_test = pd.read_csv('data/features/y_test.csv').iloc[:,1]

    # Try classifiers
    ridge_score, ridge_time = ridge_classification(X_train, X_test, y_train, y_test)
    print('\n')
    tree_score = decision_tree_classification(X_train, X_test, y_train, y_test)
    print('\n')
    forest_score, forest_time = random_forest_classification(X_train, X_test, y_train, y_test)
    print('\n')
    ada_score, ada_time = template_classification(AdaBoostClassifier(), 'ada', X_train, X_test, y_train, y_test)
    print('\n')
    ada_score, ada_time = template_classification(ExtraTreesClassifier(), 'extra trees', X_train, X_test, y_train, y_test)
    print('\n')
    ada_score, ada_time = template_classification(BaggingClassifier(), 'bagging', X_train, X_test, y_train, y_test)
    print('\n')
    #ada_score, ada_time = template_classification(GradientBoostingClassifier(), 'grad_boost', X_train, X_test, y_train, y_test)
    print('\n')
    ada_score, ada_time = template_classification(MLPClassifier(max_iter=300), 'mlp', X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()

