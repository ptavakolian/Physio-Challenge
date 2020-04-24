''' See http://scikit.ml/api/skmultilearn.ensemble.rakelo.html and
    http://scikit.ml/modelselection.html'''

import os
import pandas as pd
import numpy as np
import joblib
import glob, shutil
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, hamming_loss
from sklearn.model_selection import cross_val_score
#import matplotlib.pyplot as plt
from skmultilearn.problem_transform import LabelPowerset
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.ensemble import RakelO
from skmultilearn.adapt import MLkNN, MLTSVM, MLARAM
from evaluate_12ECG_score import evaluate_12ECG_score

folder = r'D:\PhysioNet'
truthFolder = os.path.join(folder, 'testTruth')
predFolder = os.path.join(folder, 'predictions')

# Remove contents from folders
oldFiles = glob.glob(truthFolder + '\*.hea')
for f in oldFiles:
    os.remove(f)
oldFiles = glob.glob(predFolder + '\*.csv')
for f in oldFiles:
    os.remove(f)

# Load the data
dataDF = pd.read_csv(r'D:\PhysioNet\Training_WFDB\PhysioNetDF.csv')
#dataDF = dataDF.sample(n=1000)
labelTypes = ['Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']

class Train_and_Classify:

    def __init__(self, classifierType='MLTSVM', doTrain=True, numPrincComp=25, percToDrop=0.20):
        self.classifierType = classifierType
        self.doTrain = doTrain
        self.numPrincComp = numPrincComp
        self.percToDrop = percToDrop
        self.y_testID = None
        self.sc = None
        self.pca = None

    def prepareTrainAndTestSets(self):
        # Adapted from  https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/

        # Get X and y
        X = dataDF.iloc[:,:250]
        y = dataDF[labelTypes]
        y = pd.concat([y, dataDF['ID']], axis=1)

        # Create training and testing datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        y_train = y_train.drop(['ID'], axis=1)
        self.y_testID = y_test['ID'].reset_index(drop=True)
        y_test = y_test[labelTypes]

        # Apply principal component analysis
        self.pca = PCA(n_components=self.numPrincComp)
        X_train = self.pca.fit_transform(X_train)
        joblib.dump(self.pca, 'pca.sav')
        X_test = self.pca.transform(X_test)

        # Apply scaling
        self.sc = StandardScaler()
        X_train = self.sc.fit_transform(X_train)
        joblib.dump(self.sc, 'scaler.sav')
        X_test = self.sc.transform(X_test)
        return X_train, X_test, y_train.to_numpy(), y_test.to_numpy()

    def getClassifier(self):
        if self.classifierType.lower() == 'rakelo':
            classifier = RakelO(
                base_classifier=LabelPowerset(GaussianNB()),
                #base_classifier_require_dense=[True, True],
                model_count=10,
                labelset_size=2 #len(labelTypes) // 4
            )
        elif self.classifierType.lower() == 'mlknn':
            classifier = MLkNN(k=3)
        # elif self.classifierType.lower() == 'mltsvm':
        #     classifier = MLTSVM(c_k = 2**-1)
        elif self.classifierType.lower() == 'mlaram':
            classifier = MLARAM()
        elif self.classifierType.lower() == 'labelpowerset':
            classifier = LabelPowerset(
                        classifier = RandomForestClassifier(n_estimators=100),
                        require_dense = [False, True]
                        )
        return classifier

    def trainModel(self, X_train, y_train):
        if self.doTrain:
            classifier = self.getClassifier()
            classifier.fit(X_train, y_train)
            joblib.dump(classifier, open(self.classifierType+'.sav', 'wb'))
        else:
            classifier = joblib.load(open(self.classifierType+'.sav', 'rb'))
        return classifier

    def predictClass(self, classifier, testSet):
        y_pred = classifier.predict(testSet)
        return y_pred

    def predictProbability(self, classifier, testSet):
        proba = classifier.predict_proba(testSet)
        return proba

    def writeProbsToFile(self, probMatrix, y_test, y_pred):
        for i in range(probMatrix.get_shape()[0]):
            with open(os.path.join(predFolder, self.y_testID[i]+'.csv'), 'w') as f:
                f.write(self.y_testID[i]+'\n')  # recording_output
                f.write(','.join(labelTypes)+'\n') # classes_output
                f.write(','.join(map(str, y_pred[i,:].toarray().tolist()[0]))+'\n') # single_recording_output
                f.write(','.join(map(str, probMatrix[i,:].toarray().tolist()[0]))) # single_probabilities_output
            shutil.copyfile(os.path.join(folder, 'Training_WFDB', self.y_testID[i]+'.hea'),
                            os.path.join(folder, 'testTruth', self.y_testID[i]+'.hea'))
        return

    def calcScores(self, truthFolder, predFolder, scoresCsv=True):
        auroc, auprc, accuracy, f_measure, f_beta, g_beta = evaluate_12ECG_score(truthFolder, predFolder)
        output_string = 'AUROC|AUPRC|Accuracy|F-measure|Fbeta-measure|Gbeta-measure\n{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}'.format(
            auroc, auprc, accuracy, f_measure, f_beta, g_beta)
        if scoresCsv:
            with open(os.path.join(folder, 'results', self.classifierType+'_scores.csv'), 'w') as f:
                f.write(output_string)
        else:
            print(output_string)
        return

    def calcClassifierAccuracy(self, y_test, y_pred):
        print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
        print('Jaccard Score: %.2f' % jaccard_score(y_test, y_pred, average='micro'))
        print('Precision: %.2f' % precision_score(y_test, y_pred, average='micro'))
        print('Recall: %.2f' % recall_score(y_test, y_pred, average='micro'))
        print('F1 Score: %.2f' % f1_score(y_test, y_pred, average='micro'))
        print('Hamming Loss: %.2f' % hamming_loss(y_test, y_pred))
        return

    def calcCrossValScore(self, classifier, X_train, X_test, y_train, y_test, kFolds=5):
        X = np.append(X_train, X_test, axis=0)
        y = np.append(y_train, y_test, axis=0)
        scores = cross_val_score(classifier, X, y, cv=kFolds)
        print('Cross validation results: %.4f (+/- %.4f))' % (np.mean(scores), np.std(scores)))
        return

    def calcMedian(self, dataDF):
        medianBeat = dataDF.median()
        return medianBeat

    def removeOutliersFromIndividual(self, dataDF):
        medianBeat = self.calcMedian(dataDF)
        dataMinusMedianDF = dataDF.sub(medianBeat)
        dataMinusMedianDF['norm'] = np.linalg.norm(dataMinusMedianDF.values, axis=1)
        dataMinusMedianDF.sort_values(by=['norm'], inplace=True)
        numDropRows = int(np.round(self.percToDrop * len(dataMinusMedianDF)))
        dataMinusMedianDF = dataMinusMedianDF[:-numDropRows]
        return dataMinusMedianDF

    # def plotBeats(self, beatsDF, patient, numPlots=10):
    #     ''' Plots the segmented beats individually '''
    #     plt.figure()
    #     beatsDF = beatsDF.sample(frac=1)
    #     for i in range(numPlots):
    #         beat = beatsDF.iloc[i]
    #         plt.plot(range(len(beat)), beat)
    #         plt.title(patient)
    #     plt.savefig(patient + '.png')
    #     return


############################################################################
if __name__ == '__main__':
    # Create the class object
    tc = Train_and_Classify(classifierType='MLkNN', doTrain=True)

    # Prepare data for training
    print('Preparing datasets...')
    X_train, X_test, y_train, y_test = tc.prepareTrainAndTestSets()

    # Train the model
    print('Training the model...')
    model = tc.trainModel(X_train, y_train)

    # Predict classifications for the test set
    print('Predicticing labels...')
    y_pred = tc.predictClass(model, X_test)

    # Calculate the classifier accuracy for a single train/test split
    print('Calculating metrics...')
    tc.calcClassifierAccuracy(y_test, y_pred)

    # Calculate the probabilities for each label
    print('Predicting probabilities...')
    probMatrix = tc.predictProbability(model, X_test)

    # Write probabilities to output
    print('Writing probabilities and truth to file...')
    tc.writeProbsToFile(probMatrix, y_test, y_pred)

    # Conduct official evaluation
    print('Evaluating scores...')
    tc.calcScores(os.path.join(folder,'testTruth'), os.path.join(folder,'predictions'), scoresCsv=True)

    # # Perform cross validation
    print('Performing cross validation...')
    tc.calcCrossValScore(tc.getClassifier(), X_train, X_test, y_train, y_test)