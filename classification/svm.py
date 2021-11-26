# ? SVM:Support-vector machines
from sklearn.svm import SVC
from sklearn.metrics import jaccard_score, f1_score, precision_score, accuracy_score


# Precisión
def analyzeModelF1(prediction, y_test, mode):
    print("Score con metrica F1 con " + mode
          + "->", f1_score(y_test, prediction, average=mode))


def analyzePrecision_score(prediction, y_test, mode):
    print("Medida de la precisión con " + mode + "->", precision_score(y_test, prediction, average=mode))


# Exactitud
def analyzeAccuracy(prediction, y_test):
    print("Medida de la Exactitud ->", accuracy_score(y_test, prediction))


class Svm:
    def __init__(self, x_train, y_train, mode="rbf"):
        # tipos de kernel -> linear poly rbf sigmoid precomputed
        self.model = SVC(kernel=mode)
        # self.model = SVC(kernel=mode, gamma=0.001, C=100)
        self.model.fit(x_train, y_train)
        print('El modelo fue entrenado con exito')

    def analyzeMetric(self, x_test, y_test):
        prediction = self.model.predict(x_test)
        # micro consigue mejores resultados
        # analyzeModelF1(prediction, y_test, 'weighted')
        # analyzeModelF1(prediction, y_test, 'macro')
        analyzeModelF1(prediction, y_test, 'micro')

        # analyzePrecision_score(prediction, y_test, 'weighted')
        # analyzePrecision_score(prediction, y_test, 'macro')
        analyzePrecision_score(prediction, y_test, 'micro')

        analyzeAccuracy(prediction, y_test)
