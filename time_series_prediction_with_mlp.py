
from copy import deepcopy
from random import randint
import matplotlib.pyplot as plt
import math
import random

A = float(input("Entrer la valeur de A :"))
n = 50


def product(w, x):
    vector = list()
    for i in range(len(x)):
        vector.append(x[i]*w[i])
    matrix_product = 0
    for value in vector:
        matrix_product += value

    return matrix_product


def sigmoid(matrix_product):

    expo = (math.e**(-1*matrix_product))
    result = float(1/(1+expo))

    return result


class Node():
    """docstring for [object Object]."""

    def __init__(self):
        pass

    def calculate_output_node(self, x, final=False):
        if not(hasattr(self, "weights")):
            weights = list()
            for e in x:
                weights.append(random.uniform(-1, 1))
            self.weights = weights
        matrix_product = product(self.weights, x)
        if final:
            # print(matrix_product)
            return matrix_product
        else:
            return sigmoid(matrix_product)


class InputNode(Node):
    def __init__(self):
        pass


class OutputNode(Node):
    def __init__(self, y=""):
        pass


class OutputNodeException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class Layer(object):
    """The class for one layer in the neural network"""

    def __init__(self, hm_nodes=""):
        if hasattr(self, "nodes"):
            hm_nodes = len(self.nodes)

        if hm_nodes:
            self.hm_nodes = hm_nodes
            nodes = list()
            for i in range(hm_nodes):
                nodes.append(Node())
            self.nodes = nodes

    def set_next_layer(self, layer):
        self.next_layer = layer
        layer.previous_layer = self
# calcul gradient

    def set_current_deltas(self):
        for j, node in enumerate(self.nodes):
            sum_of_weight_j_dot_delta_j = 0
            for next_node in self.next_layer.nodes:
                #    print(self.next_layer.nodes[0].__dict__)
                sum_of_weight_j_dot_delta_j = sum_of_weight_j_dot_delta_j + \
                    next_node.weights[j]*next_node.delta
            node.delta = node.output*(1-node.output) * \
                sum_of_weight_j_dot_delta_j


class OutputLayer(Layer):
    def __init__(self, layer):

        if hasattr(layer, "previous_layer"):
            self.previous_layer = layer.previous_layer
        if hasattr(layer, "nodes"):
            self.nodes = layer.nodes

    def set_current_deltas(self):
        for node in self.nodes:
            # print(node.__dict__)
            if not(hasattr(node, 'y')):
                raise OutputNodeException("output node don't has attribute y")
            else:
                node.delta = node.y-node.output


class HiddenLayer(Layer):
    def __init__(self, layer):
        super(HiddenLayer, self)


class InputLayer(Layer):
    def __init__(self, layer):
        nodes = list()
        if hasattr(layer, "next_layer"):
            self.next_layer = layer.next_layer
        if hasattr(layer, "nodes"):
            for i in range(len(layer.nodes)):
                nodes.append(InputNode())
            self.nodes = nodes

    def set_next_layer(self, layer):
        self.next_layer = layer


class Network(object):
    """The famous Network class"""

    def __init__(self, layers):

        input_layer = Layer()
        output_layer = Layer(hm_nodes=layers[-1])

        for hm_nodes in layers[1:-1]:
            hidden_layer = Layer(hm_nodes=hm_nodes)
            input_layer.set_next_layer(hidden_layer)
        hidden_layer.set_next_layer(output_layer)

        self.input_layer = input_layer
        self.output_layer = output_layer

    def set_input_layer(self, layer):
        self.input_layer = InputLayer(layer)

    def set_output_layer(self, layer):
        self.output_layer = OutputLayer(layer)

    def predict(self, inputs, y="", classe=True):

        nb_features = len(inputs)
        if not(hasattr(self, 'output_layer')):
            raise OutputNetworkException("network must has a output layer")
        elif not(hasattr(self, 'output_layer')):
            raise OutputNetworkException("network must has a input layer")
        if hasattr(self.input_layer, "nodes"):
            if len(inputs) != len(self.input_layer.nodes):
                raise InputNetworkException(
                    "features's number different of nodes's number in input layer")

        elif not(hasattr(self.input_layer, "nodes")):
            nodes = list()
            for i in range(nb_features):
                nodes.append(InputNode())
            self.input_layer.nodes = nodes

        current_layer = self.input_layer.next_layer
        current_inputs = inputs

        while hasattr(current_layer, "next_layer"):
            outputs = list()
            for i, node in enumerate(current_layer.nodes):
                node.output = node.calculate_output_node(current_inputs)
                outputs.append(node.output)

            current_inputs = outputs
            current_layer = current_layer.next_layer

        final_outputs = list()
        self.output_layer = OutputLayer(current_layer)
        for i, last_node in enumerate(self.output_layer.nodes):
            output = last_node.calculate_output_node(
                current_inputs, final=True)
            final_outputs.append(output)
            last_node.output = output
            if y != "":
                last_node.y = y[i]
                if len(self.output_layer.nodes) != len(y):
                    raise OutputNetworkException(
                        "number of output different of nodes's number in output layer")
        return final_outputs

    def update_weights(self, alpha, x):
        current_layer = self.output_layer
        while(hasattr(current_layer.previous_layer, "previous_layer")):
            # print(current_layer.__dict__)
            for node in current_layer.nodes:
                for i, weight in enumerate(node.weights):
                    node.weights[i] = weight + alpha * \
                        current_layer.previous_layer.nodes[i].output * \
                        node.delta
            current_layer = current_layer.previous_layer

        for node in current_layer.nodes:
            for i, weight in enumerate(node.weights):
                node.weights[i] = weight + alpha * x[i] * node.delta

    def train(self, data_x, data_y, alpha=0.01, epoch=100):

        x_train, y_train = data_x, data_y

        for step in range(epoch):

            for i, x in enumerate(x_train):
                self.predict(x, y=y_train[i])
                self.backpropagate()
                self.update_weights(alpha, x)

    def backpropagate(self):
        current_layer = self.output_layer

        while(hasattr(current_layer, "previous_layer")):
            current_layer.set_current_deltas()
            current_layer = current_layer.previous_layer

    def NMSE(self, X, Y):
        N = len(X)

        # Calcul de la moyenne
        mean = 0
        for i in range(N):
            mean += X[i][0]
        # Calcul de la variance

        variance = 0
        for i in range(N):
            variance = (X[i][0]-mean)**2

        variance = variance/(N-1)
        mean = mean/N
        sums = 0
        for i in range(N):
            sums += (Y[i][0]-self.predict(X[i])[0])**2
        return sums/(N*variance)


class InputNetworkException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class OutputNetworkException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


"""## Suite definition"""

suites = []
I = list(range(n))
value = 0.1

for i in I:
    if value > 0:
        suites.append(value)
    else:
        suites.append(0)
    value = A*value*(1-value)
plt.plot(I, suites)
plt.show()

"""## Classe Matrice"""

# TODO: Make custom exceptions
# TODO: You shouldn't be able to delete a single element from a row, only full rows and columns


class Matrix(object):

    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.matrix = []

        for i in range(rows):
            self.matrix.append([])  # Initialize empty rows

        for row in self.matrix:
            for i in range(columns):
                row.append(0)  # Fill the rows with 0s

    def __getitem__(self, key):
        return self.matrix[key]

    def __setitem__(self, key, value):
        if isinstance(value, list):
            self.matrix[key] = value
        else:
            raise TypeError(
                "A matrix object can only contain lists of numbers")
        return

    def __add__(self, otherMatrix):
        '''Add 2 matrices of the same type.'''
        return self.__add_or_sub(otherMatrix, "add")

    def __sub__(self, otherMatrix):
        '''Subtracts otherMatrix from self.'''
        return self.__add_or_sub(otherMatrix, "sub")

    def transpose(self):
        newMatrix = Matrix(self.columns, self.rows)

        for row in range(self.rows):
            for column in range(self.columns):
                # a(i,j) = a(j,i)
                newMatrix[column][row] = self.matrix[row][column]

        return newMatrix

    def __mul__(self, secondTerm):
        if isinstance(secondTerm, (int, float, complex)):
            return self.__scalar_product(secondTerm)
        elif isinstance(secondTerm, Matrix):
            if self.columns == secondTerm.rows:
                newMatrix = Matrix(self.rows, secondTerm.columns)
                transposeMatrix = secondTerm.transpose()
                '''
                Matrix multiplication is done iterating through each column of the
                second term. We calculate the transpose of the second matrix because
                it gives us a list for each column, which is far easier to iterate
                through.
                '''

                for row_self in range(self.rows):
                    for row_transpose in range(transposeMatrix.rows):
                        '''
                        The rows of the transpose correspond to the columns
                        of the original matrix.
                        '''
                        new_element = 0
                        for column_self in range(self.columns):
                            new_element += (self[row_self][column_self] *
                                            transposeMatrix[row_transpose][column_self])

                        newMatrix[row_self][row_transpose] = new_element

                return newMatrix

            else:
                raise Exception(
                    "Can't multiply (%d, %d) matrix with (%d, %d) matrix" %
                    (self.rows, self.columns, secondTerm.rows, secondTerm.columns)
                )
        else:
            raise TypeError(
                "Can't multiply a matrix by non-int of type " + type(secondTerm).__name__)

    def __rmul__(self, secondTerm):
        return self.__mul__(secondTerm)

    def __scalar_product(self, number):
        newMatrix = Matrix(self.rows, self.columns)

        for row in range(self.rows):
            for column in range(self.columns):
                newMatrix[row][column] = self[row][column] * number

        return newMatrix


"""## Recherche nombre d'entrée"""

# RECHERCHE DU NOMBRE D'UNITE D'ENTREE PAR L'ALGORITHME DE TAKENS
m1 = Matrix(1, len(suites))
m2 = Matrix(len(suites), 1)


m1[0] = suites

for i in range(len(suites)):
    m2[i] = [suites[i]]

# conversion vercteur en matrice 1d


def max_ligne(matrice):
    maxi = matrice[0][1]
    r_max = 0
    for i in range(matrice.rows):
        for j in range(matrice.columns):
            if i != j and matrice[i][j] >= maxi:
                maxi = matrice[i][j]
                r_max = i
    return r_max


def max_colonne(matrice):
    maxi = m[0][1]
    c_max = 0
    for i in range(matrice.rows):
        for j in range(matrice.columns):
            if i != j and m[i][j] >= maxi:
                maxi = m[i][j]
                r_max = j
    return c_max

# PRODUIT VECTORIEL POUR LA RECHERCHE DE THETA: Covariance


def prod_vec(a, b):
    prod = Matrix(len(I), len(I))
    for i in range(len(I)):
        vec = []
        for j in range(len(I)):
            vec.append(a[0][i]*b[j][0])
            # prod[]
        prod[i] = vec

    return prod


def matriceUnit(dim):
    m_nbLignes = dim
    m_nbColonnes = dim
    unit = Matrix(dim, dim)
    for i in range(m_nbLignes):
        for j in range(m_nbColonnes):
            unit[i][j] = 0
    return unit

# cosinus et sinus


def sinus(a, b):
    return math.sqrt(0.5*(1+(b/math.sqrt(a*a+b*b))))


def cosinus(a, b):
    return (a/(2*cosinus(a, b) * math.sqrt(a*a + b*b)))

# METHODE DE JACOBI: DIAGONALISATION


def diagonalization(matrice):
    lmax = max_ligne(matrice)
    cmax = max_ligne(matrice)

    P = matriceUnit(matrice.columns)
    a = 2*matrice[lmax][cmax]
    b = matrice[lmax][lmax] - matrice[cmax][cmax]
    if matrice[lmax][lmax] != matrice[cmax][cmax]:
        d = sinus(a, b)
        c = cosinus(a, b)
    else:
        c = math.sqrt(2)/2
        d = math.sqrt(2)/2

    for i in range(P.rows):
        for j in range(P.columns):
            if i == j and j != lmax and j != cmax:
                P[i][j] = 1
            elif i == cmax and j == lmax:
                P[i][j] = -d
            elif i == lmax and j == lmax or i == cmax and j == cmax:
                P[i][j] = c
            elif i == lmax and j == cmax:
                P[i][j] = d
            else:
                P[i][j] = 0
    transp_P = P.transpose()
    return transp_P * matrice * P

# ERREUR D'APPROXIMATION MOYENNE


def erreur_approx(vec):
    l = list()
    for i in range(len(vec)):
        l.append(math.sqrt(vec[i] + 1))
    return l

# VALEUR PROPRE


def val_prop(M):
    diagonalized = diagonalization(M)
    result = list()
    for i in range(diagonalized.rows):
        for j in range(diagonalized.columns):
            if i == j:
                result.append(diagonalized[i][i])

    return sorted(result, reverse=True)

# retour au premier plateau de la courbe


def nbUniteEntree(l):
    count = 1
    c1 = l[0]
    c2 = l[1]
    for i in range(len(l)):
        erreur = abs(c2-c1)
        if erreur <= 0.001 and erreur > 0.:
            count = i
            break
        else:
            c1 = c2
            c2 = l[i]
    return count


theta = prod_vec(m1, m2)
vp = val_prop(theta)
erreur = erreur_approx(vp)
print(f'Erreur approximation moyenne: {sum(erreur)/len(erreur)}')
nb_input = nbUniteEntree(erreur)
print(f'Nombre unité d\'entrée : {nb_input}')

# ETO TSY AIKO INTSONY
X = list()
Y = list()


counter = 0
while counter < len(suites) - nb_input-1:
    x = list()
    temp_counter = counter
    for n in range(nb_input):
        x.append(suites[temp_counter])
        temp_counter = temp_counter + 1
    X.append(x)
    counter = counter + 1
    y = suites[counter]
    Y.append([y])


nmse = list()
nb_neurons = list(range(1, 20))
best_nb_neurons = 0


for nb_neuronne in nb_neurons:

    nn = Network([2, nb_neuronne, nb_neuronne, nb_neuronne, 1])

    nn.train(X, Y, alpha=0.01)

    er = nn.NMSE(X, Y)
    nmse.append(
        {
            'abs': nb_neuronne,
            'ord': er
        }
    )

plt.plot([abs['abs'] for abs in nmse], [ord['ord'] for ord in nmse])
plt.title(f'A={A}')
plt.xlabel('nombre neurones des couches cachées')
plt.ylabel('NMSE')
plt.show()

best_hidden_neurons = min(nmse, key=lambda x: x['ord'])['abs']
print(f"nombre unitées cachées : {best_hidden_neurons}")

final_rna = Network(
    [2, best_hidden_neurons, best_hidden_neurons, best_hidden_neurons, 1])
final_rna.train(X, Y, alpha=0.01)

predictions = list()
for x in X:
    predictions.append(final_rna.predict(x)[0])


plt.plot(I[nb_input+1:], predictions, label="prédictions")
plt.plot(I, suites, label="valeurs attendues")
plt.title(f'prédiction à une pas en avant pour A={A}')
plt.xlabel('n')
plt.ylabel('x(n)')
plt.legend()
plt.show()


def plusieurs_pas(pas, x, nb_input, suites):

    # Prédictions avec plusieurs pas
    pp_predictions = list()
    for _ in range(pas):
        y_predict = final_rna.predict(x)
        x = y_predict+x

        # On ajoute la valeur prédite dans l'élément d'entrée
        x = x[:len(x)-1]
        pp_predictions.append(y_predict[0])

    plt.plot(list(range(nb_input+1, nb_input+pas+1)),
             pp_predictions, label="prédictions")
    plt.plot(I, suites, label="valeurs attendues")
    plt.title(f'Prédictions à {pas} pas en avant pour A={A}')
    plt.xlabel('n')
    plt.ylabel('x(n)')
    plt.legend()
    plt.show()


plusieurs_pas(3, X[0], nb_input, suites)
plusieurs_pas(10, X[0], nb_input, suites)
plusieurs_pas(20, X[0], nb_input, suites)
print(f'NMSE : {final_rna.NMSE(X, Y)}')
