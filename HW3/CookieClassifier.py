import numpy

# ======================================= Fortune Cookie Classifier ==================================================

# Files for Fortune Cookie Classifier
STOP_LIST = 'stoplist.txt'
TEST_DATA = 'testdata.txt'
TEST_LABELS = 'testlabels.txt'
TRAINING_DATA = 'traindata.txt'
TRAINING_LABELS = 'trainlabels.txt'

# Output file
OUT_FILE = './output.txt'

# number of report iteration
ITERATIONS = 20

# learning rate η=1.
LEARNING_RATE = 1

# Obtain training all vocabularies.
training_words = set()
f = open(TRAINING_DATA, 'r')
lines = f.read().split('\n')
for line in lines:
    words = line.split(' ')
    for word in words:
        training_words.add(word)
f.close()

# Remove list of stop words from the training_words.
f = open(STOP_LIST, 'r')
stop_words = f.read().split('\n')
for word in stop_words:
    training_words.discard(word)
f.close()
del stop_words

# To make debugging easy sort training_words.
vocabulary = dict()
M = len(training_words)  # Let M be the size of vocabulary
for index, word in enumerate(sorted(training_words)):
    vocabulary[word] = index
del training_words

# For these M slots, if the ith slot is 1, it means that the ith word in the vocabulary is present in
# the fortune cookie message; otherwise, if it is 0, then the ith word is not present in the message
fortunes = numpy.zeros((len(lines), M + 1))
for index, line in enumerate(lines):
    words = line.split(' ')
    for word in words:
        if vocabulary.get(word) is not None:
            fortunes[index][vocabulary[word]] = 1
    fortunes[index][M] = 1
del lines

# Make a list of training liable.
f = open(TRAINING_LABELS, 'r')
train_labels = f.read().split('\n')
for index, label in enumerate(train_labels):
    if int(label) == 0:
        train_labels[index] = -1
    else:
        train_labels[index] = 1
f.close()

# Make a vector of testing data.
f = open(TEST_DATA, 'r')
lines = f.read().split('\n')
lines.pop()
test_data = numpy.zeros((len(lines), M + 1))
for index, line in enumerate(lines):
    words = line.split(' ')
    for word in words:
        if vocabulary.get(word) is not None:
            test_data[index][vocabulary[word]] = 1
    test_data[index][M] = 1
f.close()
del lines

# Make list of testing labels.
f = open(TEST_LABELS, 'r')
test_labels = f.read().split('\n')
test_labels.pop()
for index, label in enumerate(test_labels):
    if int(label) == 0:
        test_labels[index] = -1
    else:
        test_labels[index] = 1
f.close()


# Accuracy for Fortune Cookie Classifier.
def accuracy(weight, examples, labels):
    correct = 0
    S = numpy.shape(examples)
    for index in range(0, S[0]):
        predicted = numpy.dot(examples[index], numpy.transpose(weight))
        if ((predicted[0] > 0 and labels[index] > 0) or \
                (predicted[0] <= 0 and labels[index] < 0)):
            correct += 1
    return correct / S[0]


# Perceptron
w = numpy.zeros((1, M + 1))
mistakes_list = list()
train_list = list()
test_list = list()
S = numpy.shape(fortunes)
for i in range(1, ITERATIONS + 1):
    mistakes = 0
    for index in range(0, S[0]):
        predicted = numpy.dot(fortunes[index], numpy.transpose(w))
        if predicted[0] * train_labels[index] <= 0:
            mistakes += 1
            w = w + LEARNING_RATE * train_labels[index] * fortunes[index]
    mistakes_list.append(mistakes)
    train_list.append(accuracy(w, fortunes, train_labels))
    test_list.append(accuracy(w, test_data, test_labels))

# Average Perceptron
w = numpy.zeros((1, M + 1))
x = numpy.zeros((1, M + 1))
y = 1
S = numpy.shape(fortunes)
for i in range(1, ITERATIONS + 1):
    for index in range(0, S[0]):
        predicted = numpy.dot(fortunes[index], numpy.transpose(w))
        if predicted[0] * train_labels[index] <= 0:
            w = w + LEARNING_RATE * train_labels[index] * fortunes[index]
            u = x + y * LEARNING_RATE * train_labels[index] * fortunes[index]
        y += 1
w = w - x * (1 / y)
avg_train_acc = accuracy(w, fortunes, train_labels)
avg_test_acc = accuracy(w, test_data, test_labels)

# Dump on output file
f = open(OUT_FILE, 'w')
f.write('fortune-cookie classifier block' + '\n\n')
for i in range(1, ITERATIONS + 1):
    f.write('iteration-' + str(i) + ' ' + str(mistakes_list[i - 1]) + '\n')
f.write('\n')
for i in range(1, ITERATIONS + 1):
    f.write('iteration-' + str(i) + ' ' + str(train_list[i - 1]) + ' ' + str(test_list[i - 1]) + '\n')
f.write('\n')
f.write(str(train_list[ITERATIONS - 1]) + ' ' + str(test_list[ITERATIONS - 1]) + '\n')
f.write(str(avg_train_acc) + ' ' + str(avg_test_acc) + '\n\n')
f.close()

# ============================= optical character recognition (OCR) classifier =======================================

# Files for OCR classifier

OCR_TRAINING = 'ocr_train.txt'
OCR_TESTING = 'ocr_test.txt'
ALPHABET = 26

# Obtain list of orc_test.
ocr_test = list()
ocr_test_label = list()
f = open(OCR_TESTING, 'r')
lines = f.read().split('\n')
for line in lines:
    elements = line.split('\t')
    if (len(elements) > 3) and (elements[3] == '_'):
        ocr_test.append(elements[1].lstrip('im'))
        ocr_test_label.append(elements[2])
f.close()
del lines

# Obtain a list of ocr_train.
ocr_train = list()
ocr_train_label = list()
f = open(OCR_TRAINING, 'r')
lines = f.read().split('\n')
for line in lines:
    elements = line.split('\t')
    if (len(elements) > 3) and (elements[3] == '_'):
        ocr_train.append(elements[1].lstrip('im'))
        ocr_train_label.append(elements[2])
f.close()

# Feature vectors for training data.
features = len(ocr_train[0])
train_data = numpy.zeros((len(ocr_train), features + 1))
for index, example in enumerate(ocr_train):
    for i, digit in enumerate(example):
        train_data[index][i] = int(digit)
    train_data[index][features] = 1
del ocr_train

# Feature vectors for testing data.
test_data = numpy.zeros((len(ocr_test), features + 1))
for index, example in enumerate(ocr_test):
    for i, digit in enumerate(example):
        test_data[index][i] = int(digit)
    test_data[index][features] = 1
del ocr_test

# Build dictionaries to map letter to number
number_to_letter = dict()
letter_to_number = dict()
letters = sorted(list(set(ocr_train_label)))
for num, letter in enumerate(letters):
    letter_to_number[letter] = num
    number_to_letter[num] = letter

# Accuracy OCR Classifier.
def accuracy_ocr(weight, examples, labels, x):
    correct = 0
    S = numpy.shape(examples)
    for i in range(0, S[0]):
        predicted = numpy.zeros((1, ALPHABET))
        for j in range(0, ALPHABET):
            predicted[0][j] = numpy.dot(examples[i], numpy.transpose(weight[j]))
        if x[numpy.argmax(predicted)] == labels[i]:
            correct += 1
    return correct / S[0]


# Perceptron
w = numpy.zeros((ALPHABET, features + 1))
mistakes_list = list()
train_list = list()
test_list = list()
S = numpy.shape(train_data)
for i in range(1, ITERATIONS + 1):
    mistakes = 0
    for j in range(0, S[0]):
        predicted = numpy.zeros((1, ALPHABET))
        for k in range(0, ALPHABET):
            predicted[0][k] = numpy.dot(train_data[j], numpy.transpose(w[k]))
        p_index = numpy.argmax(predicted)
        a_index = letter_to_number[ocr_train_label[j]]
        if p_index != a_index:
            mistakes += 1
            w[p_index] = w[p_index] - LEARNING_RATE * train_data[j]
            w[a_index] = w[a_index] + LEARNING_RATE * train_data[j]
    mistakes_list.append(mistakes)
    train_list.append(accuracy_ocr(w, train_data, ocr_train_label, number_to_letter))
    test_list.append(accuracy_ocr(w, test_data, ocr_test_label, number_to_letter))

# AvgPerceptron
w = numpy.zeros((ALPHABET, features + 1))
u = numpy.zeros((ALPHABET, features + 1))
c = 1
for i in range(1, ITERATIONS + 1):
    for j in range(0, S[0]):
        predicted = numpy.zeros((1, ALPHABET))
        for k in range(0, ALPHABET):
            predicted[0][k] = numpy.dot(train_data[j], numpy.transpose(w[k]))
        p_index = numpy.argmax(predicted)
        a_index = letter_to_number[ocr_train_label[j]]
        if p_index != a_index:
            w[p_index] = w[p_index] - LEARNING_RATE * train_data[j]
            w[a_index] = w[a_index] + LEARNING_RATE * train_data[j]
            u[p_index] = u[p_index] - c * LEARNING_RATE * train_data[j]
            u[a_index] = u[a_index] + c * LEARNING_RATE * train_data[j]
        c += 1
w = w - u * (1 / c)
avg_train_acc = accuracy_ocr(w, train_data, ocr_train_label, number_to_letter)
avg_test_acc = accuracy_ocr(w, test_data, ocr_test_label, number_to_letter)

# Dump output file
f = open(OUT_FILE, 'a')
f.write('Optical character recognition block.' + '\n\n')
for i in range(1, ITERATIONS + 1):
    f.write('iteration-' + str(i) + ' ' + str(mistakes_list[i - 1]) + '\n')
f.write('\n')
for i in range(1, ITERATIONS + 1):
    f.write('iteration-' + str(i) + ' ' + str(train_list[i - 1]) + ' ' + str(test_list[i - 1]) + '\n')
f.write('\n')
f.write(str(train_list[ITERATIONS - 1]) + ' ' + str(test_list[ITERATIONS - 1]) + '\n')
f.write(str(avg_train_acc) + ' ' + str(avg_test_acc) + '\n')
f.close()
