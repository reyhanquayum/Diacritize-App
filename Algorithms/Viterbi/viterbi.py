import copy


class ViterbiAlgorithm():
    def __init__(self, ngram) -> None:
        self.wordLikelihood = {}
        self.transitionMatrix = {"Begin_Sent": {}, "End_Sent": {}}
        self.ngram = ngram # the n-gram number of the item

    def _normalizeProbabilities(matrix):
        for aRow in matrix:
            count = sum(matrix[aRow].values())
            for elem in matrix[aRow]:
                matrix[aRow][elem] /= count
        return matrix

    def fillMatrices(filePath):
        
def createMatrices(path):
    # training set is

    wordLikelihood = {} # initializing word likelihood matrices
    transitionMatrix = {"Begin_Sent": {}, "End_Sent": {}}

    with open(path, "r", encoding="utf-8") as trainingData:
        prevPOS = "Begin_Sent"
        row = trainingData.readline() # starting seed value
        while row:
            if row == "\n":
                # checking to see if the previous POS exists in the transition matrix
                transitionMatrix.setdefault(prevPOS, {}).setdefault("End_Sent", 0)
                transitionMatrix[prevPOS]["End_Sent"] += 1
                prevPOS = "Begin_Sent"
                row = trainingData.readline()
                continue
                # making sure row isnt a new line again
                
            word, POS = row.strip().split("\t") # splitting to get the word and pos
            # updating word likelihoood matrix

            wordLikelihood.setdefault(POS, {}).setdefault(word, 0)
            wordLikelihood[POS][word] += 1

            transitionMatrix.setdefault(prevPOS, {}).setdefault(POS, 0)
            transitionMatrix[prevPOS][POS] += 1

            prevPOS = POS
            row = trainingData.readline()
            # reading the next line
    # converting to probabilities here:
    # editing word likelihood here now
    wordLikelihoodNew = copy.deepcopy(wordLikelihood)
    for state in wordLikelihood:
        for word in wordLikelihood[state]:
            if wordLikelihood[state][word] == 1:
                wordClasses = determineClass(word)
                for wordClass in wordClasses:
                    wordLikelihoodNew.setdefault(state, {}).setdefault(wordClass, 0)
                    wordLikelihoodNew[state][wordClass] += 1

    # converting word likelihood to STATES OF UNKNOWN

    wordLikeLihood = normalizeProbabilities(wordLikelihoodNew)
    transitionMatrix = normalizeProbabilities(transitionMatrix)
    
    return wordLikeLihood, transitionMatrix


def create_word_set(word_likelihood_matrix):
    word_set = set()
    for pos in word_likelihood_matrix:
        for word in word_likelihood_matrix[pos]:
            word_set.add(word)
    return word_set


def find_best_path(backpointer, stateIndices):
    # Start with the state that has the highest probability at the end of the sequence
    bestItemEnd = backpointer[1][-1] # this is the end column bestItem
    bestPath = [stateIndices[bestItemEnd]]

    currRowPtr = backpointer[bestItemEnd]
    for t in range(len(backpointer[0]) - 2, 1, -1):
        bestPath.append(stateIndices[currRowPtr[t]])
        currRowPtr = backpointer[currRowPtr[t]]

    bestPath.reverse()
    return bestPath

def readTestFile(devPath):
    sentencesArray = []

    with open(devPath, "r", encoding="utf-8") as devFile:

        tokens = []
        token = 1
        while token:
            token = devFile.readline()
            if token == "\n":
                sentencesArray.append(tokens)
                tokens = []
            else:
                tokens.append(token.strip())
    return sentencesArray


def createStateIndices(transitionMatrix):
    stateIndices = {}
    states = list(transitionMatrix.keys())

    for i, state in enumerate(states):
        stateIndices[i] = state
    return stateIndices
    # initializing state indices


def createWordLikelihoodClasses(wordLikelihoodMatrix):
   
    secondDict = {}
    for state in wordLikelihoodMatrix:
        for word in wordLikelihoodMatrix[state]:
            wordClasses = determineClass(word)
            for wordClass in wordClasses:
                secondDict.setdefault(state, {}).setdefault(wordClass, 0)
                secondDict[state][wordClass] += 1
    secondDict = normalizeProbabilities(secondDict)
    return secondDict
                

def viterbiAlgorithm(wordLikeLihood, transitionMatrix, devPath):

    sentencesArray = readTestFile(devPath)
    knownWords = create_word_set(wordLikeLihood)
    stateIndices = createStateIndices(transitionMatrix)
    states = list(transitionMatrix.keys())
    
    writeFile = open("myresults.pos", "w", encoding="utf-8")

    unknownWordProb = 1/5000000
    
    for sentence in sentencesArray: # going through a single sentence

        # initializing viterbi
        firstWord = sentence[0]
        viterbiMatrix = [[0 for _ in range(len(sentence) + 2)] for _ in range(len(states))]
        backpointer = [[0 for _ in range(len(sentence) + 2)] for _ in range(len(states))]
        viterbiMatrix[0][0] = 1
        for index in list(stateIndices.keys()):
            POS = stateIndices[index]
            viterbiMatrix[index][1] = transitionMatrix["Begin_Sent"].get(POS, 0) * wordLikeLihood.get(POS, {}).get(firstWord, unknownWordProb)
        # done init viterbi
        
        for j in range(1, len(sentence)):
            # going through the sentence aka the words
            word = sentence[j] # making the word (lower.case) which might be the issue
            if word not in knownWords:
                wordStates = determineClass(word)
                word = wordStates[0]

            for i, state in enumerate(states):
                # the rows
                maxProb = 0
                # finding value that maximizes the next state
                # finding prev state that maximizes
                maxState = 0
                for k, prevState in enumerate(states): # going through all previous states
                    transitionProb = transitionMatrix.get(prevState, {}).get(state, 0)
                    prevProb = viterbiMatrix[k][j]
                    currProb = prevProb * transitionProb

                    if currProb > maxProb:
                        maxProb = currProb
                        backpointer[i][j + 1] = k


                viterbiMatrix[i][j + 1] = maxProb * wordLikeLihood.get(state, {}).get(word, unknownWordProb)


        # capturing end state here
        for i, state in enumerate(states):
            maxProb = 0
            for k, prevState in enumerate(states):  # going through all previous states
                transitionProb = transitionMatrix.get(prevState, {}).get(state, 0)
                prevProb = viterbiMatrix[k][len(sentence)] 
                currProb = prevProb * transitionProb
                if currProb > maxProb:
                    maxProb = currProb
                    backpointer[i][len(sentence) + 1] = k
                viterbiMatrix[i][len(sentence) + 1] = maxProb
        # done capturing end state
        
        bestPath = find_best_path(backpointer, stateIndices)
        for i in range(len(sentence)):

            writeFile.write(sentence[i] + "\t" + bestPath[i] + "\n")
        writeFile.write("\n")

    writeFile.close()

trainingFilePath = sys.argv[2]
testingFilePath = sys.argv[3]
wordLikelihood, transitionmatrix = createMatrices(trainingFilePath)
viterbiAlgorithm(wordLikelihood, transitionmatrix, testingFilePath)

