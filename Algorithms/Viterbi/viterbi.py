import copy


class ViterbiAlgorithm():
    def __init__(self) -> None:
        self.wordLikelihood = {}
        self.transitionMatrix = {"Begin_Sent": {}, "End_Sent": {}}
        
def normalizeProbabilities(matrix):
    for aRow in matrix:
        count = sum(matrix[aRow].values())
        for elem in matrix[aRow]:
            matrix[aRow][elem] /= count
    return matrix

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


def determineClass(word):
    classes = ["endsinED", "endsInING", "endInS", "endsInTION",
               "endsInMENT", "endsInNESS", "endsInABLE",
               "endsInIBLE", "endsInAL", "endsInFUL", "endsInOUS", "endsInIVE",
               "endsInANT", "endsInENT", "endsInISM", "endsInIST", "endsInITY",
               "endsInIZE", "endsInISE", "endsInSHIP", "endsInLESS", "endsInER",
               "endsInOR", "endsInEST", "endsInLY", "endsInY"
               "startsCapital", "isNumber", "isAllCaps", "isHyphenated",
               "startsInUN", "startsInRE", "startsInOVER", "startsInUNDER", "startsInOUT",
               "startsInIN", "startsInEX", "startsInNON", "startsInPOST", "startsInPRE"          
               ]
    
    wordsClasses = []
    try:
        if word[-2:] == "ed":
            wordsClasses.append(classes[0])
        elif word[-3:] == "ing":
            wordsClasses.append(classes[1])
        elif word[-1] == "s":
            wordsClasses.append(classes[2])
        elif word[-4:] == "tion":
            wordsClasses.append(classes[3])
        elif word[-1] == "ment":
            wordsClasses.append(classes[4])
        elif word[-3:] == "ness":
            wordsClasses.append(classes[5])
        elif word[-1] == "able":
            wordsClasses.append(classes[6])
        elif word[-3:] == "ible":
            wordsClasses.append(classes[7])
        elif word[-1] == "al":
            wordsClasses.append(classes[8])
        elif word[-3:] == "ful":
            wordsClasses.append(classes[9])
        elif word[-3:] == "ous":
            wordsClasses.append(classes[10])
        elif word[-3:] == "ive":
            wordsClasses.append(classes[11])
        elif word[-3:] == "ant":
            wordsClasses.append(classes[12])
        elif word[-3:] == "ent":
            wordsClasses.append(classes[13])
        elif word[-3:] == "ism":
            wordsClasses.append(classes[14])
        elif word[-3:] == "ist":
            wordsClasses.append(classes[15])
        elif word[-3:] == "ity":
            wordsClasses.append(classes[16])
        elif word[-3:] == "ize":
            wordsClasses.append(classes[17])
        elif word[-3:] == "ise":
            wordsClasses.append(classes[18])
        elif word[-4:] == "ship":
            wordsClasses.append(classes[19])
        elif word[-4:] == "less":
            wordsClasses.append(classes[20])
        elif word[-2:] == "er":
            wordsClasses.append(classes[21])
        elif word[-2:] == "or":
            wordsClasses.append(classes[22])
        elif word[-3:] == "est":
            wordsClasses.append(classes[23])
        if word[-2:] == "ly":
            wordsClasses.append(classes[24])
        elif word[-1] == "y":
            wordsClasses.append(classes[25])

        if word[1].isupper():
            wordsClasses.append(classes[26])

        for char in word:
            if char.isdigit():
                wordsClasses.append(classes[27])
                break
        isAll = True
        for char in word:
            if not char.isupper():
                isAll = False
                break
        if isAll:
            wordsClasses.append(classes[28])

        for char in word:
            if char == "-":
                wordsClasses.append(classes[29])

        if word[:2] == "un":
             wordsClasses.append(classes[30])
        elif word[:2] == "re":
            wordsClasses.append(classes[31])
        elif word[:4] == "over":
            wordsClasses.append(classes[32])
        elif word[:5] == "under":
            wordsClasses.append(classes[33])
        elif word[:3] == "out":
            wordsClasses.append(classes[34])
        elif word[:2] == "in":
            wordsClasses.append(classes[35])
        elif word[:2] == "ex":
            wordsClasses.append(classes[36])
        elif word[:3] == "non":
            wordsclasses.append(classes[37])
        elif word[:4] == "post":
            wordsClasses.append(classes[38])
        elif word[:3] == "pre":
            wordsClasses.append(classes[39])
        return wordsClasses if len(wordsClasses) > 0 else ["unknownClass"]

    except:

        return ["unknownClass"]

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

