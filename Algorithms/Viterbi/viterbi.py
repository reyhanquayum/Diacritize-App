import copy
import sys

class ViterbiAlgorithm():
    def __init__(self, ngram) -> None:
        self.wordLikelihood = {}
        self.transitionMatrix = {"Begin_Sent": {}, "End_Sent": {}}
        self.ngram = ngram # the n-gram number of the item
        # the current ngram as implemented is only 2
        # i,e it only looks at the prev item not the prev n 
        
    def _normalizeProbabilities(matrix):
        for aRow in matrix:
            count = sum(matrix[aRow].values())
            for elem in matrix[aRow]:
                matrix[aRow][elem] /= count
        return matrix

    def fillMatrices(self, filePath):
        
        with open(filePath, "r", encoding="utf8") as trainingTxt:
            prevDiacritic = "Begin_Sent" # this is the previous diacritic which is begin sent in this case
            # this could possibly be changed to begin word?
            # i dont think thats a good idea tho since word inflection depends on other words as well as letters
            row = trainingTxt.readline() # starting seed value
            while row:
                if row == "\n": # if i have reached the end of the sentence i need to do the new stuff
                    # checking to see if the previous POS exists in the transition matrix
                    self.transitionMatrix.setdefault(prevDiacritic, {}).setdefault("End_Sent", 0)
                    # the transition matrix gets the prev diacritic and the transition state from that value
                    self.transitionMatrix[prevDiacritic]["End_Sent"] += 1
                    # the transitions are updated but this doesnt really help tbh
                    prevDiacritic = "Begin_Sent"
                    # the the prev diacritic is begin sent
                    row = trainingTxt.readline() # got my new row
                    continue
                    # making sure row isnt a new line again
                    
                letter, Diacritic = row.strip().split("\t") # splitting to get the letter and the diacritic
                # updating word likelihoood matrix

                self.wordLikelihood.setdefault(Diacritic, {}).setdefault(letter, 0)
                # how often that letter in that diacritic shows up
                self.wordLikelihood[Diacritic][letter] += 1
                # incrementing the letter
                self.transitionMatrix.setdefault(prevPOS, {}).setdefault(Diacritic, 0)
                self.transitionMatrix[prevPOS][Diacritic] += 1

                prevPOS = Diacritic
                row = trainingTxt.readline()

        self.wordLikeLihood = self._normalizeProbabilities(self.wordLikelihood) # normalzing everything
        self.transitionMatrix = self._normalizeProbabilities(self.transitionMatrix) # normalizing everything to probabilities


    def _create_letter_set(word_likelihood_matrix):
        word_set = set()
        for pos in word_likelihood_matrix:
            for word in word_likelihood_matrix[pos]:
                word_set.add(word)
        return word_set

    
    def _createStateIndices(transitionMatrix):
        stateIndices = {}
        states = list(transitionMatrix.keys())

        for i, state in enumerate(states):
            stateIndices[i] = state
        return stateIndices
        # initializing state indices
    

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


    def _find_best_path(backpointer, stateIndices):
        # Start with the state that has the highest probability at the end of the sequence
        bestItemEnd = backpointer[1][-1] # this is the end column bestItem
        bestPath = [stateIndices[bestItemEnd]]

        currRowPtr = backpointer[bestItemEnd]
        for t in range(len(backpointer[0]) - 2, 1, -1):
            bestPath.append(stateIndices[currRowPtr[t]])
            currRowPtr = backpointer[currRowPtr[t]]

        bestPath.reverse()
        return bestPath



    def viterbiAlgorithm(self, testFilePath, resultsPath):

        sentencesArray = self.readTestFile(testFilePath)
        stateIndices = self._createStateIndices(self.transitionMatrix)
        states = list(self.transitionMatrix.keys())
        
        writeFile = open(resultsPath, "w", encoding="utf-8")

        unknownWordProb = 1/5000000
        
        for sentence in sentencesArray: # going through a single sentence

            # initializing viterbi
            firstWord = sentence[0]
            viterbiMatrix = [[0 for _ in range(len(sentence) + 2)] for _ in range(len(states))]
            backpointer = [[0 for _ in range(len(sentence) + 2)] for _ in range(len(states))]
            viterbiMatrix[0][0] = 1
            for index in list(stateIndices.keys()):
                POS = stateIndices[index]
                viterbiMatrix[index][1] = self.transitionMatrix["Begin_Sent"].get(POS, 0) * self.wordLikeLihood.get(POS, {}).get(firstWord, unknownWordProb)
            # done init viterbi
            
            for j in range(1, len(sentence)):
                # going through the sentence aka the words
                word = sentence[j] # making the word (lower.case) which might be the issue

                for i, state in enumerate(states):
                    # the rows
                    maxProb = 0
                    # finding value that maximizes the next state
                    # finding prev state that maximizes
                    for k, prevState in enumerate(states): # going through all previous states
                        transitionProb = self.transitionMatrix.get(prevState, {}).get(state, 0)
                        prevProb = viterbiMatrix[k][j]
                        currProb = prevProb * transitionProb

                        if currProb > maxProb:
                            maxProb = currProb
                            backpointer[i][j + 1] = k


                    viterbiMatrix[i][j + 1] = maxProb * self.wordLikeLihood.get(state, {}).get(word, unknownWordProb)


            # capturing end state here
            for i, state in enumerate(states):
                maxProb = 0
                for k, prevState in enumerate(states):  # going through all previous states
                    transitionProb = self.transitionMatrix.get(prevState, {}).get(state, 0)
                    prevProb = viterbiMatrix[k][len(sentence)] 
                    currProb = prevProb * transitionProb
                    if currProb > maxProb:
                        maxProb = currProb
                        backpointer[i][len(sentence) + 1] = k
                    viterbiMatrix[i][len(sentence) + 1] = maxProb
            # done capturing end state
            
            bestPath = self._find_best_path(backpointer, stateIndices)
            for i in range(len(sentence)):

                writeFile.write(sentence[i] + "\t" + bestPath[i] + "\n")
            writeFile.write("\n")

        writeFile.close()