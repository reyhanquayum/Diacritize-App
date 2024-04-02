from pyarabic import araby
# documentation https://github.com/linuxscout/pyarabic/blob/master/doc/features.md

trainingfile = open("quran-training.txt", "w")
with open("quran-simple-no-ayat-numbers.txt", encoding='utf-8') as Quran:
    for line in Quran:
        letters, marks = araby.separate(line)
        for idx, letter in enumerate(letters):
            trainingfile.write(f'{letter.encode("utf8")}\t{marks[idx].encode("utf8")}\n')
        trainingfile.write("\n")