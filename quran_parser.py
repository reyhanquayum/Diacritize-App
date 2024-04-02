from pyarabic import araby
# documentation https://github.com/linuxscout/pyarabic/blob/master/doc/features.md

singleLine = ""
with open("quran-simple.txt", encoding='utf-8') as Quran:
    singleLine = Quran.readline()
    # this should be 

letters, marks = araby.separate(singleLine)
print(letters) # i want to see what this looks like unencoded
# its cursed.
print(marks) # same here

print(letters.encode('utf8'))
print(marks.encode('utf8'))\


"""
Cool stuff from the docs:
>>> from pyarabic import araby
>>> araby.separate(text)
(u'\u0627\u0644\u0639\u0631\u0628\u064a\u0629', u'\u064e\u0652\u064e\u064e\u064e\u064e\u064f')
>>> letters, marks =araby.separate(text)
>>> print(letters.encode('utf8'))
العربية
>>> print(marks.encode('utf8'))
>>> for m in marks:
...     print(araby.name(m))
فتحة
سكون
فتحة
فتحة
فتحة
فتحة
ضمة
"""