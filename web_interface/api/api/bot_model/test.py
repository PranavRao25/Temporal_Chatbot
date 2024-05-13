from interface import *

t = Interface()

s = 'How many days in a week?'
k = t.userInput(s)
t.displayTranscript()
print(f"s = {k}")
t.printTranscript()
# t.clearTranscript()
# t.displayTranscript()
#
# s = 'How many days in a month?'
# t.userInput(s)
# t.displayTranscript()
# t.printTranscript()

# t.closeInterface()
