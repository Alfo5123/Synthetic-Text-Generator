''''
Markov Text Generator vs LSTM Model
'''

# Import Markov Chain Text Generator
import markovgen

# Load data
textfile = open('/home/alfredo/PycharmProjects/Synthetic-Text-Generator/docs/borges_collected-fictions.txt')

## Apply Markov Chain Text Generator
markov = markovgen.MarkovGenerator( textfile , 3 , 10 )
print(markov.generate_markov_text())
f = open('workfile.txt', 'w+')
f.write(markov.cache.__str__())
f.close()




