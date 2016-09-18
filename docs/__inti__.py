''''
Markov Text Generator vs LSTM Model
'''

# Import Markov Chain Text Generator
import markovgen

# Load data
textfile = open('/home/alfredo/Escritorio/borges_collected-fictions.txt')

## Apply Markov Chain Text Generator
markov = markovgen.MarkovGenerator( textfile , 3 , 25 )
print(markov.generate_markov_text())




