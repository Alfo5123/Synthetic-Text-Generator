''''
Markov Text Generator vs LSTM Model
'''

# Import Markov Chain Text Generator
import markovgen

# Import LSTM Model Generator
import lstmgen

# Load data
textfile = open('/home/alfredo/Escritorio/borges_collected-fictions.txt')

## Apply Markov Chain Text Generator
markov = markovgen.MarkovGenerator( textfile , 3 , 25 )
markov.generate_markov_text()

## Apply LSTM Character Language Model Text Generator
lstm = lstmgen.LSTMGenerator( textfile , 40 , 100 )
lstm.generate_lstm_text()

