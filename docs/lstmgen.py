import numpy
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

class LSTMGenerator(object):
    ''''
    Long Short Term Memory Character-based Language Model for Text Generation
    '''

    def __init__(self,input_file, seq_length , output_size):

       self.input_file = input_file
       self.seq_length = seq_length
       self.output_size = output_size
       self.characters = self.file_to_characters()
       self.input_size = len(self.characters)

       # Create a mapping to encode characters
       vocabulary = sorted(list(set(self.characters)))
       self.vocabulary_size = len( vocabulary )
       self.char_to_int = dict((c, i) for i, c in enumerate(vocabulary))
       self.int_to_char = dict((i, c) for i, c in enumerate(vocabulary))

       # Training-Data adquisition
       self.X , self.Y = self.get_training_data(self.seq_length)

       # Training model
      # if ~os.path.isfile('Trained_LSTM.hdf5') :
        #self.train_lstm(self.X , self.Y )


    def file_to_characters(self):
        """
            Split the whole text in characters
        """

        #Read data
        self.input_file.seek(0)
        data = self.input_file.read()

        # Preprocess data
        data = data.lower() # Reduce the number of characters by using lowercase
        characters = [ data[i] if data[i] != '\n' else ' ' for i in range(len(data))]

        chars_to_keep = set((" ", ",", ";", "."))
        final_characters = []
        for i in range(len(characters)):
            if characters[i].isalpha() or characters[i] in chars_to_keep:
                final_characters.append(characters[i])

        return final_characters

    def get_training_data ( self , seq_length ):
        """
            Generate input and output data by taking sequence
            and predicting the next character after that sequence
            of length seqLength.
        """
        dataX = []
        dataY = []
        for i in range(0, self.input_size - seq_length, 1):
            seq_in = self.characters[i:i + seq_length]
            seq_out = self.characters[i + seq_length]
            dataX.append([self.char_to_int[char] for char in seq_in])
            dataY.append(self.char_to_int[seq_out])
        n_patterns = len(dataX)

        # Reshape X to be [samples, time steps, features]
        X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
        # Normalize
        X = X / float(self.vocabulary_size)
        # One hot encode the output variable
        Y = np_utils.to_categorical(dataY)
        return X , Y


    def train_lstm (self , X , Y ):
        """
            Train the model and generate a text by starting at a random
            position and apply the  trained LSTM character based model
        """

        # Define the model
        model = Sequential()
        model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(Y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        # Define CheckPoint
        filepath = "Trained_LSTM.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        # Fit the model
        model.fit(X, Y, nb_epoch=1, batch_size=128, callbacks=callbacks_list)

    def generate_lstm_text(self):

        # Define the model
        model = Sequential()
        model.add(LSTM(128, input_shape=(self.X.shape[1], self.X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(self.Y.shape[1], activation='softmax'))
        filename = "Trained_LSTM.hdf5"
        model.load_weights(filename)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        # Define CheckPoint
        filepath = "Trained_LSTM.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        # Fit the model
        model.fit(self.X, self.Y, nb_epoch=1, batch_size=128, callbacks=callbacks_list)

        # Start at random point in text
        start = numpy.random.randint(0, self.input_size - 1)
        seed = self.characters[start:start + self.seq_length]
        pattern = [self.char_to_int[char] for char in seed ]
        output_text = seed

        # Generate characters
        for i in range(self.output_size):
            x = numpy.reshape(pattern, (1, len(pattern), 1))
            x = x / float(self.vocabulary_size)
            prediction = model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            result = self.int_to_char[index]
            output_text.append(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]

        print "LSTM Text Generator Example:\n \t'..." + ''.join(output_text) + "...'"

