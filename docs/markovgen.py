import random

class MarkovGenerator(object):
    def __init__(self, open_file,order, size):
        self.cache = {}
        self.open_file = open_file
        self.order = order
        self.size = size
        self.words = self.file_to_words()
        self.word_size = len(self.words)
        self.build_dictionary()

    def file_to_words(self):
        """
            Split the whole text in words
        """
        self.open_file.seek(0)
        data = self.open_file.read()
        words = data.split()
        return words

    def tokenize(self):
        """
        Generates n-gram groups of words
        """
        if len(self.words) < self.order :
            return

        for i in range( len(self.words) - self.order + 1 ):
            ngram = []
            for j in range(self.order):
                ngram.append(self.words[i+j])
            yield ngram

    def build_dictionary(self):
        """
          Build a dictionary based on the next word for a given n-gram
        """
        for ngram in self.tokenize():
            next = ngram.pop()
            key = tuple(ngram)
            if key in self.cache:
                self.cache[key].append(next)
            else:
                self.cache[key] = [next]

    def generate_markov_text(self):
        """
            Generate a text by starting at a random position and apply the Markov Chain
        """
        seed = random.randint(0, self.word_size - self.order)
        seed_words = self.words[seed:seed+self.order-1]
        curr_words = seed_words
        gen_words = []
        for i in xrange(self.size):
            gen_words.append(curr_words[0])
            curr_words.append( random.choice(self.cache[tuple(curr_words)] ) )
            curr_words = curr_words[1:]
        gen_words.append(curr_words[self.order-2])
        return "' ..." + ' '.join(gen_words) + "... ' "