import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords

class preprocessing_msg:

    def transform_text(msg):
        ps = PorterStemmer()
        msg = nltk.word_tokenize(msg)  # tokenize words into list
        y = []

        # remove all special characters
        for i in msg:
            if i.isalnum():
                y.append(i)

        msg = y[:]
        y.clear()

        # remove stopwords and punctuation
        for i in msg:
            if i.lower() not in stopwords.words('english') and i not in string.punctuation:
                y.append(i.lower())

        msg = y[:]
        y.clear()

        # stemming
        for i in msg:
            y.append(ps.stem(i))

        return " ".join(y)
