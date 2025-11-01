import  nltk
nltk.download('punkt_tab')
nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords

class preprocessing_msg:

    def transform_text(msg):
        string.punctuation
        ps = PorterStemmer()
        msg = nltk.word_tokenize(msg)  # all words convert into a list, eg ['free', 'entry', in]
        y = []
        for i in msg:
            if i.isalnum():  # remove all special character
                y.append(i)
        msg = y[:]
        y.clear()
        for i in msg:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
        msg = y[:]
        y.clear()
        for i in msg:
            y.append(ps.stem(i))
        return " ".join(y)
