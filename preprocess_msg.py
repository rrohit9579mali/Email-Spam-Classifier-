import nltk
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords

# Safe punkt downloader (handles both new and old versions)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class preprocessing_msg:

    def transform_text(msg):
        ps = PorterStemmer()
        msg = nltk.word_tokenize(msg)  # convert text into list of words
        y = []

        for i in msg:
            if i.isalnum():  # remove special chars
                y.append(i)

        msg = y[:]
        y.clear()

        for i in msg:
            if i.lower() not in stopwords.words('english') and i not in string.punctuation:
                y.append(i.lower())

        msg = y[:]
        y.clear()

        for i in msg:
            y.append(ps.stem(i))

        return " ".join(y)
