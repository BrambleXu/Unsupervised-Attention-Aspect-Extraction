from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import codecs


def parse_sentence(line):
    lmtzr = WordNetLemmatizer()
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    return text_stem


def preprocess_train(domain):
    f = codecs.open('../datasets/'+domain+'/train.txt', 'r', 'utf-8')
    out = codecs.open('../preprocessed_data/'+domain+'/train.txt', 'w', 'utf-8')
    for line in f:
        tokens = parse_sentence(line)
        # # if len(tokens) > 0:  ## if not comment this line, the sentence number will decrease
        #     out.write(' '.join(tokens)+'\n')
        if len(tokens) > 0:
            out.write(' '.join(tokens)+'\n')


## The original preprocess_test, which will first check the label in ['Food', 'Staff', 'Ambience']
## we don't need this, so here we change this function
# def preprocess_test(domain):
#     # For restaurant domain, only keep sentences with single
#     # aspect label that in {Food, Staff, Ambience}
#
#     f1 = codecs.open('../datasets/'+domain+'/test.txt', 'r', 'utf-8')
#     f2 = codecs.open('../datasets/'+domain+'/test_label.txt', 'r', 'utf-8')
#     out1 = codecs.open('../preprocessed_data/'+domain+'/test.txt', 'w', 'utf-8')
#     out2 = codecs.open('../preprocessed_data/'+domain+'/test_label.txt', 'w', 'utf-8')
#
#     for text, label in zip(f1, f2):
#         label = label.strip()
#         if domain == 'restaurant' and label not in ['Food', 'Staff', 'Ambience']:
#             continue
#         tokens = parse_sentence(text)
#         if len(tokens) > 0:
#             out1.write(' '.join(tokens) + '\n')
#             out2.write(label+'\n')

def preprocess_test(domain):
    # For restaurant domain, only keep sentences with single
    # aspect label that in {Food, Staff, Ambience}

    f1 = codecs.open('../datasets/'+domain+'/test.txt', 'r', 'utf-8')
    f2 = codecs.open('../datasets/'+domain+'/test_label.txt', 'r', 'utf-8')
    out1 = codecs.open('../preprocessed_data/'+domain+'/test.txt', 'w', 'utf-8')
    out2 = codecs.open('../preprocessed_data/'+domain+'/test_label.txt', 'w', 'utf-8')

    for text, label in zip(f1, f2):
        label = label.strip()
        tokens = parse_sentence(text)
        # if len(tokens) > 0: ## if not comment this line, the sentence number will decrease
        #     out1.write(' '.join(tokens) + '\n')
        #     out2.write(label+'\n')
        if len(tokens) > 0:
            out1.write(' '.join(tokens) + '\n')
            out2.write(label+'\n')


def preprocess(domain):
    print('\t' + domain + ' train set ...')
    preprocess_train(domain)
    print('\t' + domain + ' test set ...')
    preprocess_test(domain)


if __name__ == "__main__":
    print('Preprocessing raw review sentences ...')
    preprocess('restaurant')
    # preprocess('beer')

    for id, line in enumerate(open("../preprocessed_data/restaurant/train.txt")):
        print(line.strip())
        if id > 10:
            break
