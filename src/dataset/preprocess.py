import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from datasets import Dataset
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle


PATH_CV = 'src/models/preprocess/CV.sav'
PATH_PCA = 'src/models/preprocess/PCA.sav'




def remove_special_char(dataset):
    remove_special_characters = lambda batch: {"sentence" : [re.sub(r'[!@#$(),"%^*?:–;~`0-9[\]]', '', sentence) for sentence in batch["sentence"]]}
    dataset = dataset.map(remove_special_characters, batched=True)
    return dataset

def lower_sentences(dataset):
    lower = lambda batch: {"sentence" : [sentence.lower() for sentence in batch["sentence"]]}
    dataset = dataset.map(lower, batched=True)
    return dataset


def set_format(dataset):
    dataset.set_format(type="torch", columns=["transformed_sentence","label"])


def vectorize(dataset, pca_reduction=None,from_save=False):
    if from_save:
        CV = pickle.load(open(PATH_CV, 'rb'))
        transformed_sentences = CV.transform(dataset['sentence']).toarray()
        if not pca_reduction is None:
            pca = pickle.load(open(PATH_PCA, 'rb'))
            transformed_sentences = pca.transform(transformed_sentences)
            input_shape = pca_reduction
        else:
            input_shape = transformed_sentences.shape[1]

    else:
        CV = CountVectorizer()
        transformed_sentences = CV.fit_transform(dataset['sentence']).toarray()
        if not pca_reduction is None:
            pca = PCA(n_components=100)
            transformed_sentences = pca.fit_transform(transformed_sentences)
            print("explained_variance_ratio_: ", sum(pca.explained_variance_ratio_))
            input_shape = pca_reduction
            pickle.dump(pca, open(PATH_PCA, 'wb'))

        else:
            input_shape = transformed_sentences.shape[1]

        # save model
        pickle.dump(CV, open(PATH_CV, 'wb'))

    labels = np.array(list(dataset['label'])).reshape(-1, 1) 
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(labels)
    labels = ohe.transform(labels)
    ds = Dataset.from_dict({"transformed_sentence": transformed_sentences, "label":labels})
    return ds, input_shape
