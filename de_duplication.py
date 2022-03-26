import spacy
import pandas as pd
from time import perf_counter
from datasketch import MinHash, MinHashLSH

# Parameters
threshold = 0.8
num_perms = 16
nlp = spacy.load('en_core_web_sm')

# Dicts to store statistics
exact_cosine_dict = {'BuildTime': 0, 'QueryTime': None, 'TotalTime': None, 'Duplicates': 0, 'Parameters': '-'}
exact_jaccard_dict = {'BuildTime': 0, 'QueryTime': None, 'TotalTime': None, 'Duplicates': 0, 'Parameters': '-'}
lsh_jaccard_dict = {'BuildTime': None, 'QueryTime': None, 'TotalTime': None, 'Duplicates': 0, 'Parameters': ''}


# Read file and skip first row
def load_data(subset):
    print('Reading csv...')
    file = ''
    if subset == 'train':
        file = 'datasets/q2a/corpusTrain.csv'
    elif subset == 'test':
        file = 'datasets/q2a/corpusTest.csv'
    df = pd.read_csv(file, usecols=['Content'])
    return df


# Naive cosine similarity using SpaCy
def cosine_naive(set_1, set_2):
    print('Calculating exact cosine duplicates...')
    query_time_start = perf_counter()
    neighbors = 0
    for doc_2 in set_2:
        doc_2 = nlp(doc_2)
        for doc_1 in set_1:
            doc_1 = nlp(doc_1)
            if doc_1.similarity(doc_2) > threshold:
                neighbors += 1
                print('Exact cosine duplicates:', neighbors)
                break
    query_time_end = perf_counter()
    query_time = query_time_end - query_time_start
    exact_cosine_dict['QueryTime'] = query_time
    exact_cosine_dict['TotalTime'] = query_time
    exact_cosine_dict['Duplicates'] = neighbors
    return neighbors


# Naive jaccard distance
def jaccard_naive(set_1, set_2):
    print('Calculating exact jaccard similarity...')
    neighbors = 0
    query_time_start = perf_counter()
    for doc_2 in set_2:
        words_doc_2 = set(doc_2.lower().split())
        for doc_1 in set_1:
            words_doc_1 = set(doc_1.lower().split())
            intersection = words_doc_1.intersection(words_doc_2)
            union = words_doc_1.union(words_doc_2)
            jac_sim = float(len(intersection)) / len(union)
            if jac_sim > 0.8:
                neighbors += 1
                print('Exact Jaccard duplicates:', neighbors)
                break
    query_time_end = perf_counter()
    query_time = query_time_end - query_time_start
    exact_jaccard_dict['QueryTime'] = query_time
    exact_jaccard_dict['TotalTime'] = query_time
    exact_jaccard_dict['Duplicates'] = neighbors
    return neighbors


# Compute MinHash lsh
def minhash_lsh(set_1, set_2):
    # Time how long it takes to build the LSH index
    print('Calculating duplicates with MinHash LSH...')
    build_time_start = perf_counter()
    # Initialize LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perms)
    neighbors = 0
    # Variable to index documents in the LSH structure
    doc_id = 0
    for train_doc in set_1:
        # Compute MinHash for every document in the train set
        train_minhash = MinHash(num_perm=num_perms)
        doc_1 = set(train_doc.lower().split())
        for word in doc_1:
            train_minhash.update(word.encode('utf8'))
        lsh.insert(str(doc_id), train_minhash)
        doc_id += 1
    # Store index build time in the dict
    build_time_end = perf_counter()
    build_time = build_time_end - build_time_start
    lsh_jaccard_dict['BuildTime'] = build_time
    # Now compute minhashes of the test set and perform query against the train set
    query_time_start = perf_counter()
    for test_doc in set_2:
        test_minhash = MinHash(num_perm=num_perms)
        doc_2 = set(test_doc.lower().split())
        for word in doc_2:
            test_minhash.update(word.encode('utf8'))
        result = lsh.query(test_minhash)
        if len(result) > 0:
            neighbors += 1
    query_time_end = perf_counter()
    query_time = query_time_end - query_time_start
    lsh_jaccard_dict['QueryTime'] = query_time
    lsh_jaccard_dict['TotalTime'] = build_time + query_time
    lsh_jaccard_dict['Duplicates'] = neighbors
    lsh_jaccard_dict['Parameters'] = 'Permutations: ' + str(num_perms)
    return neighbors


def main():
    # Load data
    train_set = load_data(subset='train')
    test_set = load_data(subset='test')
    # Perform MinHash LSH
    minhash_neighbors = minhash_lsh(set(train_set['Content']), set(test_set['Content']))
    print('Duplicates (minHash LSH) found: ', minhash_neighbors)

    # Now let's try jaccard similarity

    jaccard_duplicates = jaccard_naive(train_set, test_set)
    print('Exact Jaccard duplicates', jaccard_duplicates)

    cosine_duplicates = cosine_naive(train_set, test_set)
    print('Exact cosine duplicates', cosine_duplicates)

    print('Results:')
    print('Exact cosine stats:', exact_cosine_dict)
    print('Exact jaccard stats:', exact_jaccard_dict)
    print('LSH jaccard stats:', lsh_jaccard_dict)


main()
