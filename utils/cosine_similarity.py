from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

import tkFileDialog as fd
import csv, re, sys, os
import numpy as np

maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True


def add_vectors(matrix): 
    """
    Add each vector in a matrix to create a single total vector.
    """
    ncols = matrix.shape[1] # number of elements/columns in matrix
    total = np.zeros(ncols) # create an empty vector with ncols columns

    for vector in matrix: 
        total += vector # add current vector to total vector
    return total


def condense_tfidf_matrix(matrix): 
    """
    Condense a matrix into a single average vector.
    """
    nrows = matrix.shape[0] # number of vectors in matrix
    if nrows == 1:
        return matrix
    return add_vectors(matrix) / nrows # create a single total vector and divide by nrows to get average vector


def get_tfidf_matrix_and_vectorizer(train_set): 
    """
    Train TfidfVectorizer with all of the documents in train_set
    and return the term-frequency-inverse-document-frequency matrix resulting from training 
    the vectorizer. Also return the vectorizer.
    """
    tfidf_vectorizer = TfidfVectorizer() # instantiate untrained vectorizer
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_set) # train vectorizer with trainining documents
    return tfidf_matrix, tfidf_vectorizer


def get_cosine_similarity(tfidf_train, tfidf_test):
    """
    Calculate cosine similarity between the trained average tfidf vector and
    the test tfidf vector.
    """
    return cosine_similarity(tfidf_train, tfidf_test)[0][0]


def load_csv(csv_file, header=False): 
    """
    Load csv file contents by chunks. 
    """
    with open(csv_file, 'rb') as f: 
        reader = csv.reader(f, delimiter=',')
        if header: 
            next(reader) # if the csv has a header, skip it
        for row in reader:
            # yield each row in the csv file one at a time
            yield row

def sanitize_text(s): 
    """
    Remove all characters that are not alphanumeric to avoid encoding errors caused by weird characters.
    """
    return ' '.join(re.findall(r'[a-zA-Z0-9]+', s))
     
def batch_compute_cosine_similarity(ideal, non_ideal, test_set, output_fn='output.csv'): 
    """
    Compute cosine similarities for all companies. Compare companies with ideal (b2b) company and with non-ideal (b2c)
    company.
    """
    ideal_tfidf, ideal_vectorizer = ideal # ideal_tfidf is a vector trained with b2b keywords, ideal_vectorizer is the model used to generate ideal_tfidf
    non_ideal_tfdif, non_ideal_vectorizer = non_ideal # non_ideal_tfidf is a vector trained with b2c keywords, non_ideal_vectorizer is the model used to generate non_ideal_tfidf

    # open a csv file for writing results 
    with open(output_fn, 'wb') as f: 
        writer = csv.writer(f, delimiter=',')

        writer.writerow(['Test Company', 'Ideal Company', 'Non-Ideal Company', 'Difference']) # output header

        # iterate the test companies         
        for company_name, company_content in load_csv(test_set): 
            company_content = sanitize_text(company_content) # get the company description/keywords

            ideal_test_tfidf = ideal_vectorizer.transform([company_content]) # generate a tfidf vector for this company using ideal_vectorizer
            non_ideal_test_tfidf = non_ideal_vectorizer.transform([company_content]) # generate a tfidf vector for this company using non_ideal_vectorizer

            ideal_cosim = get_cosine_similarity(ideal_tfidf, ideal_test_tfidf) # get cosine similarity between ideal company and test company
            non_ideal_cosim = get_cosine_similarity(non_ideal_tfdif, non_ideal_test_tfidf) # get cosine similarity between non-ideal company and test company

            writer.writerow([company_name, ideal_cosim, non_ideal_cosim, ideal_cosim - non_ideal_cosim]) # write company name and similarities to output csv file
    return output_fn # return name of the output csv file


def load_train_companies(companies_csv): 
    """
    Load b2b, b2c training company descriptions/keywords
    """
    with open(companies_csv, 'rb') as f: 
        rows = [row for row in csv.reader(f, delimiter=',')]
        return [sanitize_text(row[1]) for row in rows], [sanitize_text(row[2]) for row in rows]

def get_data_subset(output_home, csv_file, ideal_label='ideal output', non_ideal_label='non-ideal output', dcol=0, ctype=1, header=True): 
    
    with open(os.path.join(output_home, '{}.csv'.format(ideal_label)), 'wb') as ideal_csv, open(os.path.join(output_home, '{}.csv'.format(non_ideal_label)), 'wb') as non_ideal_csv:
        iwriter = csv.writer(ideal_csv, delimiter=',') 
        nwriter = csv.writer(non_ideal_csv, delimiter=',')

        iwriter.writerow(['Test Company', 'Ideal Company', 'Non-Ideal Company', 'Difference'])
        nwriter.writerow(['Test Company', 'Ideal Company', 'Non-Ideal Company', 'Difference'])

        for row in load_csv(csv_file, header=header): 
            if np.sign(float(row[dcol])) == np.sign(ctype): 
                iwriter.writerow(row)
            else: 
                nwriter.writerow(row)
                

# FINISHED_FIRST_STEP = False

# if not FINISHED_FIRST_STEP: 
#     companies_csv = fd.askopenfilename(title='Choose ideal companies csv')  # file dialog to select ideal b2b, b2c company descriptions

#     ideal_companies, non_ideal_companies = load_train_companies(companies_csv) # get b2b, b2c company descriptions


#     ideal_tfidf_matrix, ideal_vectorizer = get_tfidf_matrix_and_vectorizer(ideal_companies) # get ideal tfidf matrix and vectorizer trained with ideal (b2b) companies
#     non_ideal_tfidf_matrix, non_ideal_vectorizer = get_tfidf_matrix_and_vectorizer(non_ideal_companies) # get non-ideal tfidf matrix and vectorizer trained with ideal (b2c) companies


#     ideal_tfidf = condense_tfidf_matrix(ideal_tfidf_matrix) # create an average tfidf vector for b2b companies
#     non_ideal_tfidf = condense_tfidf_matrix(non_ideal_tfidf_matrix) # create an average tfidf vector for b2c companies

#     test_companies = fd.askopenfilename(title="Choose test companies file") # file dialog to select test companies (collected by Colby)

#     output_fn = raw_input('Enter company similarities output file name: ')
#     similarity_csv = batch_compute_cosine_similarity((ideal_tfidf, ideal_vectorizer), (non_ideal_tfidf, non_ideal_vectorizer), test_companies, output_fn=output_fn) # compute all cosine similarities


# ideal_label = raw_input('Enter ideal output label (e.g. b2b companies): ')
# non_ideal_label = raw_input('Enter non-ideal output label (e.g. b2c companies): ')

# company_categorization_csv = fd.askopenfilename(title='Choose company categorization file')
# get_data_subset(company_categorization_csv, ideal_label=ideal_label, non_ideal_label=non_ideal_label, dcol=3, ctype=1)





# In the future split_company_types will load the output csv file from batch_compute_cosine_similarity
# and will split companies into b2b and b2c according to a user-defined threshold


# def split_company_types(similarity_csv, ideal_output='b2b_companies.csv', non_ideal_output='other_companies.csv', ideal_th=0.5, header=True): 
#     with open(ideal_output, 'wb') as ideal_csv_obj, open(non_ideal_output, 'wb') as non_ideal_csv_obj: 
#         ideal_writer = csv.writer(ideal_csv_obj, delimiter=',')
#         non_ideal_writer = csv.writer(non_ideal_csv_obj, delimiter=',')

#         header = ['Company', 'Similarity']

#         ideal_writer.writerow(header)
#         non_ideal_writer.writerow(header)

#         for cn, i_sim, ni_sim in load_csv(similarity_csv, header=header): 
#             if float(i_sim) >= ideal_th: 
#                 ideal_writer.writerow([cn, i_sim])
#             else: 
#                 non_ideal_writer.writerow([cn, ni_sim])


# split_company_types(similarity_csv, header=False)