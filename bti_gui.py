from utils.cosine_similarity import *
from Tkinter import *

import tkFileDialog as fd 
import ttk, os, ctypes

USER_32 = ctypes.windll.user32

SCALE_X = 0.3
SCALE_Y = 0.7

WIDTH, HEIGHT = int(SCALE_X * USER_32.GetSystemMetrics(0)), int(SCALE_Y * USER_32.GetSystemMetrics(1))


PAD_X = 5
PAD_Y = 5

OUTPUT_HOME = 'C:\Users\Mark\Desktop'


class BtiGui(Tk): 
    def __init__(self, title):
        Tk.__init__(self)

        self.title(title)
        # self.geometry('{}x{}'.format(WIDTH, HEIGHT))

        self.ideal_companies_csv = None
        self.test_companies_csv = None
        self.ideal_company_label = None
        self.non_ideal_company_label = None
        self.output_filename = None

        self.choose_ideal_companies_csv_lbl = ttk.Label(self, text="Choose Ideal Companies CSV File")
        self.choose_ideal_companies_csv_lbl.grid(row=0, column=0, padx=PAD_X, pady=PAD_Y)

        self.choose_ideal_companies_csv_btn = ttk.Button(self, text="Choose", 
            command=lambda: self.choose_file('ideal_companies_csv', self.chosen_ideal_companies_csv_lbl, title="Choose Ideal Companies CSV File")
        )

        self.choose_ideal_companies_csv_btn.grid(row=0, column=1, padx=PAD_Y, pady=PAD_X)

        self.chosen_ideal_companies_csv_lbl = ttk.Label(self, text="<No File Selected>")
        self.chosen_ideal_companies_csv_lbl.grid(row=1, column=0, columnspan=2, padx=PAD_X, pady=PAD_Y)

        ttk.Separator(self).grid(row=2, column=0, columnspan=2, padx=PAD_X, pady=PAD_Y*1.5, sticky="EW")

        self.choose_test_companies_csv_lbl = ttk.Label(self, text="Choose Test Companies CSV File")
        self.choose_test_companies_csv_lbl.grid(row=3, column=0, padx=PAD_X, pady=PAD_Y)

        self.choose_test_companies_csv_btn = ttk.Button(self, text="Choose",
            command=lambda: self.choose_file('test_companies_csv', self.chosen_test_companies_csv_lbl, title="Choose Non-Ideal Companies CSV File")
        )

        self.choose_test_companies_csv_btn.grid(row=3, column=1, padx=PAD_Y, pady=PAD_X)

        self.chosen_test_companies_csv_lbl = ttk.Label(self, text="<No File Selected>")
        self.chosen_test_companies_csv_lbl.grid(row=4, column=0, columnspan=2, padx=PAD_X, pady=PAD_Y)

        ttk.Separator(self).grid(row=5, column=0, columnspan=2, padx=PAD_X, pady=PAD_Y*1.5, sticky="EW")

        self.ideal_company_label_lbl = ttk.Label(self, text="Enter Ideal Company Label")
        self.ideal_company_label_lbl.grid(row=6, column=0, padx=PAD_X, pady=PAD_Y)

        self.ideal_company_label_entry = ttk.Entry(self)
        self.ideal_company_label_entry.grid(row=6, column=1, padx=PAD_X, pady=PAD_Y)
        
        self.non_ideal_company_label_lbl = ttk.Label(self, text="Enter Non-ideal Company Label")
        self.non_ideal_company_label_lbl.grid(row=7, column=0, padx=PAD_X, pady=PAD_Y)

        self.non_ideal_company_label_entry = ttk.Entry(self)
        self.non_ideal_company_label_entry.grid(row=7, column=1, padx=PAD_X, pady=PAD_Y)

        self.output_filename_lbl = ttk.Label(self, text="Output File Name")
        self.output_filename_lbl.grid(row=8, column=0, padx=PAD_X, pady=PAD_Y)

        self.output_filename_entry = ttk.Entry(self)
        self.output_filename_entry.grid(row=8, column=1, padx=PAD_X, pady=PAD_Y)

        ttk.Separator(self).grid(row=9, column=0, columnspan=2, padx=PAD_X, pady=PAD_Y*1.5, sticky="EW")
        
        self.run_btn = ttk.Button(self, text="Run", command=self.run_cosine_similarity)        
        self.run_btn.grid(row=10, column=0, columnspan=2, padx=PAD_Y, pady=PAD_X, sticky="EW")

    def choose_file(self, storage, label, title="Untitled Dialog"): 
        f = fd.askopenfilename(title=title)

        if f:          
            setattr(self, storage, f)   
            # storage = f

            # print "STORAGE: ", storage
            label.configure(text=os.path.split(f)[1])
        
    def get_entry_value(self, storage, entry): 
        storage = entry.get()
        # print storage

    def run_cosine_similarity(self): 
        print "IDEAL CSV ", self.ideal_companies_csv

        ideal_companies, non_ideal_companies = load_train_companies(self.ideal_companies_csv)
        test_companies = self.test_companies_csv
        ideal_label = self.ideal_company_label_entry.get()
        non_ideal_label = self.non_ideal_company_label_entry.get()
        
        ideal_tfidf_matrix, ideal_vectorizer = get_tfidf_matrix_and_vectorizer(ideal_companies)
        non_ideal_tfidf_matrix, non_ideal_vectorizer = get_tfidf_matrix_and_vectorizer(non_ideal_companies)

        ideal_tfidf = condense_tfidf_matrix(ideal_tfidf_matrix)
        non_ideal_tfidf = condense_tfidf_matrix(non_ideal_tfidf_matrix)

        self.output_filename = self.output_filename_entry.get()

        output_filename = '{}.csv'.format(self.output_filename) if not self.output_filename.lower().endswith('.csv') else self.output_filename
        similarity_csv = batch_compute_cosine_similarity((ideal_tfidf, ideal_vectorizer), (non_ideal_tfidf, non_ideal_vectorizer), test_companies, output_fn=output_filename)

        get_data_subset(OUTPUT_HOME, similarity_csv, ideal_label=ideal_label, non_ideal_label=non_ideal_label, dcol=3, ctype=1)

        os.startfile(OUTPUT_HOME)



if __name__ == "__main__": 
    app = BtiGui('BTI - Business Type Identifier')
    app.mainloop()


