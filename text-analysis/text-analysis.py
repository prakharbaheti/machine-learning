# -*- coding: utf-8 -*-
import pandas as pd
import sys
import csv
import re
#from nltk import word_tokenize

#nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
from urllib.request import urlretrieve

data = pd.ExcelFile('cik_list.xlsx').parse('cik_list_ajay')
stop_words =open('StopWords_Generic.txt').read().splitlines()
master_dictionary = pd.read_csv('LoughranMcDonald_MasterDictionary_2018.csv',index_col=0)
positive_dictionary = master_dictionary['Positive']
negative_dictionary = master_dictionary['Negative']
notations = {"Management's Discussion and Analysis":'MDA',"Quantitative and Qualitative Disclosures about Market Risk":'QQDMR',"Risk Factors":"RF"}
analysis_sections = ["Management's Discussion and Analysis","Quantitative and Qualitative Disclosures about Market Risk",'Risk Factors']       

for index, row in data.iterrows():
    link = ' https://www.sec.gov/Archives/' + row['SECFNAME']
    fileName = 'log.text'
    urlretrieve(link,fileName)
    file_content = open(fileName).read()
    page_tag_occurences = [m.start() for m in re.finditer('<PAGE>', file_content)]
    
    content_with_page_labels = []
    for index_tag_position,val in enumerate(page_tag_occurences):
        if 0 <= index_tag_position < len(page_tag_occurences)-1:
            end_occurance = page_tag_occurences[index_tag_position+1]
            content_with_page_labels.insert(index_tag_position,file_content[val+6:end_occurance])  
            
    content_page = content_with_page_labels[0] if len(content_with_page_labels) !=0 else ''
    for analysis_section in analysis_sections:
        if  analysis_section.upper() in content_page:
            first_index = content_page.find(analysis_section.upper())
            second_index = content_page.find('\n',first_index)
            second_index = content_page.find('\n',second_index+1)
            third_index = content_page.rfind('\n',0,first_index)
            fourth_index = content_page.find('\n',second_index+1)
            
            first_line = content_page[third_index:second_index]
            second_line = content_page[second_index:fourth_index]
            page_number_start = int(first_line[-2:])
            page_number_end = int(second_line[-2:])
            
            words_in_tokenize_from = []
            sentence_in_tokenize_form = []
            start_index = content_with_page_labels[page_number_start].find(first_line[:6])
            last_index =  content_with_page_labels[page_number_end].find(second_line[:6])
            for page in range(page_number_start,page_number_end+1):
                if page == page_number_end:
                    words_in_tokenize_from.extend(word_tokenize(content_with_page_labels[page][:last_index]))
                    sentence_in_tokenize_form.extend(sent_tokenize(content_with_page_labels[page][:last_index]))
                elif page == page_number_start:
                    words_in_tokenize_from.extend(word_tokenize(content_with_page_labels[page][start_index:]))
                    sentence_in_tokenize_form.extend(sent_tokenize(content_with_page_labels[page][start_index:]))
                else:
                    words_in_tokenize_from.extend(word_tokenize(content_with_page_labels[page]))
                    sentence_in_tokenize_form.extend(sent_tokenize(content_with_page_labels[page]))
                    
            words_in_tokenize_from =  [x.upper() for x in words_in_tokenize_from]                    
            words_in_tokenize_from = list(set(words_in_tokenize_from) - set(stop_words))
            
            positive_count = 0
            negative_count = 0
            complex_word_count = 0
            for word in words_in_tokenize_from:
                if word in positive_dictionary.index:
                    if positive_dictionary[word] > 0:
                            positive_count += 1
                elif word in negative_dictionary.index:
                    if negative_dictionary[word] > 0:
                            negative_count +=1

                if len(set('AEIOU').intersection(set(word))) >2:
                    complex_word_count += 1
                    
            polarity_score = (positive_count - negative_count)/((positive_count + negative_count) + 0.000001)
            average_sentence_length = len(words_in_tokenize_from)/len(sentence_in_tokenize_form)
            percentage_of_complex_words = float(complex_word_count/len(words_in_tokenize_from))
            fog_index = float(0.4*(average_sentence_length + percentage_of_complex_words))
            total_word_count = len(words_in_tokenize_from)
            
            row[notations[analysis_section]+'_positive_score'] = positive_count
            row[notations[analysis_section]+'_negative_score'] = negative_count
            row[notations[analysis_section]+'_polarity_score'] = polarity_score
            row[notations[analysis_section]+'_average_sentence_length'] = average_sentence_length
            row[notations[analysis_section]+'_percentage_of_complex_words'] = percentage_of_complex_words
            row[notations[analysis_section]+'_fog_index'] = fog_index
            row[notations[analysis_section]+'_complex_word_count'] = complex_word_count
            row[notations[analysis_section]+'_word_count'] = total_word_count
            with open('final.csv', "a") as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(row.to_list())
    
