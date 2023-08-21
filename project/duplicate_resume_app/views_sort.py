from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
import spacy
nlp=spacy.load("en_core_web_sm") 
import re 
import pathlib 
import aspose.words as aw 
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from itertools import chain 
import pandas as pd 
import numpy as np
import nltk
import os
from django.conf import settings
from django.shortcuts import render
from django.templatetags.static import static
import requests
from bs4 import BeautifulSoup
import certifi
import urllib.request
import ssl
import time
import re
# from tabulate import tabulate
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from matplotlib import image as mpimg
 
import matplotlib.pyplot as plt
import numpy as np
import math  
# nltk.download('all')
ps = PorterStemmer()
from itertools import combinations
from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Times","B",10)
import matplotlib.pyplot as pyplot
import numpy as np
from mpl_toolkits import mplot3d
from PIL import Image
from docx2pdf import convert
# import shutil #module in python
# os.chdir(r'C:\Users\TESHAIKH\duplicate_resume') #chdir used to change directory
# os.system('mkdir Trash1') #os.system()

def index(request):
    if request.method == "POST":
        for files in request.FILES.getlist("file_name"):
            handle_uploaded_file(files, str(files))
        try:
            convert("C:/Users/TESHAIKH/duplicate_resume/media/resume/")
        except:
            pass
        context=read_files()
        print("contex=",context)
        path = settings.MEDIA_ROOT
        img_list = os.listdir(path + '/images')
        
        # context1={"doc1":context[0]['resume1'],"doc2":context[0]['resume2'],"similarity_index":round(list(context[0]['similarity_index'])[0][0]*100,0),"name":context[1],"path":str(img_list[0])}
        context1={"doc1":context[0]['resume1'],"doc2":context[0]['resume2'],"similarity_index":round(list(context[0]['similarity_index'])[0][0]*100,0)}
        print(context1)
        return render(request,'grid.html',context1)
    return render(request,'index.html')

def handle_uploaded_file(file, filename):
    # new_filename=convert(r'duplicate_resume_app/resume/'+filename)
    with open('C:/Users/TESHAIKH/duplicate_resume/media/resume/' + filename, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)



import mimetypes
def details(request,str):
  fl_path = 'C:/Users/TESHAIKH/duplicate_resume/media/resume/'+str
  filename = str
  fl = open(fl_path,'r',errors="ignore")#errors will remove utf error
  mime_type, _ = mimetypes.guess_type(fl_path)
  response = HttpResponse(fl, content_type="applic'ation/pdf")
  response['Content-Disposition'] = "attachment; filename=%s" % filename
  return response

def handle_uploaded_file(file, filename):
    # new_filename=convert(r'duplicate_resume_app/resume/'+filename)
    with open('C:/Users/TESHAIKH/duplicate_resume/media/resume/' + filename, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

def read_files():
    dirname=r"C:/Users/TESHAIKH/duplicate_resume/media/resume/"
    # dirname=convert(dirname1)

    ext=('.pdf') 
    L1=[] 
    L2=[] 
    L3=[] 
    L4=[]
    L5=[]
    L_file=[]
    L_similarity=[]
    temp1=[] 
    temp2=[] 
    Doc=[]
    Doc3=[]
    Cosins_Sim=[]
    email_List=[]
    
    d = dict()
    
    for files in os.listdir(dirname): 
        if files.endswith(ext): 
            if files not in L1:
                temp=extraction1(files)
                L1.append(temp[0])
                email_List.append(temp[1])
                L_file.append(files)
        else:
            continue
    coun_vect = CountVectorizer()
    count_matrix = coun_vect.fit_transform(L1)
    count_array = count_matrix.toarray()
    for i in range(len(count_array)):
        L3.append(i)
   
    n = 2
    L4=n_length_combo([x for x in L3], n) 
    L5=n_length_combo([x for x in email_List], n) 
    
    temp1=list(map(lambda item: item[0], L4))  
    temp2=list(map(lambda item: item[1], L4)) 
    pdf = FPDF()
    pdf.alias_nb_pages()
    
    pdf.set_font('Times', '', 12)
    Doc1=[]
    Doc2=[]
    max_record_dict={'resume1':[],
                     'resume2':[],
                     'similarity_index':[]}
    max_sim=0
    for (a,b) in zip(temp1,temp2): 
        vect1=np.array([count_array[a]])
        vect2=np.array([count_array[b]])
        # if cosine_similarity(np.array([count_array[a]]),vect2)>max_sim:
        max_sim=cosine_similarity(vect1,vect2)
        
        max_record_dict['resume1'] +=[L_file[a]]
        max_record_dict['resume2'] +=[L_file[b]]
        max_sim=round([max_sim][0][0][0]*100,0)
        max_record_dict['similarity_index'] +=[max_sim]
                                               
        Doc1.append(L_file[a])
        Doc2.append(L_file[b])
        Doc.append(L_file[a])
        Doc3.append(L_file[b])
        Cosins_Sim.append(cosine_similarity(vect1,vect2))

        similarity_matrix = cosine_similarity(vect1,vect2).reshape(-1) #reshape is used to convert a matrix to array     
    first_key = list(max_record_dict.values())[0]   
    second_key= list(max_record_dict.values())[1]   
    third_key   = list(max_record_dict.values())[2]  
    zipped = zip(first_key, second_key,third_key) 
    zipped = list(zipped)
    sorted_dictionary={
        'resume1':[],
                     'resume2':[],
                     'similarity_index':[]

    }
    res = list(sorted(zipped, key = lambda x: x[2],reverse=True))
    # print(res)
    
    # image_path=get_max_sim(max_record_dict)
    for a in Cosins_Sim:
        for b in a:
            for c in b:
                L_similarity.append(round(c*100,0))
                
    import pandas as pd
    data_1 = {'Document 1': Doc,
              'Document 2': Doc3,
              'Email id':L5,
              'similarity_index': L_similarity
          }
    df_1 = pd.DataFrame(data_1)
    return max_record_dict,L_file


def extraction1(f1_path):
    email_list=[]
# Folder Path accept file name, read file content remove all punutations and return string
    path = r"C:/Users/TESHAIKH/duplicate_resume/media/resume/"
    doc=aw.Document(path+f1_path)
    doc.save("output.txt")
    string=open("output.txt",encoding='utf-8').read()
    email = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+",string) #find email address

    phone= re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]',string)#Extract mobile number
    remove_email=string.replace(''.join(map(str, email)),'')
    new_str=re.sub('[^a-zA-Z0-9]',' ',remove_email.replace(''.join(map(str, phone)),''))
    
    en = spacy.load('en_core_web_sm')
    stopwords = en.Defaults.stop_words
    nlp.Defaults.stop_words.add("with")
    nlp.Defaults.stop_words.add("from")
    token_str=[]
    for token in  new_str.split(" "):
        if token.lower() not in stopwords:    #checking whether the word is not 
            token_str.append(token)

    new_str1=' '.join(map(str,token_str))                    #present in the stopword list.
           
#     converting to stem words: 
    New_List=[]
    words = word_tokenize(new_str1)
    for w in words:
        New_List.append(ps.stem(w))
    stem_str = ' '.join(map(str, New_List))
    return stem_str.strip(),email

def n_length_combo(arr, n):
# using set to deal # with duplicates 
    return list(combinations(arr, n))

'''genrate graph for highest similarty index '''
def get_max_sim(max_record_dict):
    count=1
    try:
        vect1=max_record_dict['vect1']
        vect2=max_record_dict['vect2']
        similarity_matrix = cosine_similarity(vect1,vect2).reshape(-1) #reshape is used to convert a matrix to array
        print('\n')
        # pdf.cell(0,10,txt = 'Resumes '+str(max_record_dict['resume1'])+' and '+str(max_record_dict['resume2'])+' matches by: '+str(similarity_matrix*100)+'%', ln = 1,align='L')
        x = max_record_dict['resume1'],max_record_dict['resume2']
        y = (similarity_matrix*100)
        max_y_lim = 100
        min_y_lim = 10
        plt.ylim(min_y_lim, max_y_lim)
        if y >= 80:
            print('\n')
            plt.bar(x,y,width=0.2,color='indianred')
            x=saveplot(count)
            
        else:
            plt.bar(x,y,width = 0.2)
            x=saveplot(count)
           
    except Exception as e:
        print(e)
    return x

def saveplot(a):
    plt.savefig(r"C:\Users\TESHAIKH\duplicate_resume\media\images\bar"+str(a)+".png")
    path=r"C:\Users\TESHAIKH\duplicate_resume\media\images\bar"+str(a)+".png"
    return path

