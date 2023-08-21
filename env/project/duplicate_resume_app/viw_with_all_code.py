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



def index(request):
    if request.method == "POST":
        for files in request.FILES.getlist("file_name"):
            handle_uploaded_file(files, str(files))
        try:
            convert("C:/Users/TESHAIKH/duplicate_resume/media/resume/")
        except:
            pass
        context=read_files()
        doc=zip(context[0]['Document 1'],context[0]['Document 2'])
        for x,y in doc:
            print("doc",x,"doc2",y)
        path = settings.MEDIA_ROOT
        img_list = os.listdir(path + '/images')

        context1={"details":zip(context[0]['Document 1'],context[0]['Document 2'],context[0]['Email id'],context[0]['similarity_index'])}
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
  response = HttpResponse(fl, content_type="application/pdf")
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
   
    # for i in range(len(email_List)):
    #     L3.append(i)
    # print("L3",L3)
    n = 2
    L4=n_length_combo([x for x in L3], n) 
    L5=n_length_combo([x for x in email_List], n) 
    # print("possible document combination",L4)
    # print("possible document combination",L5)
    temp1=list(map(lambda item: item[0], L4))  
    temp2=list(map(lambda item: item[1], L4)) 
    pdf = FPDF()
    pdf.alias_nb_pages()
    
    pdf.set_font('Times', '', 12)
    Doc1=[]
    Doc2=[]
    n=0
    for (a,b) in zip(temp1,temp2): 
        vect1=np.array([count_array[a]])
        vect2=np.array([count_array[b]])
        # if cosine_similarity(np.array([count_array[a]]),vect2)>max_sim:
        Doc1.append(L_file[a])
        Doc2.append(L_file[b])
        Doc.append(L_file[a])
        Doc3.append(L_file[b])
        Cosins_Sim.append(cosine_similarity(vect1,vect2))

    for x in Cosins_Sim:
        print("similrity",x)
                    
        similarity_matrix = cosine_similarity(vect1,vect2).reshape(-1) #reshape is used to convert a matrix to array     
    genrate_pdf(temp1,temp2,count_array,L_file)
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
    # x_coordinates=[x[0] for x in L4]
    # y_coordinates=[x[1] for x in L4]
    # fig = pyplot.figure(figsize = (6,8))
    # ax= pyplot.axes(projection = '3d')
    # np.random.seed(100)
    # rng = np.random.default_rng()
    # xs = rng.uniform(x_coordinates,max(x_coordinates))
    # ys = rng.uniform(y_coordinates,max(y_coordinates))
    # zs = rng.uniform(L_similarity,max(L_similarity))
    # ax.scatter(xs, ys, zs, c = 'r', s = 50,marker=".")
    # ax.set_xlabel("Document Index") #x label
    # ax.set_ylabel("Document Index") #y label
    # ax.set_zlabel("Similarity Index")

    # # zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
    # # for zdir, x, y, z in zip(zdirs, x_coordinates,y_coordinates,L_similarity):
    # #     label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)
    # #     ax.text(x, y, z, label, zdir)
    # plt.savefig('Figure1.png',dpi=100)
    # path=r"C:\Users\TESHAIKH\duplicate_resume\Figure1.png"
    # # plt.show()

    # pdf.add_page()
    # pdf.image("Figure1.png")
    # # from PIL import Image

    # image_1 = Image.open(r'Figure1.png')
    # im_1 = image_1.convert('RGB')
    # im_1.save(r'duplicate_resume_app/output.pdf')
    # # pdf.add_page()
    # ch=8
    # pdf.add_page()
    # for i in range(0, len(df_1)):
    #     pdf.cell(w=150, h=ch,txt=df_1['Document 1'].iloc[i], border=1, ln=0, align='C')
    #     pdf.cell(w=150, h=ch,txt=df_1['Document 2'].iloc[i], border=1, ln=0, align='C')
    #     pdf.cell(w=30, h=ch, txt=df_1['similarity_index'].iloc[i].astype(str), border=1, ln=1, align='C')
    # pdf.output('duplicate_resume_app/output.pdf', 'F')
    return data_1,L_file


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


# def open_linkedin_link(request,string1):
#     pattern1="['"
#     match1=string1.find(pattern1, string1.find(pattern1))
#     pattern2="']"
#     match2=string1.find(pattern2, string1.find(pattern2))
#     string2 = string1[match1+2:match2]
#     print("new string",string2)

    
#     try:
#         from urllib.request import urlopen
#     except ImportError:
#         from urllib2 import urlopen

#     ssl._create_default_https_context = ssl._create_unverified_context
#     from urllib.request import urlopen
#     profile_urls = [] #To store the Profile URLs
#     ctr = 0 #To traverse through Google results pages
#     linkedin_url=[]
#     try:
#         while ctr < 150:
#             query = 'https://www.google.com/search?q=site:linkedin.com/in AND'+" "+'"'+string2+''+'&start='+str(ctr)
#             response = requests.get(query)
#         soup = BeautifulSoup(response.text,'html.parser')
#         time.sleep(10)
#         # print("soup===",soup)
#         for anchor in soup.find_all('a'):
#             # print("anchor=====",anchor)
#             url = anchor["href"]
#             # print("url======",url)
#             if ':linkedin.com/' in url:
#                 # print("url======",url)
#                 pattern1 = 'linkedin'
#                 match1=url.find(pattern1, url.find(pattern1))
#                 pattern2='.com'
#                 match2=url.find(pattern2, url.find(pattern2)+1)
#                 url = url[match1:match2+4]
#                 # print(url)
#                 if url not in profile_urls:
#                     profile_urls.append(url)
#             ctr = ctr+10
    
#     except Exception as e:
#         print(e)
#     for u in profile_urls:
#         query1=urllib.parse.unquote(u).replace(" ","")
#         if query1 not in linkedin_url:
#             linkedin_url.append("https//www."+query1)
#     for link in linkedin_url:
#         if link=="https//www.linkedin.com":
#             linkedin_url.remove(link)
#         if link=="https//www.":
#             linkedin_url.remove(link)
#     print(linkedin_url[-1:][0])
#     return render(request,linkedin_url[-1:][0])
    
def genrate_pdf(temp1,temp2,count_array,L_files):
    count=1
    try:
        for a,b in zip(temp1,temp2): #zip takes in iterables as argument n returns iterators
            vect1=np.array([count_array[a]])
            vect2=np.array([count_array[b]])
            similarity_matrix = cosine_similarity(vect1,vect2).reshape(-1) #reshape is used to convert a matrix to array
            # print('Resumes '+str(L_files[a])+' and '+str(L_files[b])+' matches by: '+str(similarity_matrix*100)+'%')
            # print('\n')
            pdf.cell(0,10,txt = 'Resumes '+str(L_files[a])+' and '+str(L_files[b])+' matches by: '+str(similarity_matrix*100)+'%', ln = 1,align='L')
            x = np.array((L_files[a],L_files[b]))
            y = (similarity_matrix*100)
            max_y_lim = 100
            min_y_lim = 10
            plt.ylim(min_y_lim, max_y_lim)
            if y >= 80:
                # print("Highly Matching Resumes:")
                # print(str(L_files[a]))
                # print(str(L_files[b]))
                # print('\n')
                plt.bar(x,y,width=0.2,color='indianred')
                x=saveplot(count)
                pdf.image(x[1],w = 150)
                # print("image x====",x)
                # pdf.image(x[1],w = 150)
                # print("file moved to trash",dirname+max_record_dict['resume1'])
                # shutil.move(dirname+max_record_dict['resume1'],r'C:\Users\TESHAIKH\duplicate_resume\Trash1') #shutil.moveource, destination
                # shutil.move(dirname+max_record_dict['resume2'],r'C:\Users\TESHAIKH\duplicate_resume\Trash1')
                # print("file moved to trash") shutil.move(dirname+str(L[b]),r'C:\Users\NRAJPURO\Documents\Casestudy\Trash')
            else:
                plt.bar(x,y,width = 0.2)
                x=saveplot(count)
                pdf.image(x[1],w = 150)
            plt.xlabel("Resumes")
            plt.ylabel("Percentage Range")
            # plt.show()
            count+=1
        pdf.output(r"C:\Users\TESHAIKH\duplicate_resume\media\images\FinalReport.pdf")
    except Exception as e:
        print(e)


def saveplot(a):
    return plt.savefig(r"C:\Users\TESHAIKH\duplicate_resume\bar"+str(a)+".png"),"bar"+str(a)+".png"
    
def check_validity(request):
    if request.method == "GET":
        
        x=request.GET.get('temp_document1')
        y=request.GET.get('temp_document2')
        data_resume_1=extract_work_experience(x)
        data_resume_2=extract_work_experience(y)
        context={"data_resume_1":data_resume_1,"data_resume_2":data_resume_2}
       
        print("x",x)
        print("y",y)
        # document_name_one=request.GET.get('document1')
        # document_name_two=request.GET.get('document2')
        # print("document1",document_name_one)
        # print("document2",document_name_two)
        return render(request,'profile_detail.html',context)

def extract_work_experience(f1_path):
    path = r'C:/Users/TESHAIKH/duplicate_resume/media/resume/'
    doc=aw.Document(path+f1_path)
    doc.save("output.txt")
    resume_text=open("output.txt",encoding='utf-8').read()
    
    resume_text = resume_text.replace('\n', ' ').replace('\r', '')

    # Define a regular expression pattern to match work experience sections
    pattern = r'WORK EXPERIENCE:(.*?)\b(?:Education\b|\bSkills\b|\bProjects\b|\bCertifications\b|\bAwards\b|\bLanguages\b|\bLeadership\b|\bHigher\b|\bTechnical\b|$)'

    # Search for the work experience section in the resume text
    match = re.search(pattern, resume_text, re.IGNORECASE | re.DOTALL)

    if match:
        work_experience = match.group(1).strip()
        return work_experience
    else:
        return None
    