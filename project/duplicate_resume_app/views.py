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
#from fpdf import FPDF
#pdf = FPDF()
#pdf.add_page()
#pdf.set_font("Times","B",10)
import matplotlib.pyplot as pyplot
import numpy as np
from mpl_toolkits import mplot3d
from PIL import Image
from docx2pdf import convert



def index(request):
    if request.method == "POST":
        print("you are here")
        for files in request.FILES.getlist("file_name"):
            handle_uploaded_file(files, str(files))
        #try:
            #convert(r"C:\Users\NRAJPURO\Documents\Casestudy\CV_datasets\\")
        #except:
            #pass
        context=read_files()
        doc=zip(context[0]['Document 1'],context[0]['Document 2'])
        for x,y in doc:
            print("doc",x,"doc2",y)
        path = settings.MEDIA_ROOT
        #img_list = os.listdir(path + '/images')
        context1={"details":zip(context[0]['Document 1'],context[0]['Document 2'],context[0]['Email id'],context[0]['similarity_index'])}
        return render(request,'grid.html',context1)
    return render(request,'Home.html')


import mimetypes
def details(request,str):
  fl_path = r"C:\Users\NRAJPURO\Documents\Casestudy\CV_datasets\\"+str
  filename = str
  fl = open(fl_path,'r',errors="ignore")#errors will remove utf error
  mime_type, _ = mimetypes.guess_type(fl_path)
  response = HttpResponse(fl, content_type="application/pdf")
  response['Content-Disposition'] = "attachment; filename=%s" % filename
  return response

def handle_uploaded_file(file, filename):
    with open(r"C:\Users\NRAJPURO\Documents\Casestudy\CV_datasets\\" + filename, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

def read_files():
    dirname=r"C:\Users\NRAJPURO\Documents\Casestudy\CV_datasets\\"
    ext=('.docx','.pdf') 
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
    #pdf = FPDF()
    #pdf.alias_nb_pages()
    #pdf.set_font('Times', '', 12)
    Doc1=[]
    Doc2=[]
    n=0
    for (a,b) in zip(temp1,temp2): 
        vect1=np.array([count_array[a]])
        vect2=np.array([count_array[b]])
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
    return data_1,L_file


def extraction1(f1_path):
    email_list=[]
    path = r"C:\Users\NRAJPURO\Documents\Casestudy\CV_datasets\\"
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
                plt.bar(x,y,width=0.2,color='indianred')
                x=saveplot(count)
                pdf.image(x[1],w = 150)
            else:
                plt.bar(x,y,width = 0.2)
                x=saveplot(count)
                pdf.image(x[1],w = 150)
            plt.xlabel("Resumes")
            plt.ylabel("Percentage Range")
            # plt.show()
            count+=1
    except Exception as e:
        print(e)


def saveplot(a):
    return plt.savefig(r"C:\Users\NRAJPURO\Documents\Casestudy\Identicals\bar"+str(a)+".png"),"bar"+str(a)+".png"
    
    
def check_validity(request):
    if request.method == "GET":
        x=request.GET.get('temp_document1')
        y=request.GET.get('temp_document2')
        data,output,output1=company_details(x)
        data1,output2,output3=company_details(y)
        context={"dict":data,"dict1":output,"dict2":output1,"dict3":data1,"dict4":output2,"dict5":output3}
        print("context",context)
        return render(request,'profile_detail.html',context)


def company_details(file):
    file_name=file
    print("file_name")
    print(file_name)
    data=[]
    data={"File Name":'',"Candidate Name":'',"Candidate Email Id":'',"Candidate Phone No":''}
    data["File Name"]=file_name
    email=get_email(file_name)
    print(get_email(file_name))
    data["Candidate Email Id"]=email
    name=get_name(file_name)
    print(get_name(file_name))
    data["Candidate Name"]=name
    phone=get_phone(file_name)
    data["Candidate Phone No"]=phone
    text=extract_work_experience(file_name)
    colleges=[]
    colleges.append(extract_college_and_university_from_text(file_name))
    #print(text)
    text1=[]
    L5=[]
    text1=extract_company_names(text)
    #print(text1)
    if text1 is not None:
        L5.append(text1[1:])
    #print(L5)
    L6 = []
    [L6.extend(sublist) for sublist in L5]
    #print(L6)        
    #print('\n')
    input_text = L6
    cleaned_text = eliminate_words(input_text)
    #print(cleaned_text)
    #print("Company details:")
    print(info(cleaned_text))
    print(info(colleges))
    #data=info(cleaned_text)
    output=info(cleaned_text)
    output1=info(colleges)
    print(data)
    return data,output,output1




def get_email(file):
    doc=aw.Document(file)
    doc.save("output.txt")
    resume_text=open("output.txt",encoding='utf-8').read()
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    matches = re.findall(email_pattern, resume_text)
    if len(matches) > 0:
        email_id = matches[0]
    else:
        email_id = "Email ID not found"
    return email_id




def get_name(file):
    doc=aw.Document(file)
    doc.save("output.txt")
    with open('output.txt', 'r', encoding='utf-8') as file:
        content = file.read()
# Remove the watermark text
    watermark = "Evaluation Only. Created with Aspose.Words. Copyright 2003-2023 Aspose Pty Ltd."
    clean_content = content.replace(watermark, '')
# Save the modified content to a new TXT file
    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write(clean_content)
    resume_text=open("output.txt",encoding='utf-8').read()
    name = ''
    lines = resume_text.split('\n')
    for line in lines:
        # Add more conditions or patterns to match the name format
        if len(line.strip().split()) >= 2:
            name = line.strip()
            break
    return name

def get_phone(f1_path):
    email_list=[]
    # Folder Path accept file name, read file content remove all punutations and return string
    path = r"C:\Users\NRAJPURO\Documents\Casestudy\CV_datasets\\"
    doc=aw.Document(f1_path)
    doc.save("output.txt")
    string=open("output.txt",encoding='utf-8').read()
    email = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+",string) #find email address
    phone= re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]',string)#Extract mobile number
    return phone[0]

def extract_work_experience(f1_path):
    doc=aw.Document(f1_path)
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

   

def extract_company_names(resume_text):
    if resume_text is not None:
        doc = nlp(resume_text)
        company_names = []
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                company_names.append(ent.text)
        return company_names
    else:
        return None





def extract_college_and_university_from_text(f1_path):
    doc=aw.Document(f1_path)
    doc.save("output.txt")
    text=open("output.txt",encoding='utf-8').read()
    college=[]
    university=[]
    college_name = None
    university_name = None
    # Define regular expressions to match patterns for college and university names
    college_regex = r'(college|institute)\s+of\s+([\w\s&.,()]+)'
    university_regex = r'(\w+\s+university)'
    # Extract college name
    college_match = re.search(college_regex, text, re.IGNORECASE)
    if college_match:
        college_name = college_match.group(2).strip()
        college.append(college_name)
    # Extract university name
    university_match = re.search(university_regex, text, re.IGNORECASE)
    if university_match:
        university_name = university_match.group(1).strip()
        university.append(university_name)
    return university,college  


def eliminate_words(text):
    size = len(text)
    cleaned=[]
    # Define the words you want to eliminate
    words_to_eliminate = ['Pvt', 'Ltd', 'Services', 'Technologies','Client','Software','Infotech','Private','Limited']
    # Create a regex pattern to match the words to eliminate
    pattern = r'\b(?:{})\b'.format('|'.join(words_to_eliminate))
    # Eliminate the words using regex substitution
    for i in range(size):
        cleaned_text = re.sub(pattern,"",text[i],flags=re.IGNORECASE)
        if cleaned_text.startswith(' '):
            cleaned_text = re.sub(r"\s+", "", cleaned_text)
        cleaned.append(cleaned_text)
    return cleaned

   

def info(company):
    company_dict={"Website":'',"Contact":'',"Email_Id":''}
    name=company
    B = len(name)
    search_query = f"{name} contact"  # Modify the search query as needed
    url = f"https://www.google.com/search?q={search_query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36"
        }
    response = requests.get(url, headers=headers) # Use a user-agent header to mimic a browser request
    soup = BeautifulSoup(response.content, "html.parser") # Step 2: Create a BeautifulSoup object to parse the search results
    first_result = soup.select_one(".g")
    if first_result:
        website_url = first_result.select_one(".yuRUbf a")["href"]
        if website_url.startswith("/url?q="):
            website_url = website_url[7:]
            # Step 4: Open the website and extract the contact information
        website_response = requests.get(website_url, headers=headers)
        website_soup = BeautifulSoup(website_response.content, "html.parser")
           # Step 5: Extract the contact information from the website
        contact_info = website_soup.get_text()
            # Step 6: Extract the address from the contact information
        address = "Address not found"
            # Try different patterns to extract the address
        address_patterns = [
            r"\b\d+\s+\w+\s+\w+,\s+\w+.*\b",
            r"\b\d+.*,\s*\w+.*\b",
            r"\b[A-Za-z]+\s*\d+.*,\s*\w+.*\b"
                ]
        for pattern in address_patterns:
            address_matches = re.findall(pattern, contact_info, re.IGNORECASE)
            if address_matches:
                address = address_matches[0]
                break
            # Step 7: Extract the contact number and email ID from the contact information
        company_dict["Website"]=website_url
        contact_number = re.search(r"\b(?:\+\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b", contact_info)
        contact_number = contact_number.group(0) if contact_number else "Contact number not found"
        company_dict["Contact"]=contact_number
        email = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", contact_info)
        email = email.group(0) if email else "Email ID not found"
        company_dict["Email_Id"]=email
        print("Website:", website_url)
            #print("Address:", address)
        print("Contact number:", contact_number)
        print("Email ID:", email)
        print("\n")
    else:
        print("No search results found")  
    return company_dict






    

    