#Code with op to pdf and duplicate resumes to Trash folder
import spacy
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
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# nltk.download('all')
ps = PorterStemmer()
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Times","B",10)
from bs4 import BeautifulSoup
import os
import shutil #module in python
import requests
nlp=spacy.load("en_core_web_sm") 
os.chdir(r'C:/Users/TESHAIKH/Desktop/CV_datasets/') #chdir used to change directory
os.system('mkdir Trash') #os.system() method execute the command (a string) in a subshell...mkdir for creating new folder


def n_length_combo(arr, n):
    
    return list(combinations(arr, n))


def extraction1(f1_path):
    # Folder Path accept file name, read file content remove all punutations and return string
    path =  r'C:/Users/TESHAIKH/Desktop/CV_datasets/'
    doc=aw.Document(path+f1_path)
    doc.save("output.txt")
    string=open("output.txt",encoding='utf-8').read()
    
    email = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+",string) #find email address
    phone= re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]',string)#Extract mobile number
    remove_email=string.replace(''.join(map(str, email)),'')
    new_str=re.sub('[^a-zA-Z0-9]',' ',remove_email.replace(''.join(map(str, phone)),''))

    en = spacy.load('en_core_web_sm')
    stopwords = en.Defaults.stop_words
    en.Defaults.stop_words.add("with")
    en.Defaults.stop_words.add("from")
    
    #removing stopwords
    token_str=[]
    for token in  new_str.split(" "):
        if token.lower() not in stopwords:    #checking whether the word is not 
            token_str.append(token)
    #print(token_str)
    new_str1=' '.join(map(str,token_str))  #present in the stopword list.
            
    #converting to stem words: 
    New_List=[]
    words = word_tokenize(new_str1)
    for w in words:
        New_List.append(ps.stem(w))
    stem_str = ' '.join(map(str, New_List))    
    
    return stem_str.strip()

def saveplot(a):
    return plt.savefig("bar"+str(a)+".png"),"bar"+str(a)+".png"

def extract_work_experience(f1_path):
    path = r'C:\Users\TESHAIKH\Desktop\CV_datasets\Trash\\'
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
    path = r'C:\Users\TESHAIKH\Desktop\CV_datasets\Trash\\'
    doc=aw.Document(path+f1_path)
    doc.save("output.txt")
    text=open("output.txt",encoding='utf-8').read()
    college=[]
    university=[]
    college_name = None

    university_name = None

 

    # Define regular expressions to match patterns for college and university names

    college_regex = r'(college|institute)\s+of\s+([\w\s&.,()]+)'

    university_regex = r'(\w+\s+university)'

#

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
    name=company
    B = len(name)
    search_query = f"{name[i]} contact"  # Modify the search query as needed
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
        contact_number = re.search(r"\b(?:\+\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b", contact_info)
        contact_number = contact_number.group(0) if contact_number else "Contact number not found"
        email = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", contact_info)
        email = email.group(0) if email else "Email ID not found"
        print("Website:", website_url)
            #print("Address:", address)
        print("Contact number:", contact_number)
        print("Email ID:", email)
        print("\n")
    else:
        print("No search results found")
    
       
    
    

if __name__ == "__main__":
    report = open('report.txt','w')
    dirname= r'C:/Users/TESHAIKH/Desktop/CV_datasets/'
    ext=('.pdf','.docx') 
    L=[]
    L1=[] 
    L2=[] 
    L3=[] 
    L4=[] 
    L5=[]
    L6=[]
    temp1=[] 
    temp2=[] 
    d = dict() 
    
    print("List of files with required extensions in a folder: ")
    pdf.cell(0,10,txt = "List of files with required extensions in a folder:", ln = 1,align='L')
    #pdf.cell(0,10,txt = "\n", ln = 1,align='L')
    for files in os.listdir(dirname): 
        if files.endswith(ext): 
            print(files)
            pdf.cell(0,10,txt = files, ln = 1,align='L')
            L.append(files) #created new list named L and appended just the names of files in it
            L1.append(extraction1(files))# append each new file as string into list as an element
        else:
            continue   
    print('\n')
    pdf.cell(0,10,txt = "\n", ln = 1,align='L')
    #print("Filtered Dataset:\n "+ str(L1))
    #print('\n')
    coun_vect = CountVectorizer() #to convert text into a vector on the basis of the freq(count) of each word that occurs in the entire text. 
    count_matrix = coun_vect.fit_transform(L1) #to fit L1 in vectoriser and have vectors of text
    count_array = count_matrix.toarray() #converting matrix to array 
    
    print("Count Vetorizer Array:\n",count_array)
    pdf.cell(0,10,txt = "Count Vectorizer Array:", ln = 1,align='L')
    pdf.cell(0,10,txt = str(count_array), ln = 1,align='L')
    print('\n') 
    pdf.cell(0,10,txt = "\n", ln = 1,align='L')
    
    # finding possible combination of files for finding similiarity
    for i in range(len(count_array)):
        L3.append(i)
    n = 2
    L4=n_length_combo([x for x in L3], n) 
    print("Possible Document Combinations:\n",L4)
    pdf.cell(0,10,txt = "Possible Document Combinations:\n", ln = 1,align='L')
    pdf.cell(0,10,txt = str(L4), ln = 1,align='L')
    
    temp1=list(map(lambda item: item[0], L4))  
    temp2=list(map(lambda item: item[1], L4))  
    print('\n')
    pdf.cell(0,10,txt = "\n", ln = 1,align='L')
    print("Matching Percentage of each combination of Resumes: ")
    pdf.cell(0,10,txt = "Matching percentage of each combination of Resumes", ln = 1,align='L')
    count=1
    for a,b in zip(temp1,temp2): #zip takes in iterables as argument n returns iterators
        vect1=np.array([count_array[a]])
        vect2=np.array([count_array[b]])
        
        similarity_matrix = cosine_similarity(vect1,vect2).reshape(-1) #reshape is used to convert a matrix to array
        
        print('Resumes '+str(L[a])+' and '+str(L[b])+' matches by: '+str(similarity_matrix*100)+'%')
        print('\n')
        pdf.cell(0,10,txt = 'Resumes '+str(L[a])+' and '+str(L[b])+' matches by: '+str(similarity_matrix*100)+'%', ln = 1,align='L')
        
        x = np.array((L[a],L[b]))
        y = (similarity_matrix*100)
        max_y_lim = 100
        min_y_lim = 10
        plt.ylim(min_y_lim, max_y_lim)
    
        if y >= 80:
            print("Highly Matching Resumes:")
            print(str(L[a]))
            print(str(L[b]))
            print('\n')
            
            plt.bar(x,y,width=0.2,color='indianred')
            x=saveplot(count)
            pdf.image(x[1],w = 150)
            print("bar name=",x[1])
        
            shutil.move(dirname+str(L[a]),r'C:\Users\TESHAIKH\Desktop\CV_datasets\Trash') #shutil.moveource, destination
            shutil.move(dirname+str(L[b]),r'C:\Users\TESHAIKH\Desktop\CV_datasets\Trash')
            
        else:
            plt.bar(x,y,width = 0.2)
            x=saveplot(count)
            pdf.image(x[1],w = 150)
            print("bar name=",x[1])
        plt.xlabel("Resumes")
        plt.ylabel("Percentage Range")
        plt.show()
        count+=1
    resumes = []
    colleges=[]
    dirtrash = r'C:\Users\TESHAIKH\Desktop\CV_datasets\Trash\\'
    for files in os.listdir(dirtrash):
        resumes.append(files)
        colleges.append(extract_college_and_university_from_text(files))
        text=extract_work_experience(files)
        #print(text)
        #print("List:")
        #print(extract_company_names(text))
        new_list = extract_company_names(text)
        if new_list is not None: 
            L5.append(new_list[1:])
            #print("\n")
    #converting double list of L5 to single L6 for further use  
    #print(L5)
    L6 = []
    [L6.extend(sublist) for sublist in L5]
    #print(L6)        
    print('\n')
    input_text = L6

    cleaned_text = eliminate_words(input_text)
    #print(cleaned_text)
       
    A=len(cleaned_text)
    
 # search company in excel
    #info(cleaned_text)
    print("Company details:")
    for i in range(len(resumes)):
        print(resumes[i])
        print(cleaned_text[i])
        info(cleaned_text)
    print("\n")
    print("College details:")
    for i in range(len(colleges)):
        print(resumes[i])
        print(colleges[i])
        info(colleges)
        
   
    # pdf.image("bar1.png",w = 150)
    # pdf.image("bar2.png",w = 150)
    # pdf.output("Final Report1.pdf")
