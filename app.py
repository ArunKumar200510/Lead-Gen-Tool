from flask import Flask, render_template, request , send_file
import os
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import intel_extension_for_pytorch as ipex
from langchain_community.utilities import GoogleSerperAPIWrapper
from optimum.intel import OVModelForCausalLM
from selenium import webdriver
import time
import re
import json
import modin.pandas as pd
import dask.dataframe as dd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
from csv import DictWriter

os.environ["MODIN_ENGINE"] = "dask"  

os.environ["SERPER_API_KEY"] = "API_KEY"

# Loading microsoft-phi-2 model
model_name = "microsoft/phi-2"
client = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = ipex.optimize(model_name, weights_prepack=False,max_len=512)

model_id = "microsoft/phi-2"
model_opt = OVModelForCausalLM.from_pretrained(model_id, export=True)

search = GoogleSerperAPIWrapper(gl='in')
app = Flask(__name__)


def read_csv(file_path):
    companies = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            companies.append(row)
    return companies

def scrape_content_bs(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.get_text(separator=" ", strip=True)
        else:
            print(f"Failed to fetch data from {url}. HTTP Status Code: {response.status_code}")
            return ""
    except Exception as e:
        print(f"An error occurred while fetching data from {url}: {e}")
        return ""
    
def scrape_content(url):
    content = scrape_content_bs(url)
    return content

def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_text_into_chunks(text, word_count):
    words = re.split(r'\s+', text)
    chunks = [' '.join(words[i:i + word_count]) for i in range(0, len(words), word_count)]
    print("chunk",chunks)
    return chunks


def company_names(text):
    """Extract names of Indian Retail companies from the given text"""
    company_name_list = {
        "Input_text": text,
        "Company names": ["Apple",'Google','Microsoft',"Reliance Retail", "Future Retail", "Avenue Supermarts"]
    }
    return json.dumps(company_name_list)

def extract_company_names(list_of_company_names):
    tools = [
    {
        "type": "function",
        "function": {
            "name": "company_names",
            "description": "Extract names of Indian Retail companies from the given list_of_company_names",
            "parameters": {
                "type": "object",
                "properties": {
                    "list_of_company_names": {
                        "type": "string",
                        "description": "The list_of_company_names containing company names",
                        "items": {
                            "type": "array",
                            "description": "The name of the Indian Retail company"
                        }
                    }
                },
                "required": ["list_of_company_names"],
            },
        }
    }
    ]
    messages = [
            {
                "role": "system",
                "content": "You are an AI bot that helps in extracting the Company names"
            },
        {
            "role": "user",
            "content": "Extract the names of Indian Retail companies from the following text:\n\n" + list_of_company_names
        }
    ]
    return messages

def validate_company_names(extracted_names):
    prompt = f"""
    You are an intelligent assistant. Your task is to verify the following list of company names to determine their validity as proper Indian retail company names.

    Extracted Company Names:
    {', '.join(extracted_names)}

    Guidelines:
    - Please return only the valid company names from the list.
    - Remove any names that do not conform to typical company naming conventions.

    Example of valid names: "ABC Retail Limited", "XYZ Corporation Pvt Ltd"
    Example of invalid names: "Retail", "XYZ Corp", "htmlheadmeta"
    
    Verified Company Names:
    """
    return prompt

def clean_extracted_name(validated_names): 
    lines = validated_names.split('\n')
    extracted_names = []
    for line in lines:
        match = re.match(r'^\d+\.\s(.+)$', line.strip())
        if match:
            company_name = match.group(1).strip()
            extracted_names.append(company_name)
    return extracted_names

def LinkedinSearch(extracted_names):

    linkedin=[]
    for company_name in extracted_names:
        query = f"{company_name} Company Linkedin ID"
        results = search.results(query)
        links = [result['link'] for result in results['organic']]
        linkedin.append(links[0])
        
    return linkedin



def categorize_information(info_list):
    info_dict = {
        'website': '',
        'industry': '',
        'employee_size': '',
        'associated_members': '',
        'location': '',
        'founded': '',
        'additional_info': []
    }
    
    for info in info_list:
        if info.startswith('http'):
            info_dict['website'] = info
        elif re.match(r'^\d{4}$', info): 
            info_dict['founded'] = info
        elif 'employees' in info:
            info_dict['employee_size'] = info
        elif 'LinkedIn members who’ve listed' in info:
            info_dict['associated_members'] = info
        elif 'associated members' in info:
            info_dict['associated_members'] = info
        elif re.match(r'^[A-Za-z\s]+,\s[A-Za-z\s]+$', info): 
            info_dict['location'] = info
        elif 'employees' not in info and re.match(r'^[A-Za-z\s,]+$', info):
            if info_dict['industry'] == '':
                info_dict['industry'] = info
            else:
                info_dict['additional_info'].append(info)
        else:
            info_dict['additional_info'].append(info)
            
    return info_dict

def clean_list(data_list):

    return [str(item) if item is not None else 'NA' for item in data_list]

def detect_sub_domain_and_categories(industry):
    categories_dict = {
        "retail": ["Grocery", "Food & Beverages", "Fashion", "Beauty & Personal Care", 
                   "Electronics", "Appliances", "Home & Kitchen", "Health & Wellness"],
        "tech": ["Electronics", "Software", "Hardware", "Gadgets"],
        "healthcare": ["Health & Wellness", "Medical Devices", "Pharmaceuticals"],
    }

    sub_domains_dict = {
        "retail": ["Consumer Packaged Goods (CPG)", "Fast Moving Consumer Goods (FMCG)", 
                   "Omnichannel", "Supermarkets", "D2C Brands", "Brand Aggregators", 
                   "Food & Beverages", "Restaurant Chains"],
        "tech": ["SaaS", "Cloud Computing", "Artificial Intelligence", "IoT"],
        "healthcare": ["Hospitals", "Clinics", "Pharmaceutical Companies", "Medical Research"],
    }

    categories = categories_dict.get(industry.lower(), [])

    sub_domains = sub_domains_dict.get(industry.lower(), [])
    
    return categories, sub_domains


def prompt_template(company_name, industry,about):
    categories, sub_domains = detect_sub_domain_and_categories(industry)

    prompt = f"""
    You are an intelligent assistant tasked with categorizing companies.

    Company: {company_name}
    Industry: {industry}
    About Company: {about}

    Based on the information provided, identify the appropriate categories and sub-domains for this company.

    Categories it should belong to:
    {', '.join(categories) if categories else 'No categories found'}

    Sub-Domains:
    {', '.join(sub_domains) if sub_domains else 'No sub-domains found'}

    Guidelines:
    - Return only valid categories and sub-domains relevant to the company.

    Example of valid categories: "Food & Beverages", "Restaurant Chains"
    Example of invalid categories: "Hospitals", "Fast Moving Consumer Goods (FMCG)", "SaaS"

    Categories:

    Sub-Domains:
    """
    messages = [
        {"role": "system", "content": "You are an AI bot that helps finding a categories and sub_domains of the company"},
        {"role": "user", "content": prompt}
    ]
    
    return messages
    
def sub_details(company_name,industry,about):

    response_text = prompt_template(company_name, industry, about)

    # Use regular expressions to extract categories and sub-domains from the response
    categories_pattern = re.compile(r'Categories:\n((?:- .+\n)+)')
    sub_domains_pattern = re.compile(r'Sub-Domains:\n((?:- .+\n)+)')

    categories_matches = categories_pattern.search(response_text)
    sub_domains_matches = sub_domains_pattern.search(response_text)

    if categories_matches:
        categories = [match.strip() for match in categories_matches.group(1).strip().split('\n') if match.startswith('-')]
        categories = [category[2:].strip() for category in categories]  # Remove leading '- '
    else:
        categories = []

    if sub_domains_matches:
        sub_domains = [match.strip() for match in sub_domains_matches.group(1).strip().split('\n') if match.startswith('-')]
        sub_domains = [sub_domain[2:].strip() for sub_domain in sub_domains]  # Remove leading '- '
    else:
        sub_domains = []

    if  categories and sub_domains:
        return categories,sub_domains
    else:
        response_text = prompt_template(company_name, industry, about)

        # Use regular expressions to extract categories and sub-domains from the response
        categories_pattern = re.compile(r'Categories:\n((?:- .+\n)+)')
        sub_domains_pattern = re.compile(r'Sub-Domains:\n((?:- .+\n)+)')

        categories_matches = categories_pattern.search(response_text)
        sub_domains_matches = sub_domains_pattern.search(response_text)

        if categories_matches:
            categories = [match.strip() for match in categories_matches.group(1).strip().split('\n') if match.startswith('-')]
            categories = [category[2:].strip() for category in categories]  # Remove leading '- '
        else:
            categories = []

        if sub_domains_matches:
            sub_domains = [match.strip() for match in sub_domains_matches.group(1).strip().split('\n') if match.startswith('-')]
            sub_domains = [sub_domain[2:].strip() for sub_domain in sub_domains]  # Remove leading '- '
        else:
            sub_domains = []
        return categories,sub_domains



def get_annual_revenue(snippets):


    prompt = f"""
    You are an intelligent assistant tasked with identifying the annual revenue of a company based on the provided text.

    Text:
    {snippets}

    ### Task:
    Extract the annual revenue from the given article text, focusing on the most recent data available.

    Guidelines:
    - Provide the annual revenue in INR only.
    - Ensure the revenue is from the most recent date.
    - If the annual revenue is not mentioned, estimate it based on the company's recent fiscal year.
    - Return only the annual revenue value.

    Annual Revenue:
    """

    messages = [
        {"role": "system", "content": "You are an AI bot that helps for Finding Annual Revenue of the company"},
        {"role": "user", "content": prompt}
    ]
    
    return messages

def annual_revenue_article(company_name):
    query = f"Provide the latest annual revenue of {company_name} in INR. Also, mention the fiscal year for which this revenue is reported."

    results = search.results(query)

    snippets = ""
    for result in results['organic']:
        snippets += result['snippet']
        if 'date' in result and result['date']:
            snippets += " " + result['date']
    result_revenue=get_annual_revenue(snippets)
    return result_revenue



def extract_persona(user_input):
    messages = [
        {"role": "system", "content": "You are an Intelliegent AI bot that helps in extracting specific executive-level information from a list of names, positions, and LinkedIn URLs."},
        {"role": "user", "content": f"""

    ### Task :
        - Given the following list of names, positions, and LinkedIn URLs, extract the details (name, position, LinkedIn URL) for individuals holding related positions listed under the headlines .
            
    ### Position: 
    - C-Suite
    - Head of Business
    - Founder
    - CEO (Chief Executive Officer)
    - COO (Chief Operations Officer)
    - CTO (Chief Technology Officer)
    - President - Retail Business
    - Vice President
    - Director
    - VP Partnerships
    - Executive Director
    - Managing Director & CEO
    - CMO (Chief Marketing Officer)
    - Business
            
    ### Input :{user_input}

    ### GuideLines:

    - Provide only the JSON output without any additional text.

    ### Include similar positions. For example:
    - 'Chief Executive Officer' and 'CEO'
    - 'Chief Operations Officer' and 'COO'
    - 'Vice President' and 'VP'
    - 'Managing Director & CEO' and 'MD & CEO'
    - 'President' and 'president - retail Business'


    ### sample Output Format
    The output should list only those individuals who hold the specified positions or similar positions or related positions, in the following format:

    [
    {{"name": "Name X", "position": "Position X", "linkedin_url": "URL X"}},
    {{"name": "Name Y", "position": "Position Y", "linkedin_url": "URL Y"}},
    ...
    ]

    ### Example

    **Input:**
    [
    {{"name": "John Doe", "position": "CEO (Chief Executive Officer) @ reliance ", "linkedin_url": "https://www.linkedin.com/in/johndoe"}},
    {{"name": "Jane Smith", "position": "Software Engineer @ intern", "linkedin_url": "https://www.linkedin.com/in/janesmith"}},
    {{"name": "Alice Johnson", "position": "COO (Chief Operations Officer)", "linkedin_url": "https://www.linkedin.com/in/alicejohnson"}},
    {{"name": "Bob Brown", "position": "Marketing Manager @ google", "linkedin_url": "https://www.linkedin.com/in/bobbrown"}},
    {{"name": "Eve Davis", "position": "VP Partnerships - amazon", "linkedin_url": "https://www.linkedin.com/in/evedavis"}},
    {{"name": "Eve Davis", "position": "President @ adya ", "linkedin_url": "https://www.linkedin.com/in/evedavis"}}
    ]

    **Output:**
    [
    {{"name": "John Doe", "position": "CEO (Chief Executive Officer) @ reliance ", "linkedin_url": "https://www.linkedin.com/in/johndoe"}},
    {{"name": "Alice Johnson", "position": "COO (Chief Operations Officer)", "linkedin_url": "https://www.linkedin.com/in/alicejohnson"}},
    {{"name": "Eve Davis", "position": "VP Partnerships - amazon", "linkedin_url": "https://www.linkedin.com/in/evedavis"}},
    {{"name": "Eve Davis", "position": "President @ adya ", "linkedin_url": "https://www.linkedin.com/in/evedavis"}}
    ]
    """}]

    names=[]
    positions=[]
    linkedin_urls=[]
    try:
        outputs = model.generate(**messages, max_length=512)
        # outputs = model_opt.generate(**messages, max_length=512)
        response = tokenizer.batch_decode(outputs)[0]
        result = response.choices[0].message.content
        data_list = json.loads(result)
        names = [entry['name'] for entry in data_list]
        positions = [entry['position'] for entry in data_list]
        linkedin_urls = [entry['linkedin_url'] for entry in data_list]
    except Exception as e:
        print("Error in Headline")
    return names,positions,linkedin_urls


def get_persona(input_data):
    titles = [
        'C-Suite', 'Head of Business', 'Founder', 'CEO', 'Chief Executive Officer', 'COO', 'Chief Operations Officer',
        'CTO', 'Chief Technology Officer', 'President - Retail Business', 'Vice President', 'Director', 'VP Partnerships',
        'Executive Director', 'Managing Director & CEO', 'CMO', 'Chief Marketing Officer', 'Head', "Vice President - Head Of Apparels",
        "Head FMCG", "Retail Leader", "Head Of Merchandising", "Head Customer Experience", "VP, Chief Marketing & Omnichannel Officer", "Head - Omni Operations", "Operation & Sales Head",
        "Cluster Head", "Head of Business Development", "Head- Growth & Transformation", "VP-Operations",
        "AVP- Buying and Merchandising General Merchandise category", "Zonal Business Head", "Head - Sales & Marketing",
        "Head IT", "Head - Retail Operations", "Regional Manager", "Regional Manager Retail operations & business development",
        "Head Of Ecommerce", "Regional Head Sales", "Head of Digital Product, Growth, Retention & 𝖤𝗇𝗀𝖺𝗀𝖾𝗆𝖾𝗇𝗍",
        "Head Sales & Marketing", "Retail Sales Specialist", "E-Commerce Market Place", "Head of National Retailing Operations",
        "Head Marketing", "Retail operations", "Head of Business Development and Expansion", "Omni Sports leader", "Sports leader",
        "Customer Acquisition leader", "Head Sales Operation", "Head - Sales Operations", "Manager Operations", "Sales and Store Operations",
        "Senior Manager Merchandising", "Manager - Sales", "Retail Operations Manager", "Head of Institutional Sales",
        "Manager Business Development", "Assistant General Manager - Retail Operations", "Senior General Manager- Business Insights",
        "General Manager - Retail Operations", "Director- Retail", "Retail Planner, buyer & merchandiser", "GM eCommerce", "Head of Sales and Retail"
    ]

    # Create a regex pattern for the titles
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, titles)) + r')\b', re.IGNORECASE)

    # Separate lists to store the filtered results
    names = []
    positions = []
    linkedin_urls = []

    # Filter the list based on the pattern and store the results
    for person in input_data:
        if pattern.search(person['position']):
            names.append(person['name'])
            positions.append(person['position'])
            linkedin_urls.append(person['linkedin_url'])
    
    print(names, positions, linkedin_urls)
    return names, positions, linkedin_urls

def extract_details(url):

    driver = webdriver.Chrome()
    # Login 
    driver.get('https://www.linkedin.com/login')
    time.sleep(2) 
    username_field = driver.find_element(By.ID, 'username')
    username_field.send_keys('barathraj.p2022ai-ds@sece.ac.in')
    password_field = driver.find_element(By.ID, 'password')
    password_field.send_keys('sece.ac.in')
    password_field.send_keys(Keys.RETURN)
    time.sleep(30) 
    
    # About page:
    home_page_url = str(url) + '/about'
    driver.get(home_page_url)
    
    # Title
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'org-top-card-summary__title')))
        company_name = driver.find_element(By.CLASS_NAME, 'org-top-card-summary__title').text
    except:
        company_name = ""
    
    # Overview
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'p')))
        about = driver.find_elements(By.TAG_NAME, 'p')[1].text  
    except:
        about = ""
    
    # Details
    detail = []
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'dd')))
        details = driver.find_elements(By.TAG_NAME, 'dd')
        for det in details:
            detail.append(det.text)
        clean_details = categorize_information(detail)
    except:
        clean_details = {
            'website': "",
            'industry': "",
            'employee_size': "",
            'location': "",
            'founded': "",
            'additional_info': ""
        }

    people_url = str(url) + '/people'
    driver.get(people_url)
    
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'org-people-profile-card__profile-title')))
    except:
        pass

    SCROLL_PAUSE_TIME = 4
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    i = 0
    linked_url = []
    while True:
        lnks = driver.find_elements(By.ID, f"org-people-profile-card__profile-image-{i}")
        if not lnks:
            break
        for lnk in lnks:
            linked_url.append(lnk.get_attribute('href'))
        i += 1

    titles = []
    try:
        title_elements = driver.find_elements(By.CLASS_NAME, 'lt-line-clamp--single-line')
        for tit in title_elements:
            titles.append(tit.text)
    except:
        titles = []
    
    headline = []
    try:
        head_elements = driver.find_elements(By.CLASS_NAME, 'lt-line-clamp--multi-line')
        for i, hea in enumerate(head_elements):
            if i % 2 == 0:
                headline.append(hea.text)
        clean_headline = clean_list(headline)
    except: 
        clean_headline = []
    

    url_link=clean_list(linked_url)

    persona = [{'name': n, 'position': h, 'linkedin_url': u} for n, h, u in zip(titles, clean_headline, url_link)]
    print("persona",persona)
    person_name,headline,person_url=get_persona(persona)
    
    data_list=[{'name': n1, 'position': h1, 'linkedin_url': u1} for n1, h1, u1 in zip(person_name, headline, person_url)]

    results_persona = []
    
    
    time.sleep(5)

    people_u = people_url+'/?keywords=ceo'
    print("People_url",people_u)
    driver.get(people_u)

    time.sleep(2)
    
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'org-people-profile-card__profile-title')))
    except:
        print(f"Failed to load page for title: {people_u}")
    
    SCROLL_PAUSE_TIME = 4
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    i = 0
    linked_url = []
    while True:
        lnks = driver.find_elements(By.ID, f"org-people-profile-card__profile-image-{i}")
        if not lnks:
            break
        for lnk in lnks:
            linked_url.append(lnk.get_attribute('href'))
        i += 1

    titles_list = []
    try:
        title_elements = driver.find_elements(By.CLASS_NAME, 'lt-line-clamp--single-line')
        for tit in title_elements:
            titles_list.append(tit.text)
    except:
        titles_list = []
    
    headline = []
    try:
        head_elements = driver.find_elements(By.CLASS_NAME, 'lt-line-clamp--multi-line')
        for i, hea in enumerate(head_elements):
            if i % 2 == 0:
                headline.append(hea.text)
        clean_headline = clean_list(headline)
    except:
        clean_headline = []
    
    url_link = clean_list(linked_url)

    persona = [{'name': n, 'position': h, 'linkedin_url': u} for n, h, u in zip(titles_list, clean_headline, url_link)]
    
    for person in persona:
        results_persona.append({
            'name': person['name'],
            'position': person['position'],
            'linkedin_url': person['linkedin_url']
        })

    total_list=data_list+results_persona

    # Remove duplicates by using a set to track unique entries
    unique_entries = {frozenset(item.items()): item for item in total_list}.values()

# Convert back to a list
    final_list = list(unique_entries)
    
    categories,sub_domains=sub_details(company_name,clean_details['industry'],about)

    annual_revenue = annual_revenue_article(company_name)

    total_name,total_poition,total_url=extract_persona(final_list)
    
    data = {
        'Company_name': company_name,
        'about': about,
        'categories':categories,
        'sub_domains':sub_domains,
        'website': clean_details['website'],
        'industry': clean_details['industry'],
        'employee_size': clean_details['employee_size'],
        'annual_revenue':annual_revenue,
        'location': clean_details['location'],
        'founded': clean_details['founded'],
        'additional_info': clean_details['additional_info'],
        'people_name': total_name,
        'headline':total_poition,
        'linked_url': total_url
    }
    
    driver.quit()
    return data


def get_company_name_industry(domain): 
    queries = [
        f'site:.com OR site:.org OR site:.net "{domain} Companies in India" AND ("store" OR "shop" OR "outlet" OR "consumer products" OR "sales" OR loc:"India") -("technology" OR "automobile" OR "telecom" OR "pharmaceutical" OR "Financial" OR "Manufacturer")'
        #f'site:.com OR site:.org OR site:.net "Top {domain} Companies in India" AND ("store" OR "shop" OR "outlet" OR "consumer products" OR "sales") -("technology" OR "automobile" OR "telecom" OR "pharmaceutical" OR "Financial" OR "Manufacturer")',
        # f'site:.com OR site:.org OR site:.net "Leading {domain} Companies in India" AND ("store" OR "shop" OR "outlet" OR "consumer products" OR "sales") -("technology" OR "automobile" OR "telecom" OR "pharmaceutical" OR "Financial" OR "Manufacturer")',
        # f'site:.com OR site:.org OR site:.net "{domain} Industry in India" AND ("store" OR "shop" OR "outlet" OR "consumer products" OR "sales") -("technology" OR "automobile" OR "telecom" OR "pharmaceutical" OR "Financial" OR "Manufacturer")',
        # f'site:.com OR site:.org OR site:.net "{domain} Sector in India" AND ("store" OR "shop" OR "outlet" OR "consumer products" OR "sales") -("technology" OR "automobile" OR "telecom" OR "pharmaceutical" OR "Financial" OR "Manufacturer")',
        # f'site:.com OR site:.org OR site:.net "Major {domain} Companies in India" AND ("store" OR "shop" OR "outlet" OR "consumer products" OR "sales") -("technology" OR "automobile" OR "telecom" OR "pharmaceutical" OR "Financial" OR "Manufacturer")',
        # f'site:.com OR site:.org OR site:.net "Best {domain} Companies in India" AND ("store" OR "shop" OR "outlet" OR "consumer products" OR "sales") -("technology" OR "automobile" OR "telecom" OR "pharmaceutical" OR "Financial" OR "Manufacturer")',
        # f'site:.com OR site:.org OR site:.net source:"{domain}Companiesinindia" AND ("store" OR "shop" OR "outlet" OR "consumer products" OR "sales") -("technology" OR "automobile" OR "telecom" OR "pharmaceutical" OR "Financial" OR "Manufacturer")',
        # f'site:.com OR site:.org OR site:.net intitle:"{domain} Industry Companies" loc:"India" AND ("store" OR "shop" OR "outlet" OR "consumer products" OR "sales") -("technology" OR "automobile" OR "telecom" OR "pharmaceutical" OR "Financial" OR "Manufacturer")'
    ]
    all_links=[]
    for query in queries:
        results = search.results(query)
        links = [entry['link'] for entry in results['organic']]
        all_links.extend(links)
    print(len(all_links))
    company_names=[]
    try:
        for link in all_links[:1]:
            content = scrape_content(link)
            if content:
                cleaned_text = clean_text(content)
                chunks = split_text_into_chunks(cleaned_text, 1000)
                for chunk in chunks:
                    names = extract_company_names(chunk)
                    try:
                        args = json.loads(names.tool_calls[0].function.arguments)
                        company_names.append(args)
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error parsing JSON or accessing keys: {e}")
    except Exception as e:
        print("Error parsing JSON")
    linked=[]

    try:
        result_company=[name['list_of_company_names'] for name in company_names]
        validated_names=validate_company_names(result_company)
        extracted_names=clean_extracted_name(validated_names)
        linked=LinkedinSearch(extracted_names)
        linked_updated = [url.replace('in.linkedin.com', 'www.linkedin.com') for url in linked]
    except Exception as e:
        print("Errror occurred")
        linked_updated=[]

    return extracted_names,linked_updated


@app.context_processor
def utility_processor():
    return dict(enumerate=enumerate)

#Main page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    industry = request.form['industry']
    csv_file = 'retail.csv'  
    data = read_csv(csv_file)
    return render_template('index.html', companies=data)

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_dict(orient='records')

#Extract and display details
@app.route('/company/<int:index>/<industry>')
def company_detail(index, industry):
    csv_file='retail.csv'
    data = read_csv(csv_file)
    company = data[index]
    
    # Printing company name
    print("Company_name", company['headline'])
    
    # Removing unwanted characters and splitting strings
    name = company['people_name'].strip("[]").replace("'", "").split(',')
    headline = company['headline'].strip("[]").replace("'", "").split(',')
    linkedin = company['linked_url'].strip("[]").replace("'", "").split(',')
    
    # Stripping any leading or trailing whitespaces
    name = [n.strip() for n in name]
    headline = [h.strip() for h in headline]
    linkedin = [l.strip() for l in linkedin]
    
    # Storing in the workers field
    company['workers'] = [{'name': n, 'headline': h, 'linkedin_url': u} for n, h, u in zip(name, headline, linkedin)]
    
    print(company)
    return render_template('company_detail.html', company=company, industry=industry)

@app.route('/gen_main')
def gen_main():
    return render_template('gen_main.html')

@app.route('/company_gen')
def company_gen():
    return render_template('company_gen.html')
company_session=[]
existing_names = []
existing_linkedin = []
@app.route('/company_name/submit', methods=['POST'])
def handle_submit():
    csv_file='retail.csv'
    df=pd.read_csv(csv_file)

    selected_industry = request.form.get('gen_radio')
    company_name,linkedin_url=get_company_name_industry(selected_industry)
    for name, link in zip(company_name, linkedin_url):
        if name in df['Company_name'].values:
            existing_names.append(name)
            existing_linkedin.append(link)
    
    unique_names = [name for name in company_name if name not in existing_names]
    unique_linkedin = [link for link in linkedin_url if link not in existing_linkedin]
    print(company_name)
    print(linkedin_url)
    companies = [{'name': c, 'linkedin_url': l} for c,l in zip(unique_names, unique_linkedin)]
    company_session.append(companies)
    return render_template('company_gen.html', companies=companies)

#Handle on submitting the selection
@app.route('/submit_selection', methods=['POST'])
def submit_selection():
    selected_companies = request.form.getlist('selected_companies')
    print(selected_companies)
    result_company=[]
    for select in selected_companies:
        splited_company=select.split('|')
        result_company.append(splited_company[1])
    company=[]
    for i in result_company:
        try:
            data = extract_details(i)
            company.append(data)    
        except Exception as e:
            print("Error")
    return render_template('gen_main.html', selected_companies=company)


@app.route('/submit_all', methods=['POST'])
def submit_all():
    company=[]
    print("company_sesion",company_session)
    selected_companies = company_session[0]
    print("Selected Company Details",selected_companies)
    for lin in selected_companies:
        try:
            data = extract_details(lin['linkedin_url'])
            company.append(data)    
        except Exception as e:
            print("Error")
    return render_template('gen_main.html',selected_companies=company)

@app.route('/duplicate')
def duplicate():
    try:
        companies = [{'name': c, 'linkedin_url': l} for c,l in zip(existing_names, existing_linkedin)]
    except Exception as e:
        print("No Duplicates")
    return render_template('duplicate.html',companies=companies)

@app.route('/Duplicate_extraction', methods=['POST'])
def duplicate_extraction():
    selected_companies = request.form.getlist('Duplicate_company')
    result_company = []
    for select in selected_companies:
        splited_company = select.split('|')
        result_company.append(splited_company[1])
    
    company = []
    for i in result_company:
        try:
            data = extract_details(i)
            company.append(data)  
            print("company",company)  
        except Exception as e:
            print("Error", e)
    df_new = pd.DataFrame(company)
    df_existing = pd.read_csv('retail.csv')  # Or load it from wherever you have it stored
    if set(df_new.columns) != set(df_existing.columns):
        raise ValueError("The columns of the new data do not match the existing data")
    df_existing.set_index('Company_name', inplace=True)
    df_new.set_index('Company_name', inplace=True)

    df_existing.update(df_new)

    df_combined = df_existing.combine_first(df_new)

    df_combined.reset_index(inplace=True)

    df_combined.to_csv('retail.csv', index=False)

    return render_template('index.html')


@app.route('/update', methods=['POST'])
def update():
    selected_industry = request.form.getlist('selected_companies_details')
    print("selected Company", selected_industry)
    
    for select in selected_industry:
        splited_company = select.split('#')
        print("splited", splited_company)
        
        company_data = {
            'Company_name': splited_company[0],
            'about': splited_company[1],
            'categories':splited_company[2],
            'sub_domains': splited_company[3],
            'website': splited_company[4],
            'industry': splited_company[5],
            'employee_size': splited_company[6],
            'annual_revenue': splited_company[7],
            'location': splited_company[8],
            'founded': splited_company[9],
            'additional_info':splited_company[10],
            'people_name': splited_company[11],
            'headline': splited_company[12],
            'linked_url':  splited_company[13],
            'email':[''],
            'contact_number':['']
        }
        
        df = pd.DataFrame([company_data])
        
        df.to_csv('retail.csv', mode='a', index=False, header=False)

    print("Data appended successfully.")
    return render_template('index.html')


def wrap_text(data):
    if pd.isna(data):
        return ''
    try:
        if isinstance(data, str):
            data = data.replace('"', '\\"').replace("'", '"')
            items = json.loads(data)
            if isinstance(items, list):
                return '\n'.join(items)
        return data 
    except (json.JSONDecodeError, TypeError):
        print(f"Error processing categories data: {data}")
        return data  

@app.route('/download')
def download():
    df = pd.read_csv('retail.csv')

    required_columns = ['Company_name', 'categories', 'sub_domains', 'website', 'industry', 'employee_size', 'annual_revenue', 'location', 'people_name', 'headline', 'linked_url', 'email', 'contact_number']

    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"One or more required columns are missing from the CSV. Required columns are: {required_columns}")

    with open('output1.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
    
        writer.writerow(['Company_name', 'categories', 'sub_domains', 'website', 'industry', 'employee_size', 'annual_revenue', 'location', 'people_name', 'headline', 'linked_url', 'email', 'contact_number'])

        for index, row in df.iterrows():
            # Company Name
            company_name = str(row['Company_name']).strip()
            # Categories
            categories = row['categories']
            formatted_categories = wrap_text(categories).strip()
            # Sub-domain
            sub_domain = row['sub_domains']
            formatted_sub_domain = wrap_text(sub_domain).strip()
            # Website
            website = str(row['website']).strip()
            # Industry
            industry = str(row['industry']).strip()
            # Employee size extraction
            employee_size = str(row['employee_size']).strip()
            # Annual revenue extraction
            annual_revenue = str(row['annual_revenue']).strip()
            # Location extraction
            location = str(row['location']).strip()
            # People name extraction
            people_list = str(row['people_name'])
            try:
                people_splited = json.loads(people_list.replace("'", '"'))
            except json.JSONDecodeError:
                people_splited = people_list.split(',')
            print("peple_name", people_splited)
            # Headline extraction
            clean_headline = str(row['headline'])
            try:
                headline_splited = json.loads(clean_headline.replace("'", '"'))
            except json.JSONDecodeError:
                headline_splited = clean_headline.split(',')
            print("headline", headline_splited)
            # LinkedIn URL extraction
            linkedin_ids_list = str(row['linked_url'])
            try:
                splited_link = json.loads(linkedin_ids_list.replace("'", '"'))
            except json.JSONDecodeError:
                splited_link = linkedin_ids_list.split(',')
            if not splited_link:
                splited_link = [''] * len(people_splited)
            print("linkedin_urls", splited_link)
            # Email extraction
            email_list = str(row['email']).strip() if pd.notna(row['email']) else ''
            try:
                email_splited = json.loads(email_list.replace("'", '"'))
            except json.JSONDecodeError:
                email_splited = email_list.split(',')
            if not email_splited:
                email_splited = [''] * len(people_splited)
            print("emails", email_splited)
            # Contact number Extraction
            contact_number_list = str(row['contact_number']).strip() if pd.notna(row['contact_number']) else ''
            try:
                contact_number_splited = json.loads(contact_number_list.replace("'", '"'))
            except json.JSONDecodeError:
                contact_number_splited = contact_number_list.split(',')
            if not contact_number_splited:
                contact_number_splited = [''] * len(people_splited)
            print("contact_numbers", contact_number_splited)

            if len(headline_splited) < len(people_splited):
                headline_splited.extend(['NaN'] * (len(people_splited) - len(headline_splited)))

            # Filling NaN values for LinkedIn URL based on the people_name count
            if len(splited_link) < len(people_splited):
                splited_link.extend(['NaN'] * (len(people_splited) - len(splited_link)))

            for person, headline, linkedin_id in zip(people_splited, headline_splited, splited_link):
                writer.writerow([
                    company_name, 
                    formatted_categories, 
                    formatted_sub_domain, 
                    website, 
                    industry, 
                    employee_size, 
                    annual_revenue, 
                    location, 
                    person.strip(), 
                    headline.strip(), 
                    linkedin_id.strip(), 
                    email_splited, 
                    contact_number_splited
                ])

    return send_file('output1.csv', as_attachment=True)
    
if __name__ == '__main__':
    app.run(port=8000)
