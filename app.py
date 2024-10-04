from flask import Flask, render_template, request , send_file
import os
import requests
from bs4 import BeautifulSoup
from langchain_community.utilities import GoogleSerperAPIWrapper
from openai import AzureOpenAI
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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import intel_extension_for_pytorch as ipex

os.environ["MODIN_ENGINE"] = "dask"  

os.environ["SERPER_API_KEY"] = "API_KEY"

model_name = "microsoft/phi-2"
client = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = ipex.optimize(model_name, weights_prepack=False,max_len=200)

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
        outputs = model.generate(**messages, max_length=200)
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
