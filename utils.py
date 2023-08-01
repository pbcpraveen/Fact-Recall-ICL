import os
import json
import requests
import argparse
import openai
from dotenv import load_dotenv
from tqdm import tqdm
from nltk import word_tokenize
import string
import pdb
from tqdm import tqdm
import pandas as pd
import numpy as np
from datasets import load_dataset
import re
import pickle
import random
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from selenium import webdriver
from threading import Thread

load_dotenv('api_key.env')
openai.api_key = os.getenv("OPENAI_API_KEY")


class QueryThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def clean(s):
    x= word_tokenize(s)
    return [i for i in x if i not in [".", ","]]

# TODO: add timeout
def chatgpt_query(query, temperature=0):
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = query,
            temperature = temperature,
#             request_timeout=120,
            )
    print(response)

    return response.choices[0].message["content"].replace('\n', ' ')

def get_num_citations(page_title):
    query = f"{page_title}"
    # Make a request to the Wikipedia API to get the page content
    url = f"https://en.wikipedia.org/w/api.php?action=parse&format=json&page={query.replace(' ', '_')}"
    response = requests.get(url)
    data = response.json()

    # Extract the HTML content from the API response
    html_content = data['parse']['text']['*']

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all the citation elements
    citation_elements = soup.find_all('sup', class_='reference')

    # Return the number of citations
    return len(citation_elements)

def scrape_birth_date(name):
    # Construct the Wikipedia search query
    query = f"{name}"
    search_url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"

    # Send a GET request to the Wikipedia page
    response = requests.get(search_url)
    if response.status_code == 404:
        return ""

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the infobox table on the page
    infobox = soup.find('table', {'class': 'infobox'})

    # Find the birth date information within the infobox
    birth_date = None
    if infobox:
        rows = infobox.find_all('tr')
        for row in rows:
            labels = row.find_all('th', {'class': 'infobox-label'})
            if labels:
                for label in labels:
                    if 'Born' in label.text:
                        data = row.find('td', {'class': 'infobox-data'})
                        if data:
                            birth_date = data.text.strip()
                            break


    return birth_date

def format_name_capital(name):
    formatted_name = ""

    # Split the name into individual words
    words = name.split()

    for i, word in enumerate(words):
        # Define a list of words to preserve in lowercase
        lowercase_words = ["de", "van", "von", "di", "del", "da", "el", "la", "le", "i", "of"]
        
        # Check if the word should be preserved in lowercase
        if word.lower() in lowercase_words and i != 0:
            formatted_name += word.lower()  # Preserve the word in lowercase
        else:
            if word == ".":
                 continue
            if "-" in word:
                word = "-".join(list(map(lambda x: x.capitalize(), word.split("-"))))
                formatted_name += word
            elif word[:2] == "mc":
                formatted_name += ("Mc" + word[2:].capitalize())
            else:
                formatted_name += word.capitalize()  # Capitalize other words
        
        

        # Add a space between words
        if i != len(words) - 1:
            formatted_name += " "
        
    return formatted_name.strip()

    
def format_name(input_name):
    input_name = input_name.replace("\n", "")
    if " ," in input_name:
        input_name = input_name.replace(" ,", ",")
    # Check if the input contains '-lrb-' and '-rrb-'
    if "-lrb-" in input_name and "-rrb-" in input_name:
        # Replace '-lrb-' and '-rrb-' with '(' and ')'
        input_name = input_name.replace("-lrb-", "(").replace("-rrb-", ")")
        input_name = input_name.replace("( ", "(").replace(" )", ")")
        formatted_name = input_name.split("(")
        
        final = " (".join([format_name_capital(formatted_name[0]), formatted_name[1].lower()])
    # Capitalize the first letter of each word
    else:
        final = format_name_capital(input_name)
    
    
    return final
def sample_random_people(count, wikibio_dataset):
    names = []
    occupations = []
    nationalities = []
    birth_year = []
    rejected = 0
    wikipedia_birth_year = []
    search_queries = []
    citations = []
    descriptions = []
    p = 0
    visited = []
    indices = list(range(len(wikibio_dataset)))
    while p < count:
        k = random.choice(indices)
        while k in visited:
            k = random.choice(indices)
        record = wikibio_dataset[k]
        visited.append(k)
        name = None
        nationality = None
        occupation = None
        birth_date = None
        search_query = None
        citation = None
        wikipedia_birth_date = None
        description = None
        rejected += 1 
        if "article_title" in record['input_text']['table']['column_header']:
            index = record['input_text']['table']['column_header'].index('article_title')
            search_query = format_name(record['input_text']['table']['content'][index])
        else:
            search_query = None
            continue
        scrapped = scrape_birth_date(search_query)
        if scrapped != None:
            wikipedia_birth_date = extract_year(scrapped)
            if wikipedia_birth_date != None: 
                citation = get_num_citations(search_query)
            else: 
                continue
        else:
            continue

        if "name" in record['input_text']['table']['column_header']:
            index = record['input_text']['table']['column_header'].index('name')
            name = record['input_text']['table']['content'][index].title()
        else:
            continue
        if "nationality" in record['input_text']['table']['column_header']:
            index = record['input_text']['table']['column_header'].index('nationality')
            nationality = record['input_text']['table']['content'][index]
        else:
            continue
        if "occupation" in record['input_text']['table']['column_header']:
            index = record['input_text']['table']['column_header'].index('occupation')
            occupation = record['input_text']['table']['content'][index]
        else:
            continue
        if "birth_date" in record['input_text']['table']['column_header']:
            index = record['input_text']['table']['column_header'].index('birth_date')
            birth_date = extract_year(record['input_text']['table']['content'][index])
        else:
            continue
        description = record['target_text']
        
        
        search_queries.append(format_name(search_query))
        names.append(name)
        occupations.append(occupation)
        nationalities.append(nationality)
        birth_year.append(birth_date)
        wikipedia_birth_year.append(wikipedia_birth_date)
        citations.append(citation)
        descriptions.append(description)
        p+=1
        rejected -= 1
        print("Accepted: ", p, " Rejected: ", rejected)
        print("===============================================================================")

    print("Number of rejected samples: ", rejected)
    df = pd.DataFrame({
        "name": names, 
        "search_query": search_queries,
        "occupation": occupations,
        "nationality": nationalities,
        "birth_year": birth_year,
        "wikipedia_birth_year": wikipedia_birth_year,
        "citation": citations,
        "description": descriptions
    })

    return df


def get_year(row):
    scrapped = scrape_birth_date(row['search_query'])
    return extract_year(scrapped)

def extract_year(string):
    pattern = r"\d{4}"  # Matches any 4-digit number
    match = re.search(pattern, string)
    if match:
        return match.group()
    else:
        return None
    
def check_year(row):
    return str(row['response_year']) == str(row['wikipedia_birth_year'])

def check_year_2(row):
    r1 = float(row['response_year'])
    r2 = float(row['wikipedia_birth_year'])
    return abs(r1-r2) < 2

def check_year_5(row):
    r1 = float(row['response_year'])
    r2 = float(row['wikipedia_birth_year'])
    return abs(r1-r2) < 5

def check_hallucinated_year(row):
    return str(row['response_year']) != str(row['wikipedia_birth_year']).split(".")[0]

def freq(lst):
    d = {}
    for i in lst:
        k = str(i)
        if d.get(k):
            d[k] += 1
        else:
            d[k] = 1
    return d

def get_diff(row):
    if not str(row['wikipedia_birth_year']).split(".")[0].isnumeric():
        return 0
    return abs(int(row['response_year'])- int(str(row['wikipedia_birth_year']).split(".")[0]))

def generate_prompt1(row):
    prompt = "{} was a {} {}. When was {} born?".format(row['name'], row['nationality'], row["occupation"], row['name'])
    return prompt


def generate_prompt2(row):
    prompt = "{} was a {} {}. Can you write very short biography about {}." \
             " Let's think step by step.".format(row['name'], row['nationality'], row["occupation"], row['name'])
    return prompt


def generate_prompt2_followup(row):
    prompt = "Now answer the following question based on you above response and what you know about {}" \
             ". \n When was {} born?".format(row['name'], row['name'])
    return prompt

def generate_prompt_generate_question(row):
    prompt="Consider the following biography.\n\"{}\"\n Now generate 3 questions based on the above text. The answers to the question must be just a year. Your response must be in the following format. \n 1. Question == Answer \n 2. Question == Answer \n 3. Question == Answer\n You must striclty follow the above constraint and syntax".format(row['description'])
    return prompt

def generate_intro_prompt(row):
    prompt = "{} was a {} {}. Now answer the following questions based the biography of {}".format(row['name'], row['nationality'], row["occupation"], row['name'])
    return prompt

def generate_example1(row):
    example = "{} a {} {} was born in {}.".format(row['name'], row['nationality'], row["occupation"], row['wikipedia_birth_year'])
    return example.replace('-Rrb-', '(').replace('-Lrb-', ')')

def generate_examples(sample):
    example = '\n'.join(sample.head(3)['example'].to_list())
    example += '\n'
    row = sample.iloc[3]
    example += '{} a {} {} '.format(row['name'].replace('-Rrb-', '(').replace('-Lrb-', ')'), row['nationality'], row["occupation"])
    return example
            