import numpy as np
from github import Github
from terminaltables import AsciiTable
from terminaltables import GithubFlavoredMarkdownTable
import pickle
import codecs


# Insert you username and password
g = Github("**", "***")

# Settings
number_of_reps = 100
names_of_props = ["Id", "Name", "Description", "Language", "Stars", "Forks"]
github_server_link = "https://github.com/"
last_tables_file_name = 'last_table_data.pickle'
md_file_name = 'readme.md'

# Main query
seach_query = g.search_repositories("deep-learning", sort="stars", order="desc")
results = []
for index, rep in enumerate(seach_query):
    
    # print(rep.url)  # Everything are here as json file (You can use it instead of the API)
    
    rep_prop = [index+1]
    link = github_server_link + rep.full_name
    rep_prop.append("[{}]({})".format(rep.name, link))
    rep_prop.append(rep.description)
    rep_prop.append(rep.language)
    rep_prop.append(rep.stargazers_count)
    rep_prop.append(rep.forks)

    results.append(rep_prop)
    
    if(index > number_of_reps-2):
        break

# Creating the table		
table_data = [["" for x in range(len(names_of_props))] for y in range(number_of_reps + 1)]

for i in range(len(names_of_props)):
    table_data[0][i] = names_of_props[i]
    
for i in range(number_of_reps):
    for j in range(len(names_of_props)):
        table_data[i+1][j] = results[i][j]

        
# Saving Table data (For further analysis)
with open(last_tables_file_name, 'wb') as handle:
    pickle.dump(table_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
# Generating the ascii table
table = GithubFlavoredMarkdownTable(table_data)
table_str = table.table

# Wrting the md file
with codecs.open(md_file_name, "w", "utf-8") as f:
    f.write(table_str)