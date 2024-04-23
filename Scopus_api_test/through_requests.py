import json
import requests


list_of_doi = list()
count = 200
for i in range(0, 2988, count):
    response = requests.get('https://api.elsevier.com/content/search/scopus',
        params={'apiKey': 'd3b1ecfcb3c960c8cd0ed34b2c7ec1e1',
            'query': 'AFFIL ( "Novosibirsk State University" ) AND (PUBYEAR = 2020) ',
            'field': 'prism:doi',
            'count': str(count),
            'start': str(i),},
        headers={'Accept': 'application/json'},
    )
    json_response = response.json()
    list_entry = json_response["search-results"]["entry"]
    for entry in list_entry:
        if 'prism:doi' in entry: 
            list_of_doi.append(entry['prism:doi'])

print(len(list_of_doi))

#with open( 'doi.txt', 'w' ) as f:
#    for item in list_of_doi:
#        f.write("%s\n" % item)

count_affiliation = dict()
keys_for_affilation = list()

for doi in list_of_doi:
    response = requests.get(f'https://api.crossref.org/works/{doi}')
    if response:
        pass
    else:
        break
    json_response = response.json()
    if "author" in json_response["message"]: 
        list_of_authors = json_response["message"]["author"]
    else:
        break
    for author in list_of_authors:
        if "affiliation" in author:
            list_of_affiliations = author["affiliation"]
        else:
            break
        for affiliation in list_of_affiliations:
            if 'name' in affiliation:
                if affiliation['name'] in count_affiliation:
                    count_affiliation[ affiliation['name'] ] = count_affiliation[ affiliation['name'] ] + 1
                else:
                    count_affiliation[ affiliation['name'] ] = 1
                    keys_for_affilation.append(affiliation['name'])
            else:
                break
        
print(keys_for_affilation)
print(count_affiliation)
print(len(keys_for_affilation))

with open( 'keys.txt', 'w', encoding='utf-8') as g:
    for item in keys_for_affilation:
        g.write("%s\n" % item)
with open( 'aff.txt', 'w', encoding='utf-8') as h:
    for item in count_affiliation:
        h.write(f"{item}   --------  {count_affiliation[item]}\n")
