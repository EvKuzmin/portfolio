import json
import requests


    
response = requests.get('https://api.crossref.org/works/10.1021/10.1134/S1062739120060113')
if response:
    print(1)
else: 
    print(0)
{'Institute of Chemica...90, Russia': 1, 'Novosibirsk State Un...90, Russia': 4, 'Department of Physic...rsk Russia': 2, 'Sobolev Institute of...rsk Russia': 2, 'Bayerisches Geoinsti...th Germany': 3, 'Photon Sciences Deut...rg Germany': 3, 'Vereshchagin Institu...cow Russia': 1, 'Department of Higher...on, Russia': 2, 'Department of Physic...90, Russia': 2, 'Nikolaev Institute o... Chemistry': 6, 'Siberian Branch of t...f Sciences': 6, '630090 Novosibirsk': 16, 'Russian Federation': 11, 'Novosibirsk State University': 14, ...}