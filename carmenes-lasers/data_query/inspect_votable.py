import requests
import xml.etree.ElementTree as ET
import pandas as pd

ssap_url = "http://dc.g-vo.org/carmenes/q/ssa/ssap.xml"

params = {
    "REQUEST": "queryData",
    "POS": "180,0",
    "SIZE": 90,
}

response = requests.get(ssap_url, params=params)

# Print raw response (first 2000 characters)
print("Raw XML Response:")
print(response.text[:2000])
print("\n...")

root = ET.fromstring(response.content)

namespaces = {'vo': 'http://www.ivoa.net/xml/VOTable/v1.3'}

# Print all FIELD (column) definitions
print("Available columns:")
for field in root.findall('.//vo:FIELD', namespaces):
    col_name = field.get('name')
    col_type = field.get('datatype')
    print(f"  {col_name} ({col_type})")

