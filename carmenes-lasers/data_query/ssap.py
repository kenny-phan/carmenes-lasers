import requests
import xml.etree.ElementTree as ET
import csv

ssap_url = "http://dc.g-vo.org/carmenes/q/ssa/ssap.xml"

# Query the service
params = {
    "REQUEST": "queryData",
    "POS": "180,0",
    "SIZE": 90,
}

print("Querying SSAP service...")
response = requests.get(ssap_url, params=params)

# Parse VOTable XML
root = ET.fromstring(response.content)

# Define namespaces
namespaces = {
    'vo': 'http://www.ivoa.net/xml/VOTable/v1.3',
}

# Extract table data
metadata = []

# Find all TABLEDATA rows
tabledata = root.find('.//vo:TABLEDATA', namespaces)
if tabledata is not None:
    for tr in tabledata.findall('vo:TR', namespaces):
        row = {}
        tds = tr.findall('vo:TD', namespaces)
        
        # Assuming column order; adjust based on actual service
        if len(tds) >= 3:
            row['ssa_targname'] = tds[0].text
            row['location_ra'] = tds[1].text
            row['location_dec'] = tds[2].text
            metadata.append(row)

# Save to CSV
with open("/datax/scratch/ktp/spectrum_metadata.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=['ssa_targname', 'location_ra', 'location_dec'])
    writer.writeheader()
    writer.writerows(metadata)

print(f"Extracted {len(metadata)} spectra metadata")
print(f"Saved to spectrum_metadata.csv")

