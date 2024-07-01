import streamlit as st
import joblib
import pandas as pd
from urllib.parse import urlparse, parse_qs

#importation du modèle.
model = joblib.load('phishing_model.pkl')

#Extraction des caractéristiques dont nous nous sommes servis pour créer le modèle.
def extract_features(url):
    parsed_url = urlparse(url)
    domain_parts = parsed_url.netloc.split('.')
    subdomain_parts = domain_parts[:-2] if len(domain_parts) > 2 else []
    path = parsed_url.path if parsed_url.path else "/"
    query_components = parse_qs(parsed_url.query)

    features = {
        'NumDots': url.count('.'),
        'SubdomainLevel': len(subdomain_parts),
        'PathLevel': path.count('/') - 1,
        'UrlLength': len(url),
        'NumDash': url.count('-'),
        'NumDashInHostname': parsed_url.netloc.count('-'),
        'AtSymbol': '@' in url,
        'TildeSymbol': '~' in url,
        'NumUnderscore': url.count('_'),
        'NumPercent': url.count('%'),
        'NumQueryComponents': len(query_components),
        'NumAmpersand': url.count('&'),
        'NumHash': url.count('#'),
        'NumNumericChars': sum(c.isdigit() for c in url),
        'NoHttps': parsed_url.scheme != 'https',
        'RandomString': any(c.isalnum() for c in url),
        'IpAddress': parsed_url.netloc.replace('.', '').isdigit(),
        'DomainInSubdomains': any(part in subdomain_parts for part in domain_parts),
        'DomainInPaths': any(part in path for part in domain_parts),
        'HttpsInHostname': 'https' in parsed_url.netloc,
        'HostnameLength': len(parsed_url.netloc),
        'PathLength': len(path),
        'QueryLength': len(parsed_url.query),
        'DoubleSlashInPath': '//' in parsed_url.path,
        'NumSensitiveWords': sum(1 for word in ['login', 'secure', 'bank'] if word in url.lower()),
        'EmbeddedBrandName': any(brand in url.lower() for brand in ['microsoft', 'google']),
    }
    return features

#Fonction récupérant les informations nécessaires de l'URL dans une base de données avant d'y appliquer  notre modèle predictif.
def check_url(url):
    features = extract_features(url)
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)
    return "Phishing" if prediction[0] == 1 else "Legitimate"

#Création de l'interface utilisateur de l'application.
st.title('Phishing URL Detector')
url = st.text_input("Enter the URL to check:")

if st.button("Check URL"):
    result = check_url(url)
    st.write(f"The URL is {result}.")
