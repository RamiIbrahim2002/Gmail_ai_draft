import os
import time
import requests
import json
import logging
import backoff
import base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# Scopes for Gmail API access
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Ollama API endpoint
OLLAMA_API = 'http://localhost:11434/api/chat'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def authenticate_gmail():
    """Authenticate with Gmail API and return a service object."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json',
                SCOPES
            )
            creds = flow.run_local_server(port=0)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

def decode_email_body(msg):
    """Extract the body from a Gmail message."""
    try:
        parts = msg['payload'].get('parts', [])
        for part in parts:
            if part['mimeType'] in ['text/plain', 'text/html']:
                body = part.get('body', {}).get('data', '')
                return base64.urlsafe_b64decode(body).decode('utf-8')
        
        # If no parts, check the main body
        body = msg['payload'].get('body', {}).get('data', '')
        return base64.urlsafe_b64decode(body).decode('utf-8') if body else "No content"
    except Exception as e:
        logging.error(f"Error decoding email body: {e}")
        return "Error extracting body"

@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=5)
def classify_email_with_ollama(email_content):
    """Classify email content using Ollama."""
    try:
        prompt = f"""
        Classify this email into one of these categories:
        - Meetings
        - Workshops
        - Business

        Reply with only the category , if the email isn't in any category , simply reply by other .
        Email content:
        {email_content}
        """
        payload = {
            "model": "mistral",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        response = requests.post(OLLAMA_API, json=payload)
        response.raise_for_status()

        classification = response.json().get('message', {}).get('content', "Unclassified")
        logging.info(f"Classification result: {classification}")
        return classification
    except Exception as e:
        logging.error(f"Error during email classification: {e}")
        return "Unclassified"

def fetch_primary_unread_emails(service):
    """Fetch unread emails from the primary category."""
    try:
        logging.info("Fetching unread primary emails...")
        results = service.users().messages().list(
            userId='me',
            labelIds=['INBOX'],
            q='is:unread category:primary'
        ).execute()
        return results.get('messages', [])
    except Exception as e:
        logging.error(f"Error fetching emails: {e}")
        return []

def process_emails(service):
    """Fetch, decode, classify, and log primary emails."""
    logging.info("Starting email processing...")
    while True:
        try:
            messages = fetch_primary_unread_emails(service)
            if not messages:
                logging.info("No unread primary emails found. Retrying in 30 seconds...")
                time.sleep(30)
                continue
            
            for msg in messages:
                full_msg = service.users().messages().get(
                    userId='me',
                    id=msg['id']
                ).execute()
                subject = next((h['value'] for h in full_msg['payload']['headers'] if h['name'] == 'Subject'), 'No Subject')
                sender = next((h['value'] for h in full_msg['payload']['headers'] if h['name'] == 'From'), 'Unknown Sender')
                email_body = decode_email_body(full_msg)
                classification = classify_email_with_ollama(email_body)

                logging.info("\n--- NEW EMAIL ---")
                logging.info(f"From: {sender}")
                logging.info(f"Subject: {subject}")
                logging.info(f"Classification: {classification}")
                logging.info(f"Body: {email_body}")
                logging.info("-----------------\n")
            
            logging.info("Waiting 30 seconds before checking for new emails...")
            time.sleep(30)
        except Exception as e:
            logging.error(f"An error occurred during email processing: {e}")
            logging.info("Retrying in 60 seconds...")
            time.sleep(60)

def main():
    """Main function to authenticate and process emails."""
    service = authenticate_gmail()
    process_emails(service)

if __name__ == '__main__':
    main()
