import os
import time
import smtplib
from dotenv import load_dotenv
import imaplib
import email
from email.header import decode_header
from email.mime.text import MIMEText

_ = load_dotenv()

from llama_index.core import PropertyGraphIndex
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import Settings

AZURE_OPENAI_NORTH_API_KEY = os.getenv("AZURE_OPENAI_NORTH_API_KEY")
AZURE_OPENAI_NORTH_ENDPOINT = os.getenv("AZURE_OPENAI_NORTH_ENDPOINT")
AZURE_OPENAI_EAST_ENDPOINT = os.getenv("AZURE_OPENAI_EAST_ENDPOINT")
AZURE_OPENAI_EAST_API_KEY = os.getenv("AZURE_OPENAI_EAST_API_KEY")
API_VERSION = "2024-06-01"

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

llm_4o = AzureOpenAI(
    azure_deployment="gpt-4o-mini",
    api_key=AZURE_OPENAI_EAST_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_EAST_ENDPOINT
)

# Instantiate the Embedding model
embedding_model = AzureOpenAIEmbedding(
    azure_deployment="text-embedding-ada-002",
    api_key=AZURE_OPENAI_NORTH_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_NORTH_ENDPOINT
)

# Set the LLM and Embedding models in the Settings
Settings.embed_model = embedding_model
Settings.llm = llm_4o

graph_store = Neo4jPropertyGraphStore(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)

index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    embed_kg_nodes=True,
    embedding_model=Settings.embed_model,
    show_progress=True
)

IMAP_SERVER = os.getenv("IMAP_SERVER")
EMAIL_ACCOUNT = os.getenv("EMAIL_ACCOUNT")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

mail = imaplib.IMAP4_SSL(IMAP_SERVER)
mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
mail.select("inbox")  # Select the inbox

last_checked_email = None  # Track last checked email

def get_last_email():
    global last_checked_email
    status, messages = mail.search(None, "ALL")
    email_ids = messages[0].split()
    
    if not email_ids:
        return None

    latest_email_id = email_ids[-1]  # Get latest email
    if latest_email_id == last_checked_email:
        return None  # No new email

    last_checked_email = latest_email_id  # Update last checked email
    
    status, msg_data = mail.fetch(latest_email_id, "(RFC822)")
    for response_part in msg_data:
        if isinstance(response_part, tuple):
            msg = email.message_from_bytes(response_part[1])
            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8")
            sender = msg.get("From")
            return sender, subject
    return None

def send_alert(message):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    alert_email = os.getenv("EMAIL_ACCOUNT")  # Your email to receive alerts
    alert_password = os.getenv("EMAIL_PASSWORD")  # App password if using Gmail

    msg = MIMEText(message)
    msg["Subject"] = "🚨 Phishing Alert 🚨"
    msg["From"] = EMAIL_ACCOUNT
    msg["To"] = alert_email

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(EMAIL_ACCOUNT, alert_password)
        server.sendmail(EMAIL_ACCOUNT, alert_email, msg.as_string())

if __name__ == "__main__":
    chat_engine = index.as_chat_engine()
    last_email = None  # Store the last processed email

    while True:
        email_data = get_last_email()
        if email_data:
            sender, subject = email_data
            current_email = f"{sender}|{subject}"  # Create a unique identifier

            if current_email == last_email:
                continue  # Skip processing if the email is the same
            
            last_email = current_email  # Update last_email

            response = chat_engine.chat(f"Is this a phishing email: {sender}, {subject}? Just answer 'yes' or 'no'")
            print(f"Checked email from {sender}: {subject} -> Response: {response.response}")

            response_text = response.response  # Use .response instead of .text

            if "yes" in response_text.lower() or "yes." in response_text.lower():
                alert_message = f"⚠️ Potential phishing email detected! ⚠️\n\nSender: {sender}\nSubject: {subject}"
                send_alert(alert_message)
                print("🚨 ALERT SENT! 🚨")