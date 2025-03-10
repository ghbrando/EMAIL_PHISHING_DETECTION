# EMAIL_PHISHING_DETECTION
Phishing Email Detection using IMAP and Python This project uses Python libraries such as "llamaIndex", "AzureOpenAI" LLMS and "neo4j"to detect phishing emails by connecting to an email account via IMAP. 

Features:

1: Uses an LLM to determine whether an email is considered phishing
2: Uses an LLM to determine whether an email is considered spam
3. Can be ran in the background to always detect what emails are spam in Linux by using
  :"nohup python script.py &" in the case, the script is phishing_detection.py
4. If an email is considered spam, the script will send an email to the account specified in the .env file, this is built using the SMTP python package.
