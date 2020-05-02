import yagmail
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os.path

def yagmail_results(path):
    yag = yagmail.SMTP('pgmikhael.development', oauth2_file = '/home/ec2-user/MLExperiments/secrets/oauth_cred.json')
    contents = ["Experiment Results"]
    yag.send('pgmikhael@gmail.com', 'NAB Exp - Results', contents, attachments= path)

def email_results(path):

    E = 'pgmikhael.development@gmail.com'
    P = 'Xl#0JpYOVc'
    send_to_email = 'pgmikhael@gmail.com'
    subject = 'NAB Exp - Results'
    message = '\n'
    file_location = path

    msg = MIMEMultipart()
    msg['From'] = E
    msg['To'] = send_to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(message, 'plain'))

    # Setup the attachment
    filename = os.path.basename(file_location)
    attachment = open(file_location, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    # Attach the attachment to the MIMEMultipart object
    msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(E, P)
    text = msg.as_string()
    server.sendmail(E, send_to_email, text)
    server.quit()
