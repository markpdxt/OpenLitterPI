# OpenLitterPI - Automated cat litterbox
# Copyright (C) 2025 Mark Nelson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Notification module for OpenLitterPI.
Handles email notifications with photo attachments.
"""

import os
import ssl
import cv2
import imghdr
import smtplib
from email.message import EmailMessage

# Email configuration from environment variables
EMAIL_SUBJECT = "OpenLitterPI"
EMAIL_FROM = os.environ.get("OPENLITTERPI_EMAIL_FROM", "")
EMAIL_PASSWORD = os.environ.get("OPENLITTERPI_EMAIL_PASSWORD", "")
EMAIL_TO = os.environ.get("OPENLITTERPI_EMAIL_TO", "")


def send_message(message_text, photo=None, include_image=True):
    """
    Send an email notification with optional photo attachment.
    
    Args:
        message_text: Subject/body text for the email
        photo: OpenCV image array to attach
        include_image: Whether to attach the photo
    """
    if not all([EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO]):
        print(f"Warning: Email not configured. Skipping notification: {message_text}")
        return

    msg = EmailMessage()
    msg['Subject'] = f"{EMAIL_SUBJECT} - {message_text}"
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO
    msg.set_content(message_text)

    if include_image and photo is not None:
        photo_name = f'photos/{message_text}.jpg'
        cv2.imwrite(photo_name, photo)
        with open(photo_name, 'rb') as f:
            image_data = f.read()
            image_type = imghdr.what(f.name)
            image_name = f.name
        msg.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(EMAIL_FROM, EMAIL_PASSWORD)
        smtp.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
