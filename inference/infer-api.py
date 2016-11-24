"""Script to call the mjolnir-running classification api. It converts images to jpeg first.

Usage:
    python /media/esteva/ExtraDrive1/ThrunResearch/tensorflow_master/lib/inference/infer-api.py \
            /home/esteva/Pictures/Melanoma.jpg

"""
import requests
import json
import base64 
import argparse
import sys

from PIL import Image
from io import BytesIO
import cStringIO


parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str,
                            help='path to the image file')
args = parser.parse_args()

BASEURL = 'https://mjolnir-collect.stanford.edu:8000'
AUTH = None
APIKEY = 'zWnXAz8n8XpCvp79AExQDYrkTR9XfXDACXpXaVrv' 

def DumpReq(req):
	print "Code = ", req.status_code
	print "Headers = ", req.headers
	print "Body = ", req.text


print "Infering from API: %s" % args.image_path

# Open the image file, convert to JPEG.
f = open(args.image_path, 'rb').read()
im = Image.open(BytesIO(f)).convert("RGB")
buffer = cStringIO.StringIO()
im.save(buffer, format="JPEG")
image_str = base64.b64encode(buffer.getvalue())

# Call the API
user_d = {
        'api_key': APIKEY,
        'image': image_str,
        'content_type': 'string' }
r = requests.post('%s/classify/base64'  % (BASEURL), data=json.dumps(user_d), auth=AUTH, verify=True)
DumpReq(r)

