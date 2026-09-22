"""Download original resources from their publishers; never redistributes the lexicon."""
import argparse
import urllib.request
import zipfile
import io
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA_URL='https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz'
NRC_URL='https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',action='store_true')
    ap.add_argument('--lexicon',action='store_true',help='Use only after reading the publisher terms; educational/research use, no redistribution')
    args=ap.parse_args()
    if not args.data and not args.lexicon: ap.error('Select --data and/or --lexicon')
    if args.data:
        target=ROOT/'Gift_Cards.jsonl.gz'
        if not target.exists():
            with urllib.request.urlopen(DATA_URL,timeout=120) as response:
                content=response.read()
            if not content.startswith(b'\x1f\x8b'): raise ValueError('Download is not a gzip file')
            target.write_bytes(content)
        print(target)
    if args.lexicon:
        target=ROOT/'resources';target.mkdir(exist_ok=True)
        # The publisher returns HTTP 406 to urllib's default user agent.
        # Its public ZIP endpoint accepts the curl user agent (verified in a clean checkout).
        request=urllib.request.Request(NRC_URL,headers={'User-Agent':'curl/8.7.1'})
        with urllib.request.urlopen(request,timeout=120) as response:
            archive=zipfile.ZipFile(io.BytesIO(response.read()))
        for source,name in [('NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt','NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'),('NRC-Emotion-Lexicon/README.txt','NRC-README.txt')]:
            (target/name).write_bytes(archive.read(source))
        print('NRC lexicon downloaded locally. Read resources/NRC-README.txt; do not commit resources/.')

if __name__=='__main__': main()
