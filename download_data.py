"""Download and verify the exact dataset used by this benchmark."""
import hashlib
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent / 'data'
FILES = {'asr_bangal_1000.csv':'4a5b61cfff934da67b8f8db56ebfee61d56d92b04ee317f5144495b5bcfde1fa',
         'asr_bengali_1000_wav.zip':'71d365006c278b4853d89e6c83c9aece17109094488dd3c3864da3c1dd799dfd'}

def main():
    ROOT.mkdir(exist_ok=True)
    for name, digest in FILES.items():
        target = ROOT / name
        if not target.exists():
            temporary = target.with_suffix(target.suffix + '.part')
            urllib.request.urlretrieve('https://raw.githubusercontent.com/menon92/BangalASR/main/data/asr-bengali-1000/'+name, temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest()!=digest:
                raise ValueError('Upstream dataset changed: '+name)
            temporary.replace(target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise ValueError('Dataset checksum mismatch: '+name)
    with zipfile.ZipFile(ROOT/'asr_bengali_1000_wav.zip') as archive:
        for member in archive.infolist():
            if not (ROOT/member.filename).resolve().is_relative_to(ROOT.resolve()):
                raise ValueError('Unsafe archive path')
        archive.extractall(ROOT)
    print('Verified and extracted dataset')

if __name__=='__main__': main()
