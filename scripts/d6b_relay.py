"""Exclusive copy and receipt checks for D6 segment evidence."""
import argparse
import shutil
from pathlib import Path

from d6a3_produce import canonical, need, pinned, plain, write_new
from d6a_research_qa import sha
from d6b_bundle import relative
from r1_dataset import safe_path

SCHEMA='ouem-d6-evidence-v1'


def inventory(folder):
    return {p.relative_to(folder).as_posix():dict(size=p.stat().st_size,sha256=sha(plain(p)))
            for p in sorted(folder.rglob('*')) if p.is_file() and p != folder/'receipt.json'}


def check_files(folder, receipt):
    need(inventory(folder)==receipt['files'],'Evidence inventory or SHA mismatch')


def seal(folder, details):
    receipt=dict(schema=SCHEMA,details=details,files=inventory(folder))
    path=folder/'receipt.json'
    write_new(path,canonical(receipt),2*1024**2)
    return sha(path)


def verify(folder,pin):
    folder=plain(folder)
    receipt=pinned(safe_path(folder,'receipt.json'),pin)
    need(receipt['schema']==SCHEMA and receipt['files'],'Invalid evidence receipt')
    check_files(folder,receipt)
    return receipt


def copy(folder,pin,dest):
    """Local/mount copy. The caller verifies a separately fetched remote copy."""
    folder,dest=map(plain,(folder,dest))
    need(not dest.is_relative_to(folder) and not folder.is_relative_to(dest),'Evidence paths overlap')
    receipt=verify(folder,pin)
    dest.mkdir()
    for name in sorted(receipt['files']):
        path=dest/relative(name)
        path.parent.mkdir(parents=True,exist_ok=True)
        with safe_path(folder,name).open('rb') as stream,path.open('xb') as out:
            shutil.copyfileobj(stream,out,1024**2)
    check_files(dest,receipt)
    path=dest/'receipt.json'
    write_new(path,safe_path(folder,'receipt.json').read_bytes(),2*1024**2)
    need(sha(path)==pin,'Evidence receipt changed')
    return receipt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('evidence',type=Path)
    ap.add_argument('--receipt-sha256',required=True)
    ap.add_argument('--archive',type=Path)
    args=ap.parse_args()
    receipt=(copy(args.evidence,args.receipt_sha256,args.archive) if args.archive
             else verify(args.evidence,args.receipt_sha256))
    print(canonical(dict(status='SHA_MATCH',receipt_sha256=args.receipt_sha256,
                         bytes=sum(f['size'] for f in receipt['files'].values()),
                         details=receipt['details'])).decode(),end='')


if __name__=='__main__':
    main()
