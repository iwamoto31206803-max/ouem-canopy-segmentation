"""Private D6-A5 and historical Val18 portable bundle for Notebook scratch."""
import argparse
import re
import shutil
import stat
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from d6a3_produce import canonical, identifier, need, pinned, plain
from d6a4 import D6Shards
from d6a_research_qa import digest, sha, strict_json
from r1_dataset import safe_path
import research_cuda as d5
from research_data import ResearchData

SCHEMA='ouem-d6-input-v1'
A5_SHA='66b2ab066eb362c583456bc270609dca533da2bafbf8f9645c07529409e4cff5'
A5_ID='9571151c0037d2ba7b05c9ad46a860579efa547d62a7248eccac7c96b47faf60'
LIMIT=128*1024**2
MEMBERS=1536
V5_PINS={'crop_plan.json':d5.PLAN,'research_review.json':d5.REVIEW,
         'candidates.json':d5.CANDIDATES,'dataset_freeze_v5.json':d5.FREEZE}


class HistoricalVal(ResearchData):
    """Pinned Val-only projection. Uses the frozen product reader and patch API."""
    def __init__(self, root):
        self.root=Path(root)
        self.plan_path,self.review_path,self.candidates_path=(
            safe_path(self.root,k) for k in ('crop_plan.json','research_review.json','candidates.json'))
        plan, review, _,freeze=(pinned(safe_path(self.root,k),pin) for k,pin in V5_PINS.items())
        self.expanded=True
        self.identity=freeze['identity']
        need(self.identity['crop_plan_sha256']==d5.PLAN and self.identity['research_review_sha256']==d5.REVIEW
             and self.identity['candidate_manifest_sha256']==d5.CANDIDATES, 'Historical freeze binding')
        self.targets={i:'val' for i in d5.VAL}
        allowed={self._key(r) for r in review['samples'] if r['image_id'] in self.targets
                 and r['decision']=='CLEAR_FOR_EXPLORATION'}
        self._records={self._key(r):r for r in plan['samples'] if self._key(r) in allowed}
        need(set(self._records)==allowed and len(allowed)==len(self.targets)
             and {k[0] for k in allowed}==set(self.targets),'Fixed historical Val18 scope')
        self._allowed=allowed
        self.entries=[]
        self.samples={}
        for rid in d5.VAL:
            row=next(r for r in self._records.values() if r['image_id']==rid)
            need(row['split']=='val' and row['validation_fold']==0,'Historical Val fold')
            self._read(row)
            item={k:row[k] for k in ('image_id','oam_id','crop_xywh')}
            self.entries.append(item)
            self.samples[rid]=dict(item,split='val')


def relative(name):
    need(isinstance(name,str) and chr(92) not in name and ':' not in name
         and all(p not in ('','.','..') and not p.endswith((' ','.')) for p in name.split('/')),
         'Unsafe bundle member')
    return name


def read_bundle(root):
    root=plain(Path(root))
    spec=strict_json(safe_path(root,'bundle.json').read_bytes())
    need(spec['schema']==SCHEMA and spec['a5_sha256']==A5_SHA and spec['initializer_sha256']==d5.INITIALIZER
         and spec['satlas']=='0.3.1' and re.fullmatch('[0-9a-f]{40}',spec['commit']),'Bundle identity')
    expected=spec['files']
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    need(actual==set(expected)|{'bundle.json'},'Bundle inventory')
    for name,info in expected.items():
        path=safe_path(root,relative(name))
        need(path.stat().st_size==info['size'] and sha(path)==info['sha256'],'Bundle product SHA')
    def resolve(ref,kind):
        key=identifier([ref['path'],ref['sha256'],kind])
        return root/relative(spec['relocations'][key])
    data=D6Shards(safe_path(root,'a5/manifest.json'),A5_SHA,resolve=resolve)
    val=HistoricalVal(root/'historical')
    needs_roles(data)
    return spec,data,val


def needs_roles(data):
    need(data.complete and data.identity['dataset_id']==A5_ID
         and len(data.samples)==200
         and sum(row['split']=='train' for row in data.samples.values())==175,'Fixed A5 scope')


def pack(a5, v5, archive, commit):
    need(re.fullmatch('[0-9a-f]{40}',commit),'Pinned code commit required')
    archive=plain(archive)
    a5=plain(a5)
    v5=plain(v5)
    files, relocations={},{}
    def add(name,path):
        name=relative(name)
        path=plain(path)
        need(name not in files and not archive.is_relative_to(path.parent),'Bundle destination overlap')
        files[name]=path

    def aggregate(path,pin,folder):
        doc=pinned(path,pin)
        add(folder+'/manifest.json',path)
        ref=doc['roster']
        add(folder+'/'+relative(ref['path']),safe_path(path.parent,ref['path']))
        if 'parent' in doc:
            r=doc['parent']
            target='parent/'+r['sha256']+'/manifest.json'
            relocations[identifier([r['path'],r['sha256'],'aggregate'])]=target
            aggregate(plain(Path(r['path'])),r['sha256'],target.rsplit('/',1)[0])
        for r in doc['shards']:
            root=plain(Path(r['path']))
            target='shards/'+r['sha256']
            relocations[identifier([r['path'],r['sha256'],'shard'])]=target
            for f in root.rglob('*'):
                if f.is_file():
                    add(target+'/'+f.relative_to(root).as_posix(),f)
    needs_roles(D6Shards(a5,A5_SHA))
    aggregate(a5,A5_SHA,'a5')
    plan=pinned(safe_path(v5/'dataset-v5','crop_plan.json'),d5.PLAN)
    review=pinned(safe_path(v5,'research_review.json'),d5.REVIEW)
    def key(row):
        return row['image_id'],*row['crop_xywh']
    allow={key(r) for r in review['samples'] if r['image_id'] in d5.VAL
           and r['decision']=='CLEAR_FOR_EXPLORATION'}
    for name in V5_PINS:
        add('historical/'+name,safe_path(v5/'dataset-v5' if name=='crop_plan.json' else v5,name))
    for row in plan['samples']:
        if key(row) in allow:
            for pin in row['products'].values():
                add('historical/'+pin['path'],safe_path(v5/'dataset-v5',pin['path']))
    need(len(files)<MEMBERS and sum(p.stat().st_size for p in files.values())<LIMIT,'Bundle capacity')
    spec=dict(schema=SCHEMA,commit=commit,a5_sha256=A5_SHA,initializer_sha256=d5.INITIALIZER,
              satlas='0.3.1',relocations=relocations,
              files={n:dict(size=p.stat().st_size,sha256=sha(p)) for n,p in files.items()})
    encoded=canonical(spec)
    with archive.open('xb') as dst, ZipFile(dst,'w') as zipped:
        for name,data in [('bundle.json',encoded),*((n,p) for n,p in sorted(files.items()))]:
            info=ZipInfo(name)
            info.compress_type=ZIP_DEFLATED
            with zipped.open(info,'w') as member:
                if isinstance(data,bytes):
                    member.write(data)
                else:
                    with data.open('rb') as stream:
                        shutil.copyfileobj(stream,member)
    return dict(archive_sha256=sha(archive),bundle_sha256=identifier(spec),archive_bytes=archive.stat().st_size,
                payload_bytes=sum(item['size'] for item in spec['files'].values()),files=len(files))


def unpack(archive,pin,output):
    need(digest(pin) and sha(plain(archive))==pin,'Archive SHA')
    output=plain(output)
    with ZipFile(archive) as zipped:
        entries=zipped.infolist()
        names=[relative(e.filename).casefold() for e in entries]
        need(len(entries)<=MEMBERS and len(names)==len(set(names))
             and sum(e.file_size for e in entries)<=LIMIT
             and all(not e.is_dir() and stat.S_IFMT(e.external_attr>>16) in (0,stat.S_IFREG)
                     for e in entries),'Unsafe archive inventory')
        output.mkdir()
        for entry in entries:
            path=output/entry.filename
            path.parent.mkdir(parents=True,exist_ok=True)
            with zipped.open(entry) as src,path.open('xb') as dst:
                shutil.copyfileobj(src,dst)
    spec,data,val=read_bundle(output)
    return dict(commit=spec['commit'],bundle_sha256=sha(output/'bundle.json'),
                train=sum(row['split']=='train' for row in data.samples.values()),
                val=sum(row['split']=='val' for row in data.samples.values()),historical_val=len(val.entries),identity=data.identity)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='action',required=True)
    p=sub.add_parser('pack')
    for key in ('a5','v5','archive'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--commit',required=True)
    p=sub.add_parser('unpack')
    for key in ('archive','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--sha256',required=True)
    args=ap.parse_args()
    result=(pack(args.a5,args.v5,args.archive,args.commit) if args.action=='pack'
            else unpack(args.archive,args.sha256,args.output))
    print(canonical(result).decode(),end='')


if __name__=='__main__':
    main()
