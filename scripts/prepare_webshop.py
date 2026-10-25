"""Download official 1k WebShop assets and build the original Lucene index."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'agent_system/environments/env_package/webshop/webshop'
FILES={
    'items_shuffle_1000.json':'1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib',
    'items_ins_v2_1000.json':'1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu',
    'items_human_ins.json':'14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root',required=True,type=Path)
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args();assets=args.asset_root.resolve()
    (assets/'data').mkdir(parents=True,exist_ok=True)
    for name,file_id in FILES.items():
        path=assets/'data'/name
        if not path.is_file():
            if not args.download:raise FileNotFoundError(path)
            temporary=path.with_suffix('.json.partial')
            subprocess.run([sys.executable,'-m','gdown',f'https://drive.google.com/uc?id={file_id}',
                            '-O',str(temporary)],check=True)
            json.loads(temporary.read_text(encoding='utf8'))
            temporary.replace(path)
        json.loads(path.read_text(encoding='utf8'))
    os.environ['WEBSHOP_ASSET_ROOT']=str(assets)
    sys.path.insert(0,str(SOURCE))
    from web_agent_site.engine.engine import load_products
    products,*_=load_products(filepath=str(assets/'data/items_shuffle_1000.json'),
        attrpath=str(assets/'data/items_ins_v2_1000.json'),num_products=1000,human_goals=False)
    resource=assets/'search_engine/resources_1k';index=assets/'search_engine/indexes_1k'
    if index.exists() and any(index.iterdir()):
        from pyserini.search.lucene import LuceneSearcher
        assert LuceneSearcher(str(index)).num_docs==len(products)
        print('Existing index retained',index);return
    resource.mkdir(parents=True,exist_ok=True)
    with (resource/'documents.jsonl').open('x',encoding='utf8') as out:
        for product in products:
            options=', and '.join(f'{k}: {", ".join(v)}' for k,v in product.get('options',{}).items())
            contents=' '.join([product['Title'],product['Description'],product['BulletPoints'][0],options]).lower()
            out.write(json.dumps(dict(id=product['asin'],contents=contents,product=product))+'\n')
    assert 0<len(products)<=1000 and len({p['asin'] for p in products})==len(products)
    subprocess.run([sys.executable,'-m','pyserini.index.lucene','--collection','JsonCollection',
        '--input',str(resource),'--index',str(index),'--generator','DefaultLuceneDocumentGenerator',
        '--threads','1','--storePositions','--storeDocvectors','--storeRaw'],check=True)
    from pyserini.search.lucene import LuceneSearcher
    assert LuceneSearcher(str(index)).num_docs==len(products)
    print('INDEX_READY',len(products),index)


if __name__=='__main__':main()
