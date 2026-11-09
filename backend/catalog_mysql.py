"""Verified MySQL 8.4 catalog contract. No free-form SQL or driver options."""
import re
import socket
import ssl
from contextlib import contextmanager

ALLOWED_TYPES={'char','varchar','tinytext','text','mediumtext','longtext','tinyint','smallint','mediumint','int','bigint','decimal','float','double','date','datetime','timestamp','time','year','json','enum','set'}
READ_CONTRACT='MySQL 8.4: explicit SELECT/USAGE grants only, verified read-only transaction, ordinary InnoDB base tables and scalar/JSON fields only.'

def quote(name):
    if not isinstance(name,str) or not 1<=len(name)<=64 or any(ord(c)<32 for c in name):
        raise ValueError('Invalid catalog identifier')
    return '`'+name.replace('`','``')+'`'

def verify_grants(grants):
    has_select=False
    for (grant,) in grants:
        # Unknown grants, roles, grant option and proxy delegation fail closed.
        match=re.fullmatch(r"GRANT (USAGE|SELECT) ON (?:\*|`(?:[^`]|``)+`)\.(?:\*|`(?:[^`]|``)+`) TO `(?:[^`]|``)+`@`(?:[^`]|``)+`",grant,re.IGNORECASE)
        if not match:raise ValueError('MySQL account has unapproved/write privileges or roles; use a direct SELECT-only account without grant option')
        has_select|=match.group(1).upper()=='SELECT'
    if not has_select:raise ValueError('MySQL account has no verifiable explicit SELECT grant')

@contextmanager
def session(services,endpoint,credential,ca_file):
    from catalog_connections import db_endpoint,driver
    p,address,port=db_endpoint(endpoint,'mysql')
    if not credential.get('username'):raise ValueError('Use an existing read-only MySQL account')
    if p.path.strip('/').lower() in ('mysql','sys','information_schema','performance_schema'):raise ValueError('Choose a user catalog database')
    trust=str(services.source(ca_file,{'.pem','.crt'},2*1024**2)) if ca_file else None
    tls=ssl.create_default_context(cafile=trust)
    mysql=driver('pymysql')
    conn=mysql.connect(host=p.hostname,port=port,user=credential['username'],password=credential.get('password',''),database=p.path.strip('/'),charset='utf8mb4',connect_timeout=8,read_timeout=8,write_timeout=5,defer_connect=True,local_infile=False,ssl=tls)
    sock=socket.socket(socket.AF_INET6 if ':' in address else socket.AF_INET,socket.SOCK_STREAM)
    try:
        sock.settimeout(8);sock.connect((address,port));conn.connect(sock)
        with conn.cursor() as query:
            query.execute('SELECT VERSION(), @@GLOBAL.mandatory_roles')
            version,mandatory=query.fetchone()
            if not re.match(r'^8\.4\.',version) or 'mariadb' in version.lower():raise ValueError('This read-only adapter is verified for MySQL 8.4; this server needs its own reviewed contract')
            if mandatory:raise ValueError('Mandatory MySQL roles cannot be verified by this adapter')
            query.execute('SHOW GRANTS');verify_grants(query.fetchall())
            query.execute('SET SESSION TRANSACTION READ ONLY')
            query.execute('SET SESSION MAX_EXECUTION_TIME=5000')
            query.execute('SET SESSION lock_wait_timeout=2')
            query.execute('START TRANSACTION READ ONLY')
            query.execute('SELECT @@SESSION.transaction_read_only')
            if query.fetchone()!=(1,):raise ValueError('MySQL did not enforce a read-only session')
        yield conn,p.path.strip('/')
    finally:
        try:
            if conn.open:
                try:conn.rollback()
                finally:conn.close()
        finally:sock.close()

def tables(conn,database,table=None):
    with conn.cursor() as query:
        suffix=' AND TABLE_NAME=%s' if table is not None else ''
        args=(database,table) if table is not None else (database,)
        query.execute("SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s AND TABLE_TYPE='BASE TABLE' AND ENGINE='InnoDB'"+suffix+' ORDER BY TABLE_NAME LIMIT 501',args)
        names=[row[0] for row in query.fetchall()]
        if len(names)>500:raise ValueError('Too many tables; use a narrower catalog account')
        inventory=[]
        for name in names:
            query.execute("SELECT COLUMN_NAME,DATA_TYPE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND (GENERATION_EXPRESSION IS NULL OR GENERATION_EXPRESSION='') ORDER BY ORDINAL_POSITION LIMIT 501",(database,name))
            columns=query.fetchall()
            if len(columns)>500:raise ValueError('Table exceeds the bounded schema size; use a narrower catalog')
            inventory.append({'schema':database,'name':name,'columns':[{'name':col,'type':kind} for col,kind in columns if kind in ALLOWED_TYPES]})
        return inventory

def preview(conn,database,schema,mapping,limit):
    if tables(conn,database,schema['name'])!=[schema]:raise ValueError('MySQL table or schema changed; inspect a new version')
    if set(mapping.values())-{c['name'] for c in schema['columns']}:raise ValueError('Choose discovered scalar/JSON fields only; arbitrary SQL is blocked')
    columns=','.join('LEFT(CAST('+quote(col)+' AS CHAR CHARACTER SET utf8mb4),8193)' for col in mapping.values())
    with conn.cursor() as query:
        query.execute('SELECT '+columns+' FROM '+quote(database)+'.'+quote(schema['name'])+' LIMIT %s',(limit,))
        rows=query.fetchall()
    if any(isinstance(v,str) and len(v)>8192 for row in rows for v in row):raise ValueError('A mapped MySQL value exceeds 8192 characters; no truncated preview was saved')
    return [dict(zip(mapping,row)) for row in rows]
