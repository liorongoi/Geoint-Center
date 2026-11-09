"""Pinned macOS Db2 LUW 12.1.3 read-only catalog connector."""
import hashlib
import ipaddress
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

WHEEL_SHA256 = 'b801774c68c303e5a53fbd982b79bbfa5905b789eb3476b8ae1db22ce6726c20'
MODULE = 'ibm_db.cpython-312-darwin.so'
IDENTIFIER = re.compile(r'[A-Z][A-Z0-9_]{0,127}')
ENDPOINT = re.compile(r'db2://127\.0\.0\.1:([1-9][0-9]{0,4})/([A-Z][A-Z0-9_]{0,127})')
READ_CONTRACT = ('Pinned Db2 LUW 12.1.3.0 and verified TLS over an approved loopback relay. '
                 'Least-privilege discovery and bounded base-table reads only.')
METADATA = {
    ('SYSIBMADM', 'ENV_INST_INFO'), ('SYSIBMADM', 'OBJECTOWNERS'),
    ('SYSIBMADM', 'PRIVILEGES'), ('SYSCAT', 'TABLES'), ('SYSCAT', 'COLUMNS'),
    ('SYSCAT', 'COLAUTH'), ('SYSCAT', 'ROUTINES'),
}
AUTH_ROUTINES = ('AUTH_LIST_AUTHORITIES_FOR_AUTHID', 'AUTH_LIST_GROUPS_FOR_AUTHID',
                 'AUTH_LIST_ROLES_FOR_AUTHID')
SCALARS = {'SMALLINT', 'INTEGER', 'BIGINT', 'DECIMAL', 'CHARACTER', 'VARCHAR', 'DATE', 'TIMESTAMP'}


def verified_driver(wheel, directory):
    with Path(wheel).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != WHEEL_SHA256:
            raise ValueError('Db2 offline driver does not match the reviewed wheel checksum')
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            for name in archive.namelist():
                if name == MODULE or name.startswith('clidriver/'):
                    archive.extract(name, directory)


def private_address(value):
    address = ipaddress.IPv4Address(value)
    if str(address) != '127.0.0.1':
        raise ValueError('The staged Mac Db2 adapter requires an approved loopback endpoint')
    return str(address)


def sandbox_profile(address, port):
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('Invalid Db2 TLS port')
    private_address(address)
    return ('(version 1) (allow default) (deny network*) '
            f'(allow network-outbound (remote tcp "localhost:{port}"))')


def catalog_request(services, endpoint, credential, ca_file, expected_ca_sha256=None, **selection):
    match = ENDPOINT.fullmatch(endpoint)
    if match is None or int(match[1]) > 65535:
        raise ValueError('Use db2://127.0.0.1:port/UPPERCASE_DATABASE with an approved TLS relay')
    configured = os.getenv('GEO_DB2_WHEEL')
    if not configured:
        raise ValueError('Configure the pinned offline Db2 wheel with GEO_DB2_WHEEL')
    wheel = Path(configured)
    if not wheel.is_absolute() or wheel.is_symlink() or not wheel.is_file():
        raise ValueError('Configure a local pinned Db2 wheel file')
    if not ca_file:
        raise ValueError('Db2 requires an approved TLS certificate file')
    trust = services.source(ca_file, {'.pem', '.crt'}, 2*1024**2)
    with trust.open('rb') as stream:
        certificate = stream.read(2*1024**2 + 1)
    if not 0 < len(certificate) <= 2*1024**2:
        raise ValueError('Db2 TLS certificate file is empty or exceeds the size limit')
    ca_sha256 = hashlib.sha256(certificate).hexdigest()
    if expected_ca_sha256 is not None and ca_sha256 != expected_ca_sha256:
        raise ValueError('Saved TLS trust material changed; inspect a new connection version')
    result = request(wheel, certificate, '127.0.0.1', int(match[1]), match[2],
                     credential.get('username'), credential.get('password'), **selection)
    if selection:
        return result
    return {'engine_version': result['version'], 'tables': result['tables'],
            'ca_sha256': ca_sha256, 'read_contract': READ_CONTRACT}


def request(wheel, ca, address, port, database, username, password, **selection):
    if sys.platform != 'darwin' or platform.machine() != 'arm64' or sys.version_info[:2] != (3, 12):
        raise ValueError('Db2 requires the Apple Silicon Mac Python 3.12 native-client sandbox')
    if (not isinstance(database,str) or not IDENTIFIER.fullmatch(database)
            or not isinstance(username,str) or not IDENTIFIER.fullmatch(username.upper()) or not username.isascii()):
        raise ValueError('Use an uppercase Db2 database and an ordinary ASCII reader name')
    if (not isinstance(password, str) or not 1 <= len(password) <= 1024
            or password != password.strip() or any(character in password for character in '\x00\r\n{};')):
        raise ValueError('Use a Db2 reader password without CLI delimiters or surrounding whitespace')
    profile = sandbox_profile(address, port)
    with tempfile.TemporaryDirectory(prefix='geoint-db2-client-') as temporary:
        root = Path(temporary).resolve()
        verified_driver(wheel, root)
        trust = root / 'supplier.pem'
        trust.write_bytes(ca)
        payload = {'address':address, 'port':port, 'database':database, 'username':username,
                   'password':password, **selection}
        env = {'PATH':'/usr/bin:/bin', 'TMPDIR':str(root), 'DB2CLIINIPATH':str(root),
               'DB2DSDRIVER_CFG_PATH':str(root/'db2dsdriver.cfg'), 'LANG':'en_US.UTF-8'}
        try:
            result = subprocess.run(['/usr/bin/sandbox-exec', '-p', profile, sys.executable, '-I', '-B',
                                     str(Path(__file__).resolve()), str(root)], input=json.dumps(payload),
                                    capture_output=True, text=True, timeout=30, cwd=root, env=env)
        except subprocess.TimeoutExpired as exc:
            raise ValueError('The bounded Db2 read exceeded 30 seconds') from exc
        if result.returncode != 0:
            raise ValueError('The verified Db2 client or its network sandbox could not start')
        response = json.loads(result.stdout)
        if 'error' in response:
            raise ValueError(response['error'])
        return response['data']


def connect(db, root, config):
    dsn = (f"DATABASE={config['database']};HOSTNAME={config['address']};PORT={config['port']};"
           f"PROTOCOL=TCPIP;SECURITY=SSL;SSLServerCertificate={root/'supplier.pem'};"
           "SSLClientHostnameValidation=Basic;EnableAcr=false;ConnectTimeout=6;"
           f"UID={config['username']};PWD={config['password']};")
    # The CLI connection-string path reads UID and PWD from the string.
    return db.connect(dsn, '', '')


def reader(db, connection):
    def rows(sql, parameters=(), maximum=101):
        statement = db.prepare(connection, sql, {db.SQL_ATTR_QUERY_TIMEOUT: 6})
        try:
            db.execute(statement, parameters)
            values = []
            while row := db.fetch_tuple(statement):
                if len(values) >= maximum:
                    raise ValueError('Db2 response exceeded the verified read limit')
                values.append(tuple(value.rstrip() if isinstance(value, str) else value for value in row))
            return values
        finally:
            db.free_stmt(statement)
    return rows


def verify_contract(read, database, username):
    if read('VALUES (CURRENT SERVER, SESSION_USER)', maximum=1) != [(database, username)]:
        raise ValueError('Db2 returned a different database or session identity')
    identity = read('SELECT SERVICE_LEVEL FROM SYSIBMADM.ENV_INST_INFO FETCH FIRST 2 ROWS ONLY', maximum=2)
    version = 'Db2 v12.1.3.0'
    if len(identity) != 1 or not isinstance(identity[0][0], str) or identity[0][0].casefold() != version.casefold():
        raise ValueError('The staged contract requires Db2 LUW 12.1.3.0')
    authorities = read("SELECT AUTHORITY,D_USER,D_GROUP,D_PUBLIC,ROLE_USER,ROLE_GROUP,ROLE_PUBLIC,D_ROLE "
                       "FROM TABLE(SYSPROC.AUTH_LIST_AUTHORITIES_FOR_AUTHID(SESSION_USER, 'U')) AS A")
    if (any(len(row) != 8 or set(row[1:]) - {'Y','N','*'} for row in authorities)
            or {row[0] for row in authorities if 'Y' in row[1:]} != {'CONNECT'}):
        raise ValueError('Db2 reader requires CONNECT as its only effective database or instance authority')
    if read("SELECT ROLENAME FROM TABLE(SYSPROC.AUTH_LIST_ROLES_FOR_AUTHID(SESSION_USER, 'U')) AS R FETCH FIRST 1 ROW ONLY"):
        raise ValueError('Db2 reader roles are outside the staged contract')
    groups = {row[0] for row in read('SELECT "GROUP" FROM TABLE(SYSPROC.AUTH_LIST_GROUPS_FOR_AUTHID(SESSION_USER)) AS G')}
    grantees = sorted(groups | {'PUBLIC', username})
    params = ','.join('?' for _ in grantees)
    if read(f'SELECT 1 FROM SYSIBMADM.OBJECTOWNERS WHERE OWNER IN ({params}) FETCH FIRST 1 ROW ONLY',tuple(grantees)):
        raise ValueError('Db2 reader, groups and PUBLIC must not own database objects')
    if read(f'SELECT 1 FROM SYSCAT.COLAUTH WHERE GRANTEE IN ({params}) FETCH FIRST 1 ROW ONLY',tuple(grantees)):
        raise ValueError('Db2 column-level grants are outside the staged contract')
    known_routines = {row[0] for row in read(
        "SELECT SPECIFICNAME FROM SYSCAT.ROUTINES WHERE ROUTINESCHEMA='SYSPROC' "
        "AND ROUTINENAME IN (?,?,?)", AUTH_ROUTINES)}
    grants = read(f'SELECT AUTHID, PRIVILEGE, GRANTABLE, OBJECTSCHEMA, OBJECTNAME, OBJECTTYPE '
                  f'FROM SYSIBMADM.PRIVILEGES WHERE AUTHID IN ({params}) FETCH FIRST 1001 ROWS ONLY',
                  tuple(grantees), maximum=1000)
    tables = set()
    for authid, privilege, grantable, schema, name, kind in grants:
        if grantable == 'N':
            if privilege == 'CONNECT' and kind == 'DATABASE':
                continue
            if (privilege == 'USAGE' and kind == 'WORKLOAD'
                    and not schema and name == 'SYSDEFAULTUSERWORKLOAD'):
                continue
            if privilege == 'SELECT' and (schema,name) in METADATA:
                continue
            if privilege == 'EXECUTE' and kind == 'FUNCTION' and schema == 'SYSPROC' and name in known_routines:
                continue
            if (privilege == 'EXECUTE' and kind == 'DB2 PACKAGE'
                    and (schema, name) == ('NULLID', 'SYSSH200')):
                continue
            if (authid == username and privilege == 'SELECT' and kind == 'TABLE'
                    and isinstance(schema,str) and isinstance(name,str)
                    and IDENTIFIER.fullmatch(schema) and IDENTIFIER.fullmatch(name)
                    and not schema.startswith('SYS') and schema != 'NULLID'):
                tables.add((schema,name))
                continue
        raise ValueError('Db2 reader has a grant outside the verified read contract')
    return version, tables


def inventory(read, tables):
    result = []
    if len(tables) > 100:
        raise ValueError('Choose a narrower Db2 catalog account')
    for schema, table in sorted(tables):
        columns = read("SELECT C.COLNAME, C.TYPENAME FROM SYSCAT.TABLES T JOIN SYSCAT.COLUMNS C "
                       "ON T.TABSCHEMA=C.TABSCHEMA AND T.TABNAME=C.TABNAME WHERE T.TABSCHEMA=? AND T.TABNAME=? "
                       "AND T.TYPE='T' AND T.CONTROL=' ' AND T.TEMPORALTYPE='N' AND T.STATUS='N' "
                       "AND C.HIDDEN=' ' AND C.GENERATED=' ' AND C.TYPESCHEMA='SYSIBM' "
                       "ORDER BY C.COLNO FETCH FIRST 101 ROWS ONLY", (schema,table), maximum=100)
        usable = [{'name':name,'type':kind} for name,kind in columns if kind in SCALARS and IDENTIFIER.fullmatch(name)]
        if usable:
            result.append({'schema':schema,'name':table,'columns':usable})
    return result


def preview(read, tables, table, mapping, limit):
    if (table not in tables or not isinstance(mapping,dict) or not mapping or 'path' not in mapping
            or set(mapping)-{'path','image_id','datetime','geometry','bands'}
            or set(mapping.values())-{item['name'] for item in table['columns']}
            or type(limit) is not int or not 1 <= limit <= 100):
        raise ValueError('Select mapped columns from a verified Db2 table and a limit of 1–100')
    projections = ','.join(f'LEFT(VARCHAR("{column}"),4097)' for column in mapping.values())
    target = f'"{table["schema"]}"."{table["name"]}"'
    rows = read(f'SELECT {projections} FROM {target} FETCH FIRST {limit} ROWS ONLY',maximum=limit)
    if any(value is not None and len(value)>4096 for row in rows for value in row):
        raise ValueError('A Db2 catalog value exceeds 4096 characters')
    return [dict(zip(mapping,row)) for row in rows]


def main():
    root = Path(sys.argv[1])
    sys.path.insert(0,str(root))
    import ibm_db as db
    config = json.load(sys.stdin)
    try:
        connection = connect(db,root,config)
        try:
            db.autocommit(connection,db.SQL_AUTOCOMMIT_OFF)
            read = reader(db,connection)
            version, grants = verify_contract(read,config['database'],config['username'].upper())
            tables = inventory(read,grants)
            data = (preview(read,tables,config['table'],config['mapping'],config['limit'])
                    if 'table' in config else {'version':version,'tables':tables})
        finally:
            try:db.rollback(connection)
            finally:db.close(connection)
        response = {'data':data}
    except ValueError as exc:
        response = {'error':str(exc)}
    except Exception:
        response = {'error':'Db2 TLS or verified read checks failed'}
    print(json.dumps(response,allow_nan=False))


if __name__ == '__main__':
    main()
