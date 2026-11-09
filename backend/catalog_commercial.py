"""Fixed, bounded SQL Server and Oracle catalog operations in a one-shot client."""
import ipaddress
import json
import re
import socket
import ssl
import subprocess
import sys
from contextlib import closing
from pathlib import Path

CONTRACTS = {
    "sqlserver": "SQL Server 2022. Dedicated SQL login with direct SELECT grants and database VIEW DEFINITION. Fixed reads of ordinary base tables over verified TLS.",
    "sqlserver2025": "SQL Server 2025. Dedicated SQL login with direct SELECT grants and database VIEW DEFINITION. Fixed reads of ordinary base tables over verified TLS.",
    "oracle": "Oracle AI Database 26ai (23.26.2 or newer within 26ai). Dedicated account with CREATE SESSION and direct READ grants. Read-only sessions and transactions over verified TCPS.",
}
SQL_SERVER_MAJORS = {"sqlserver": 16, "sqlserver2025": 17}
SQL_TYPES = {
    "bigint", "int", "smallint", "tinyint", "bit", "decimal", "numeric",
    "float", "real", "money", "smallmoney", "date", "datetime", "datetime2",
    "datetimeoffset", "smalldatetime", "time", "char", "nchar", "varchar", "nvarchar",
    "uniqueidentifier",
}
ORA_TYPES = {
    "CHAR", "NCHAR", "VARCHAR2", "NVARCHAR2", "NUMBER", "FLOAT", "BINARY_FLOAT",
    "BINARY_DOUBLE", "DATE", "TIMESTAMP", "TIMESTAMP WITH TIME ZONE",
    "TIMESTAMP WITH LOCAL TIME ZONE",
}


class ContractError(ValueError):
    """A safe-to-display connector policy failure."""


def request(services, engine, endpoint, credentials, ca_file, **preview):
    from catalog_connections import db_endpoint

    parsed, address, port = db_endpoint(endpoint, "sqlserver" if engine in SQL_SERVER_MAJORS else engine)
    if not ca_file:
        raise ValueError("Choose an approved PEM CA file for this database's TLS certificate")
    trust = services.source(ca_file, {".pem", ".crt"}, 2 * 1024**2)
    user = credentials.get("username")
    password = credentials.get("password")
    if any(not isinstance(value, str) or not value or len(value) > 1024 or "\0" in value
           for value in (user, password)):
        raise ValueError("Enter the existing read-only account and password")
    body = {
        "engine": engine, "host": parsed.hostname, "address": address, "port": port,
        "database": parsed.path.strip("/"), "ca": str(trust),
        "user": user, "password": password, **preview,
    }
    # Credentials go over a private pipe, never on the command line or in logs.
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve())],
            input=json.dumps(body), capture_output=True, text=True, timeout=35,
            encoding="utf-8",
        )
        if result.returncode == 0:
            payload = json.loads(result.stdout)
            if "error" in payload:
                raise ValueError(payload["error"])
            return payload["data"]
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Database inspection exceeded the 35-second limit") from exc
    raise ValueError("Database client failed. Check the verified offline driver component and TLS configuration")


def restrict_network(host, address, port):
    """Pin DNS and deny driver redirects, including attempts before authentication."""
    original = socket.getaddrinfo
    target = ipaddress.ip_address(address)

    def pinned(name, service, *args, **kwargs):
        if name not in (host, address) or str(service) != str(port):
            raise PermissionError("Database redirect outside the approved endpoint")
        return original(address, port, *args, **kwargs)

    def guard(event, args):
        if event == "socket.getaddrinfo" and (args[0] != address or str(args[1]) != str(port)):
            raise PermissionError("Unapproved database DNS lookup")
        if event == "socket.connect":
            peer = args[1]
            if not isinstance(peer, tuple) or ipaddress.ip_address(peer[0]) != target or peer[1] != port:
                raise PermissionError("Database redirect outside the approved endpoint")
        if event in {"socket.sendto", "socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo"}:
            raise PermissionError("Unapproved database discovery")

    socket.getaddrinfo = pinned
    sys.addaudithook(guard)


def sql_connect(config, driver):
    from cryptography import x509

    pytds = driver("pytds")
    driver("OpenSSL")
    from pytds import tls

    def exact_san(cert, name):
        # Require an exact SAN. The upstream matcher also accepts legacy CNs.
        try:
            sans = cert.to_cryptography().extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        except x509.ExtensionNotFound:
            return False
        host = name.decode("ascii")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return host.lower() in {value.lower() for value in sans.get_values_for_type(x509.DNSName)}
        return address in sans.get_values_for_type(x509.IPAddress)

    tls.validate_host = exact_san
    return pytds.connect(
        dsn=config["host"], port=config["port"], database=config["database"],
        user=config["user"], password=config["password"], cafile=config["ca"],
        validate_host=True, enc_login_only=False, autocommit=True, login_timeout=8,
        timeout=6, pooling=False, appname="Geo Intelligence Catalog",
    )


def oracle_connect(config, driver):
    oracle = driver("oracledb")
    oracle.enable_thin_mode()
    context = ssl.create_default_context(cafile=config["ca"])
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    params = oracle.ConnectParams(
        host=config["host"], port=config["port"], protocol="tcps",
        service_name=config["database"], ssl_context=context, ssl_server_dn_match=True,
        tcp_connect_timeout=8, retry_count=0,
    )
    conn = oracle.connect(user=config["user"], password=config["password"], params=params)
    conn.call_timeout = 6000
    return conn


def sql_permissions(cursor, engine, expected_database):
    cursor.execute("SELECT CONVERT(int,SERVERPROPERTY('ProductMajorVersion')), "
                   "CONVERT(varchar(32),SERVERPROPERTY('ProductVersion')), DB_NAME(), USER_NAME(), ORIGINAL_LOGIN(), SUSER_SNAME()")
    major, version, database, user, original, login = cursor.fetchone()
    expected = SQL_SERVER_MAJORS[engine]
    if (major != expected or not isinstance(version, str) or not re.fullmatch(rf"{expected}(?:\.[0-9]+){{3}}", version)
            or database.lower() in {"master", "tempdb", "model", "msdb"}
            or engine == "sqlserver2025" and database != expected_database):
        raise ContractError(f"Choose a SQL Server {2022 if expected == 16 else 2025} user database")
    if user in {"dbo", "guest", "public"} or original != login:
        raise ContractError("Use a dedicated SQL Server catalog reader")
    cursor.execute("SELECT type, name FROM sys.login_token")
    tokens = cursor.fetchall()
    if not any(kind == "SQL LOGIN" and name == login for kind, name in tokens):
        raise ContractError("The SQL Server connector requires a dedicated SQL login")
    if any(name != "public" and (kind != "SQL LOGIN" or name != login) for kind, name in tokens):
        raise ContractError("SQL Server catalog logins must have no additional server roles or groups")
    cursor.execute("SELECT name FROM sys.user_token WHERE type IN ('ROLE','APPLICATION ROLE')")
    if any(name != "public" for (name,) in cursor):
        raise ContractError("Use direct SQL Server SELECT grants without database roles")
    for scope, allowed in (
        ("SERVER", {"CONNECT SQL", "VIEW ANY DATABASE"}),
        ("DATABASE", {
            "CONNECT", "VIEW DEFINITION", "VIEW SECURITY DEFINITION", "VIEW PERFORMANCE DEFINITION",
            "VIEW ANY COLUMN ENCRYPTION KEY DEFINITION", "VIEW ANY COLUMN MASTER KEY DEFINITION",
        }),
    ):
        cursor.execute("SELECT permission_name FROM sys.fn_my_permissions(NULL, %s)", (scope,))
        effective = {name for (name,) in cursor}
        if effective - allowed or scope == "DATABASE" and "VIEW DEFINITION" not in effective:
            raise ContractError("SQL Server requires scoped direct SELECT grants and database VIEW DEFINITION")
    cursor.execute("""
        SELECT TOP (1) 1
        FROM sys.server_permissions p JOIN sys.login_token t ON t.principal_id = p.grantee_principal_id
        WHERE p.state IN ('G', 'W') AND NOT (p.state = 'G' AND (
            (p.class = 100 AND p.permission_name IN ('CONNECT SQL', 'VIEW ANY DATABASE'))
            OR (p.class = 105 AND p.permission_name = 'CONNECT' AND t.name = 'public')))
    """)
    if cursor.fetchone() is not None:
        raise ContractError("SQL Server login has an unexpected server-level grant")
    cursor.execute("""
        SELECT TOP (1) 1
        FROM sys.database_permissions AS p
        WHERE p.grantee_principal_id IN (USER_ID(), DATABASE_PRINCIPAL_ID('public'))
          AND p.state IN ('G', 'W')
          AND NOT (p.state = 'G' AND (
            (p.class = 0 AND p.permission_name IN ('CONNECT', 'VIEW DEFINITION',
                'VIEW ANY COLUMN ENCRYPTION KEY DEFINITION', 'VIEW ANY COLUMN MASTER KEY DEFINITION'))
            OR (p.class = 1 AND p.minor_id = 0 AND p.permission_name = 'SELECT')))
    """)
    if cursor.fetchone() is not None:
        raise ContractError("SQL Server account or public role has an unexpected grant")
    cursor.execute("SELECT TOP (1) 1 FROM sys.schemas WHERE principal_id = USER_ID()")
    if cursor.fetchone() is not None:
        raise ContractError("SQL Server catalog account must not own a schema")
    cursor.execute("SELECT TOP (1) 1 FROM sys.objects WHERE principal_id = USER_ID()")
    if cursor.fetchone() is not None:
        raise ContractError("SQL Server catalog account must not own database objects")
    cursor.execute("SET LOCK_TIMEOUT 3000")
    return version


def oracle_permissions(conn, cursor):
    version = tuple(int(part) for part in conn.version.split(".")[:3])
    if version[:2] != (23, 26) or version < (23, 26, 2):
        raise ContractError("Choose Oracle AI Database 26ai (23.26.2 or newer within 26ai)")
    cursor.execute("ALTER SESSION SET READ_ONLY = TRUE")
    cursor.execute("SET TRANSACTION READ ONLY")
    cursor.execute("SELECT privilege FROM SYS.SESSION_PRIVS")
    if {row[0] for row in cursor} != {"CREATE SESSION"}:
        raise ContractError("Oracle catalog readers require CREATE SESSION as their only system privilege")
    cursor.execute("SELECT 1 FROM SYS.USER_ROLE_PRIVS UNION ALL SELECT 1 FROM SYS.SESSION_ROLES FETCH FIRST 1 ROW ONLY")
    if cursor.fetchone() is not None:
        raise ContractError("Use an Oracle catalog account with direct READ grants and no roles")
    cursor.execute("SELECT 1 FROM SYS.USER_OBJECTS FETCH FIRST 1 ROW ONLY")
    if cursor.fetchone() is not None:
        raise ContractError("The Oracle catalog account must not own schema objects")
    cursor.execute("""
        SELECT 1 FROM SYS.ALL_TAB_PRIVS_RECD p
        LEFT JOIN SYS.ALL_USERS u ON u.username = p.owner
        WHERE (p.grantee = USER AND
               (p.privilege <> 'READ' OR p.grantable <> 'NO' OR p.type <> 'TABLE'
                OR NVL(u.oracle_maintained, 'Y') <> 'N'))
           OR (p.grantee = 'PUBLIC' AND NVL(u.oracle_maintained, 'Y') = 'N')
        FETCH FIRST 1 ROW ONLY
    """)
    if cursor.fetchone() is not None:
        raise ContractError("Oracle requires direct table READ grants without grant option or custom PUBLIC grants")
    cursor.execute("SELECT 1 FROM SYS.ALL_COL_PRIVS WHERE grantee IN (USER, 'PUBLIC') FETCH FIRST 1 ROW ONLY")
    if cursor.fetchone() is not None:
        raise ContractError("Oracle column-level grants are outside the catalog contract")
    cursor.execute("SELECT 1 FROM SYS.USER_SCHEMA_PRIVS UNION ALL SELECT 1 FROM SYS.SESSION_SCHEMA_PRIVS FETCH FIRST 1 ROW ONLY")
    if cursor.fetchone() is not None:
        raise ContractError("Oracle schema privileges are outside the catalog contract")


SQL_INVENTORY = """
SELECT TOP (10001) s.name, t.name, c.name, ty.name
FROM sys.tables t
JOIN sys.schemas s ON s.schema_id = t.schema_id
JOIN sys.columns c ON c.object_id = t.object_id
JOIN sys.types ty ON ty.user_type_id = c.user_type_id
JOIN sys.database_permissions p ON p.class = 1 AND p.major_id = t.object_id
WHERE p.grantee_principal_id = USER_ID() AND p.permission_name = 'SELECT'
  AND p.state = 'G' AND p.minor_id = 0
  AND t.is_ms_shipped = 0 AND t.is_external = 0 AND t.is_filetable = 0
  AND NOT EXISTS (SELECT 1 FROM sys.security_predicates r WHERE r.target_object_id = t.object_id)
  AND c.is_computed = 0 AND c.is_hidden = 0 AND c.encryption_type IS NULL
  AND ty.is_user_defined = 0
ORDER BY s.name, t.name, c.column_id
"""
ORA_INVENTORY = """
SELECT t.owner, t.table_name, c.column_name, c.data_type
FROM SYS.ALL_TABLES t
JOIN SYS.ALL_TAB_COLS c ON c.owner = t.owner AND c.table_name = t.table_name
JOIN SYS.USER_TAB_PRIVS_RECD p ON p.owner = t.owner AND p.table_name = t.table_name
WHERE p.privilege = 'READ' AND p.grantable = 'NO' AND p.type = 'TABLE'
  AND t.temporary = 'N' AND t.nested = 'NO' AND t.secondary = 'N'
  AND NOT EXISTS (SELECT 1 FROM SYS.ALL_EXTERNAL_TABLES e WHERE e.owner = t.owner AND e.table_name = t.table_name)
  AND NOT EXISTS (SELECT 1 FROM SYS.ALL_POLICIES r WHERE r.object_owner = t.owner AND r.object_name = t.table_name)
  AND c.virtual_column = 'NO' AND c.hidden_column = 'NO' AND c.data_type_owner IS NULL
ORDER BY t.owner, t.table_name, c.column_id FETCH FIRST 10001 ROWS ONLY
"""


def inventory(cursor, engine):
    cursor.execute(SQL_INVENTORY if engine in SQL_SERVER_MAJORS else ORA_INVENTORY)
    tables = {}
    allowed = SQL_TYPES if engine in SQL_SERVER_MAJORS else ORA_TYPES
    for i, (schema, table, column, kind) in enumerate(cursor):
        if i == 10000:
            raise ContractError("Catalog schema exceeds the 10,000-column inspection limit")
        # Names remain unambiguous in the shared schema.table UI and safe in SQL.
        if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$# ]{0,127}", name) for name in (schema, table, column)):
            continue
        if re.sub(r"\(\d+\)", "", kind) in allowed:
            tables.setdefault((schema, table), []).append({"name": column, "type": kind})
    if len(tables) > 500:
        raise ContractError("Catalog exposes too many tables. Scope the account more narrowly")
    return [{"schema": schema, "name": table, "columns": cols} for (schema, table), cols in tables.items()]


def preview(cursor, engine, tables, schema, mapping, limit):
    if schema not in tables or set(mapping.values()) - {c["name"] for c in schema["columns"]}:
        raise ContractError("Catalog schema or read privileges changed. Inspect the connection again")
    if not 1 <= limit <= 100:
        raise ContractError("Choose a preview limit from 1 to 100")
    # The identifiers below are exact matches for the freshly checked inventory.
    if engine in SQL_SERVER_MAJORS:
        target = f'[{schema["schema"]}].[{schema["name"]}]'
        fields = ", ".join(f"LEFT(CONVERT(nvarchar(max), [{col}]), 4097)" for col in mapping.values())
        cursor.execute(f"SELECT TOP ({limit}) {fields} FROM {target}")
    else:
        target = f'"{schema["schema"]}"."{schema["name"]}"'
        fields = ", ".join(f'SUBSTR(TO_CHAR("{col}"), 1, 4097)' for col in mapping.values())
        cursor.execute(f"SELECT {fields} FROM {target} FETCH FIRST {limit} ROWS ONLY")
    rows = []
    for row in cursor:
        if any(value is not None and len(value) > 4096 for value in row):
            raise ContractError("A mapped value exceeds 4096 characters. No truncated preview was saved")
        rows.append(dict(zip(mapping, row)))
    return rows


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from catalog_connections import driver

    config = json.load(sys.stdin)
    engine = config["engine"]
    restrict_network(config["host"], config["address"], config["port"])
    try:
        connect = sql_connect if engine in SQL_SERVER_MAJORS else oracle_connect
        with closing(connect(config, driver)) as conn, closing(conn.cursor()) as cursor:
            engine_evidence = {}
            if engine in SQL_SERVER_MAJORS:
                version = sql_permissions(cursor, engine, config["database"])
                if engine == "sqlserver2025":
                    if "schema" in config and version != config.get("expected_version"):
                        raise ContractError("SQL Server build changed. Inspect the connection again")
                    engine_evidence["engine_version"] = version
            else:
                oracle_permissions(conn, cursor)
            tables = inventory(cursor, engine)
            data = (
                preview(cursor, engine, tables, config["schema"], config["mapping"], config["limit"])
                if "schema" in config else {"tables": tables, "read_contract": CONTRACTS[engine], **engine_evidence}
            )
        result = {"data": data}
    except ContractError as exc:
        result = {"error": str(exc)}
    except Exception:
        result = {"error": "Database read failed. Check the supported version, scoped grants, offline driver and trusted TLS certificate"}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
