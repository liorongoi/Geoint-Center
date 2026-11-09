"""MariaDB 11.4 catalog access with a verified TLS, SELECT-only contract."""
import re
import socket
from contextlib import contextmanager

from catalog_mysql import preview, tables, verify_grants

READ_CONTRACT = (
    "MariaDB 11.4: verified TLS, direct SELECT grants, no roles or PUBLIC grants, "
    "read-only transaction, bounded ordinary InnoDB base-table reads."
)


def verify_account(query):
    query.execute("SELECT VERSION(), CURRENT_ROLE()")
    version, role = query.fetchone()
    if not re.fullmatch(r"11\.4\.\d+-MariaDB(?:-[A-Za-z0-9.+~_-]+)?", version):
        raise ValueError("This adapter is verified only for MariaDB 11.4")
    if role is not None:
        raise ValueError("MariaDB active roles are outside the verified read-only contract")
    query.execute("SHOW GRANTS FOR PUBLIC")
    if query.fetchall():
        raise ValueError("MariaDB PUBLIC privileges are outside the verified read-only contract")
    query.execute("SHOW GRANTS")
    grants = []
    for grant, in query.fetchall():
        # MariaDB includes authentication and TLS account attributes on USAGE.
        # Only the exact known forms are stripped before checking privileges.
        if grant.upper().startswith("GRANT USAGE ON *.* TO "):
            grant = re.sub(r" REQUIRE SSL$", "", grant, flags=re.I)
            grant = re.sub(r" IDENTIFIED BY PASSWORD '\*[0-9A-F]{40}'$", "", grant, flags=re.I)
        grants.append((grant,))
    verify_grants(grants)


@contextmanager
def session(services, endpoint, credentials, ca_file):
    from catalog_connections import db_endpoint, driver

    if not ca_file or not credentials.get("username"):
        raise ValueError("MariaDB requires a trusted CA file and an existing direct SELECT-only account")
    target, address, port = db_endpoint(endpoint, "mariadb")
    database = target.path.strip("/")
    if database.lower() in {"mysql", "sys", "information_schema", "performance_schema"}:
        raise ValueError("Choose a user catalog database")
    ca = str(services.source(ca_file, {".pem", ".crt"}, 2 * 1024**2))
    mysql = driver("pymysql")
    connection = mysql.connect(
        host=target.hostname, port=port, user=credentials["username"], password=credentials.get("password", ""),
        database=database, charset="utf8mb4", connect_timeout=8, read_timeout=8, write_timeout=5,
        defer_connect=True, local_infile=False, ssl_ca=ca, ssl_verify_cert=True, ssl_verify_identity=True,
    )
    with socket.socket(socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM) as transport:
        try:
            transport.settimeout(8)
            transport.connect((address, port))
            connection.connect(transport)
            with connection.cursor() as query:
                query.execute("SHOW SESSION STATUS LIKE 'Ssl_cipher'")
                cipher = query.fetchone()
                if not cipher or not cipher[1]:
                    raise ValueError("MariaDB requires a verified TLS session")
                verify_account(query)
                query.execute("SET SESSION TRANSACTION READ ONLY")
                query.execute("SET SESSION max_statement_time=5")
                query.execute("SET SESSION lock_wait_timeout=2")
                query.execute("START TRANSACTION READ ONLY")
                query.execute("SELECT @@SESSION.transaction_read_only, @@SESSION.max_statement_time")
                if query.fetchone() != (1, 5.0):
                    raise ValueError("MariaDB did not enforce the required read-only session and statement limit")
            yield connection, database
        finally:
            if connection.open:
                try:
                    connection.rollback()
                finally:
                    connection.close()
