"""Deterministic contract tests. They do not qualify a native Db2 connection."""
import unittest

from catalog_db2 import inventory, preview, sandbox_profile, verify_contract


class Db2Contract(unittest.TestCase):
    def fake(self, *, authorities=('CONNECT',), grants=(), roles=(), version='Db2 v12.1.3.0'):
        def read(sql, parameters=(), maximum=101):
            if sql.startswith('VALUES'):
                return [('GEOPROOF','READER')]
            if 'ENV_INST_INFO' in sql:
                return [(version,)]
            if 'AUTH_LIST_AUTHORITIES_FOR_AUTHID(SESSION_USER' in sql:
                return [('DBADM','N','N','N','N','N','N','*'),
                        *((item,'Y','N','N','N','N','N','*') for item in authorities)]
            if 'AUTH_LIST_ROLES_FOR_AUTHID(SESSION_USER' in sql:
                return [(item,) for item in roles]
            if 'AUTH_LIST_GROUPS_FOR_AUTHID(SESSION_USER' in sql:
                return [('READERGROUP',)]
            if 'OBJECTOWNERS' in sql or 'COLAUTH' in sql:
                self.assertEqual(parameters,('PUBLIC','READER','READERGROUP'))
                return []
            if 'SYSCAT.ROUTINES' in sql:
                return [('IBM_AUTH_FUNCTION',)]
            if 'SYSIBMADM.PRIVILEGES' in sql:
                self.assertEqual(parameters,('PUBLIC','READER','READERGROUP'))
                self.assertEqual(maximum,1000)
                return list(grants)
            self.fail('Unexpected SQL: '+sql)
        return read

    def test_privileges_fail_closed(self):
        legitimate=('READER','SELECT','N','CATALOG','IMAGES','TABLE')
        workload=('READER','USAGE','N',None,'SYSDEFAULTUSERWORKLOAD','WORKLOAD')
        package=('READER','EXECUTE','N','NULLID','SYSSH200','DB2 PACKAGE')
        self.assertEqual(verify_contract(self.fake(grants=(legitimate,workload,package),version='DB2 v12.1.3.0'), 'GEOPROOF','READER'),
                         ('Db2 v12.1.3.0',{('CATALOG','IMAGES')}))
        for label, options in (
            ('wrong engine', {'version':'Db2 v11.5.9.0'}),
            ('role', {'roles':('READER_ROLE',)}),
            ('write authority', {'authorities':('CONNECT','BINDADD')}),
            ('group update', {'grants':(('READERGROUP','UPDATE','N','CATALOG','IMAGES','TABLE'),)}),
            ('grant option', {'grants':(('READER','SELECT','Y','CATALOG','IMAGES','TABLE'),)}),
            ('unreviewed workload', {'grants':(('READER','USAGE','N',None,'CUSTOM','WORKLOAD'),)}),
            ('unreviewed package', {'grants':(('READER','EXECUTE','N','NULLID','CUSTOM','DB2 PACKAGE'),)}),
            ('public execute', {'grants':(('PUBLIC','EXECUTE','N','CATALOG','MUTATE','PROCEDURE'),)}),
        ):
            with self.subTest(label=label),self.assertRaises(ValueError):
                verify_contract(self.fake(**options),'GEOPROOF','READER')

    def test_preview_uses_only_inventory_identifiers(self):
        def columns(sql, parameters=(), maximum=101):
            self.assertIn("T.TEMPORALTYPE='N'",sql)
            self.assertEqual(parameters,('CATALOG','IMAGES'))
            self.assertEqual(maximum,100)
            return [('PATH','VARCHAR'),('EXPRESSION','XML')]
        tables=inventory(columns,{('CATALOG','IMAGES')})
        table=tables[0]
        def projected(sql, parameters=(), maximum=101):
            self.assertEqual(sql,'SELECT LEFT(VARCHAR("PATH"),4097) FROM "CATALOG"."IMAGES" FETCH FIRST 1 ROWS ONLY')
            self.assertEqual(maximum,1)
            return [('scene.tif',)]
        self.assertEqual(preview(projected,tables,table,{'path':'PATH'},1),[{'path':'scene.tif'}])
        for bad in ('PATH";DELETE','EXPRESSION'):
            with self.assertRaises(ValueError):preview(projected,tables,table,{'path':bad},1)

    def test_private_candidate_network_contract(self):
        self.assertIn('(remote tcp "localhost:50001")',sandbox_profile('127.0.0.1',50001))
        for address in ('10.0.0.1','8.8.8.8','localhost'):
            with self.assertRaises(ValueError):sandbox_profile(address,50001)
