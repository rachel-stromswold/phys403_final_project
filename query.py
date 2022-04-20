from astroquery.sdss import SDSS

query = "select ra,dec,z,zErr \n \
        from SpecObj\n \
        where ra BETWEEN 0 and 3 AND\n \
        dec BETWEEN 0 and 1"

res = SDSS.query_sql(query)
print(res[:5])
