import matplotlib.pyplot as plt
from astroquery.sdss import SDSS
import pesummary.io

gal_list_fname = 'GW_Events/GW200202_gals.txt'

#specify the center and ranges for the right ascension and declanation (convert from hours to degrees)
ra_center = (9 + 45/60)*15
dec_center = 20
ra_rad = 15*5/60
dec_rad = 5

#generate the SQL query
query ="select ra,dec,z,zErr \n\
from SpecObj\n\
where ra BETWEEN " + str(ra_center-ra_rad) + " and " + str(ra_center+ra_rad) + " AND\n\
dec BETWEEN " + str(dec_center-dec_rad) + " and " + str(dec_center+dec_rad)
print("submitting: \n" + query)

res = SDSS.query_sql(query)
print("data received! saving to " + gal_list_fname)

with open(gal_list_fname, 'w') as file:
    file.write("#ra dec z zErr\n")
    for row in res:
        file.write( "{} {} {} {}\n".format(row['ra'], row['dec'], row['z'], row['zErr']) )

print("finished saving to file")
