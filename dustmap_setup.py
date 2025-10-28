import time

from dustmaps.config import config
config['data_dir'] = './dustmaps_data'



'''import dustmaps.edenhofer2023
dustmaps.edenhofer2023.fetch()
time.sleep(1)  # to avoid overloading server'''

import dustmaps.sfd
dustmaps.sfd.fetch()

time.sleep(1)  # to avoid overloading server

import dustmaps.csfd
dustmaps.csfd.fetch()
time.sleep(1)  # to avoid overloading server

import dustmaps.planck
dustmaps.planck.fetch()
time.sleep(1)  # to avoid overloading server

import dustmaps.planck
dustmaps.planck.fetch(which='GNILC')
time.sleep(1)  # to avoid overloading server

import dustmaps.bayestar
dustmaps.bayestar.fetch()

time.sleep(1)  # to avoid overloading server

import dustmaps.iphas
dustmaps.iphas.fetch()
time.sleep(1)  # to avoid overloading server

import dustmaps.marshall
dustmaps.marshall.fetch()
time.sleep(1)  # to avoid overloading server

import dustmaps.chen2014
dustmaps.chen2014.fetch()
time.sleep(1)  # to avoid overloading server

import dustmaps.lenz2017
dustmaps.lenz2017.fetch()
time.sleep(1)  # to avoid overloading server

import dustmaps.pg2010
dustmaps.pg2010.fetch()
time.sleep(1)  # to avoid overloading server

import dustmaps.leike_ensslin_2019
dustmaps.leike_ensslin_2019.fetch()
time.sleep(1)  # to avoid overloading server

import dustmaps.leike2020
dustmaps.leike2020.fetch()
time.sleep(1)  # to avoid overloading server


import dustmaps.gaia_tge
dustmaps.gaia_tge.fetch()
time.sleep(1)  # to avoid overloading server

import dustmaps.decaps
dustmaps.decaps.fetch()
time.sleep(1)  # to avoid overloading server