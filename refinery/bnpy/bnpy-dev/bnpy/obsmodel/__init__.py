'''
'''

from DiagGaussObsModel import DiagGaussObsModel
from GaussObsModel import GaussObsModel
from ZMGaussObsModel import ZMGaussObsModel
from MultObsModel import MultObsModel
from BernRelObsModel import BernRelObsModel

ObsModelConstructorsByName = { \
           'DiagGauss':DiagGaussObsModel,
           'Gauss':GaussObsModel,
           'ZMGauss':ZMGaussObsModel,
           'Mult':MultObsModel,
           'BernRel':BernRelObsModel,
          }

ObsModelNameSet = set(ObsModelConstructorsByName.keys())