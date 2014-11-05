from AllocModel import AllocModel

from mix.MixModel import MixModel
from mix.DPMixModel import DPMixModel
from mix.HardDPMixModel import HardDPMixModel

from admix.AdmixModel import AdmixModel
from admix.HDPModel import HDPModel
from admix.HDPPE import HDPPE
from admix.HDPFullHard import HDPFullHard
from admix.HDPSoft2Hard import HDPSoft2Hard
from admix.HDPHardMult import HDPHardMult
from admix.HDPRelModel import HDPRelAssortModel

AllocModelConstructorsByName = { \
           'MixModel':MixModel,
           'DPMixModel':DPMixModel,
           'HardDPMixModel':HardDPMixModel,
           'AdmixModel':AdmixModel,
           'HDPModel':HDPModel,
           'HDPPE':HDPPE,
           'HDPFullHard':HDPFullHard,
           'HDPSoft2Hard':HDPSoft2Hard,
           'HDPHardMult':HDPHardMult,
           'HDPRelAssortModel':HDPRelAssortModel,
          }

AllocModelNameSet = set(AllocModelConstructorsByName.keys())

__all__ = list()
for name in AllocModelConstructorsByName:
  __all__.append(name)
